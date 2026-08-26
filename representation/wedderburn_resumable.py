from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from audit.data_io import write_json
from audit.run_manifest import sha256_file
from bulk.finite_cover_model import load_action
from representation.gap_job_runner import (
    StageExecutionError,
    StageResult,
    StageTimeoutError,
    atomic_json,
    run_streamed,
)
from representation.wedderburn_exact import cyclotomic_complex


REP_BEGIN_RE = re.compile(r"REP_BEGIN index=(\d+) degree=(\d+)")
GEN_ENTRY_RE = re.compile(
    r"GEN_ENTRY rep=(\d+) generator=(\d+) inverse=(true|false) row=(\d+) col=(\d+) value=(.*)"
)
TRACE_CHECK_RE = re.compile(r"TRACE_CHECK generator=(\d+) equal=(true|false)")


class B07StageFailure(RuntimeError):
    def __init__(self, payload: dict[str, object]):
        super().__init__(json.dumps(payload, sort_keys=True))
        self.payload = payload


def to_cygwin(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive
    if not drive:
        raise ValueError(f"Windows drive is required for GAP/MSYS path conversion: {resolved}")
    return "/cygdrive/" + drive[0].lower() + resolved.as_posix()[2:]


def group_definition(permutations: np.ndarray) -> list[str]:
    lines = ["SetInfoLevel(InfoWarning,0);;", "SizeScreen([1000000,1000000]);;"]
    names: list[str] = []
    for number, permutation in enumerate(permutations[:4], start=1):
        name = f"p{number}"
        names.append(name)
        images = ",".join(str(int(value) + 1) for value in permutation)
        lines.append(f"{name}:=PermList([{images}]);;")
    lines.append(f"B07_GENS:=[{','.join(names)}];; B07_G:=Group(B07_GENS);;")
    return lines


def gap_command(config: dict[str, object], script: Path, workspace: Path | None = None) -> list[str]:
    backend = config["gap_backend"]
    gap = str(backend["gap_binary_cygwin"])
    arguments = [shlex.quote(gap), "-b", "-q"]
    if workspace is not None:
        arguments.extend(["-L", shlex.quote(to_cygwin(workspace))])
    arguments.append(shlex.quote(to_cygwin(script)))
    return [str(backend["gap_bash"]), "--login", "-c", " ".join(arguments)]


def _attempt_number(log_dir: Path, stage: str) -> int:
    return len(list(log_dir.glob(f"{stage}_attempt_*.heartbeat.json"))) + 1


def _checkpoint_reader(state_path: Path):
    def read() -> dict[str, object]:
        if not state_path.exists():
            return {"last_completed_irrep": 0, "last_completed_block": 0}
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return {
            "last_completed_irrep": int(state.get("last_completed_irrep", 0)),
            "last_completed_block": int(state.get("last_completed_block", 0)),
        }

    return read


def _execute_gap_stage(
    *,
    config: dict[str, object],
    script_text: str,
    stage: str,
    group_name: str,
    log_dir: Path,
    state_path: Path,
    timeout_seconds: int,
    workspace: Path | None = None,
    maximum_job_memory_bytes: int | None = None,
) -> StageResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    attempt = _attempt_number(log_dir, stage)
    prefix = f"{stage}_attempt_{attempt:03d}"
    script = log_dir / f"{prefix}.g"
    stdout = log_dir / f"{prefix}.stdout.log"
    stderr = log_dir / f"{prefix}.stderr.log"
    heartbeat = log_dir / f"{prefix}.heartbeat.json"
    script.write_text(script_text, encoding="ascii")
    result = run_streamed(
        gap_command(config, script, workspace),
        stdout_path=stdout,
        stderr_path=stderr,
        heartbeat_path=heartbeat,
        timeout_seconds=timeout_seconds,
        heartbeat_seconds=float(config["gap_backend"].get("heartbeat_seconds", 10)),
        stage_metadata={"task_id": "B-07", "stage": stage, "group": group_name, "attempt": attempt},
        progress_reader=_checkpoint_reader(state_path),
        maximum_job_memory_bytes=maximum_job_memory_bytes,
    )
    history = log_dir / "stage_history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "group": group_name, "attempt": attempt, **result.to_dict()}, sort_keys=True) + "\n")
    return result


def _stage_timeout(config: dict[str, object], stage: str) -> int:
    timeouts = config["gap_backend"]["stage_timeouts_seconds"]
    return int(timeouts[stage])


def _state_update(state_path: Path, **updates: object) -> dict[str, object]:
    state: dict[str, object] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(updates)
    atomic_json(state_path, state)
    return state


def _parse_scalar(log: str, key: str) -> str:
    matches = re.findall(rf"^{re.escape(key)}=(.*)$", log, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"missing GAP marker {key}")
    return matches[-1].strip()


def _group_audit(
    permutations: np.ndarray,
    metadata: dict[str, object],
    action_hash: str,
    raw_dir: Path,
    log_dir: Path,
    state_path: Path,
    config: dict[str, object],
) -> tuple[dict[str, object], StageResult | None]:
    audit_path = raw_dir / "group_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("action_sha256") != action_hash or audit.get("status") != "COMPLETE":
            raise RuntimeError("existing group audit is incompatible with the current action")
        return audit, None
    script = group_definition(permutations)
    script.extend(
        [
            'Print("STAGE_BEGIN=group_audit\\n");',
            'Print("GROUP_ORDER=",Size(B07_G),"\\n");',
            'Print("GENERATOR_COUNT=",Length(B07_GENS),"\\n");',
            'Print("GENERATOR_ORDERS=",List(B07_GENS,Order),"\\n");',
            'Print("MOVED_POINTS=",LargestMovedPoint(B07_G),"\\n");',
            'Print("STAGE_COMPLETE=group_audit\\n");',
            "QUIT;",
        ]
    )
    result = _execute_gap_stage(
        config=config,
        script_text="\n".join(script) + "\n",
        stage="group_audit",
        group_name=str(metadata["tower_id"]) + f"_L{metadata['level']}",
        log_dir=log_dir,
        state_path=state_path,
        timeout_seconds=_stage_timeout(config, "group_audit"),
    )
    log = Path(result.stdout_log).read_text(encoding="utf-8", errors="replace")
    audit = {
        **metadata,
        "status": "COMPLETE",
        "action_sha256": action_hash,
        "computed_order": int(_parse_scalar(log, "GROUP_ORDER")),
        "generator_count": int(_parse_scalar(log, "GENERATOR_COUNT")),
        "generator_orders": ast.literal_eval(_parse_scalar(log, "GENERATOR_ORDERS")),
        "moved_points": int(_parse_scalar(log, "MOVED_POINTS")),
        "profile": result.to_dict(),
    }
    atomic_json(audit_path, audit)
    _state_update(state_path, current_stage="character_table", group_audit="COMPLETE")
    return audit, result


def _character_table(
    permutations: np.ndarray,
    metadata: dict[str, object],
    action_hash: str,
    raw_dir: Path,
    log_dir: Path,
    state_path: Path,
    config: dict[str, object],
) -> tuple[dict[str, object], Path, StageResult | None]:
    metadata_path = raw_dir / "character_table.json"
    workspace = raw_dir / "character_table.workspace"
    if metadata_path.exists() and workspace.exists():
        table = json.loads(metadata_path.read_text(encoding="utf-8"))
        if table.get("action_sha256") != action_hash or table.get("status") != "COMPLETE":
            raise RuntimeError("existing character table is incompatible with the current action")
        return table, workspace, None
    attempt = _attempt_number(log_dir, "character_table")
    attempt_workspace = raw_dir / f"character_table_attempt_{attempt:03d}.workspace"
    script = group_definition(permutations)
    script.extend(
        [
            f'B07_ACTION_SHA:="{action_hash}";;',
            'Print("STAGE_BEGIN=character_table\\n");',
            "B07_CHAR_START:=Runtime();;",
            "B07_IRR:=Irr(B07_G);;",
            'Print("CHAR_TABLE_MS=",Runtime()-B07_CHAR_START,"\\n");',
            'Print("CHAR_COUNT=",Length(B07_IRR),"\\n");',
            'Print("DEGREES=",List(B07_IRR,x->x[1]),"\\n");',
            'Print("SUM_SQUARES=",Sum(B07_IRR,x->x[1]^2),"\\n");',
            'Print("STAGE_PROGRESS=character_table_ready\\n");',
            f'SaveWorkspace("{to_cygwin(attempt_workspace)}");',
            'Print("STAGE_COMPLETE=character_table\\n");',
            "QUIT;",
        ]
    )
    result = _execute_gap_stage(
        config=config,
        script_text="\n".join(script) + "\n",
        stage="character_table",
        group_name=str(metadata["tower_id"]) + f"_L{metadata['level']}",
        log_dir=log_dir,
        state_path=state_path,
        timeout_seconds=_stage_timeout(config, "character_table"),
    )
    log = Path(result.stdout_log).read_text(encoding="utf-8", errors="replace")
    if "STAGE_COMPLETE=character_table" not in log or not attempt_workspace.exists():
        raise RuntimeError("character-table stage did not produce a complete workspace")
    os.replace(attempt_workspace, workspace)
    degrees = [int(value) for value in ast.literal_eval(_parse_scalar(log, "DEGREES"))]
    table = {
        **metadata,
        "status": "COMPLETE",
        "action_sha256": action_hash,
        "character_count": int(_parse_scalar(log, "CHAR_COUNT")),
        "degrees": degrees,
        "sum_degree_squares": int(_parse_scalar(log, "SUM_SQUARES")),
        "character_table_elapsed_seconds": int(_parse_scalar(log, "CHAR_TABLE_MS")) / 1000.0,
        "workspace": workspace.name,
        "workspace_sha256": sha256_file(workspace),
        "profile": result.to_dict(),
    }
    atomic_json(metadata_path, table)
    _state_update(
        state_path,
        current_stage="individual_irreps",
        character_table="COMPLETE",
        total_irreps=len(degrees),
    )
    return table, workspace, result


def _irrep_script(index: int, target: Path) -> str:
    return "\n".join(
        [
            "SetInfoLevel(InfoWarning,0);;",
            "SetInfoLevel(InfoCharacterTable,1);;",
            "SizeScreen([1000000,1000000]);;",
            f"idx:={index};;",
            'Print("STAGE_BEGIN=individual_irrep\\n");',
            'Print("IRREP_INDEX=",idx,"\\n");',
            'Print("IRREP_DEGREE=",B07_IRR[idx][1],"\\n");',
            "irrepStart:=Runtime();;",
            "rep:=IrreducibleRepresentationsDixon(B07_G,B07_IRR[idx]);;",
            'if rep=fail then Error("Dixon returned fail"); fi;',
            'Print("IRREP_CONSTRUCTION_MS=",Runtime()-irrepStart,"\\n");',
            "d:=DimensionOfMatrixGroup(Image(rep));;",
            f'out:=OutputTextFile("{to_cygwin(target)}",false);;',
            "SetPrintFormattingStatus(out,false);;",
            'PrintTo(out,"REP_BEGIN index=",idx," degree=",d,"\\n");',
            "for g in [1..Length(B07_GENS)] do",
            "  mf:=Image(rep,B07_GENS[g]);; mi:=Image(rep,B07_GENS[g]^-1);;",
            "  expected:=B07_IRR[idx][PositionProperty(ConjugacyClasses(B07_G),c->B07_GENS[g] in c)];;",
            '  PrintTo(out,"TRACE_CHECK generator=",g," equal=",TraceMat(mf)=expected,"\\n");',
            "  for r in [1..d] do for c in [1..d] do",
            "    if not IsZero(mf[r][c]) then",
            '      PrintTo(out,"GEN_ENTRY rep=",idx," generator=",g," inverse=false row=",r," col=",c," value=",String(mf[r][c]),"\\n");',
            "    fi;",
            "    if not IsZero(mi[r][c]) then",
            '      PrintTo(out,"GEN_ENTRY rep=",idx," generator=",g," inverse=true row=",r," col=",c," value=",String(mi[r][c]),"\\n");',
            "    fi;",
            "  od; od;",
            "od;",
            'PrintTo(out,"REP_END\\n");',
            "CloseStream(out);;",
            'Print("STAGE_COMPLETE=individual_irrep\\n");',
            "QUIT;",
            "",
        ]
    )


def _valid_irrep(path: Path, index: int) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return f"REP_BEGIN index={index} " in text and text.rstrip().endswith("REP_END")


def _build_block(irrep_path: Path, block_path: Path, index: int, action_hash: str) -> dict[str, object]:
    text = irrep_path.read_text(encoding="utf-8")
    begin = next((REP_BEGIN_RE.fullmatch(line.strip()) for line in text.splitlines() if REP_BEGIN_RE.fullmatch(line.strip())), None)
    if begin is None or int(begin.group(1)) != index:
        raise ValueError(f"invalid irrep header in {irrep_path}")
    degree = int(begin.group(2))
    matrices = {(generator, inverse): np.zeros((degree, degree), dtype=np.complex128) for generator in range(1, 5) for inverse in (False, True)}
    trace_checks: list[bool] = []
    entry_count = 0
    for line in text.splitlines():
        entry = GEN_ENTRY_RE.fullmatch(line.strip())
        if entry:
            rep_index = int(entry.group(1))
            if rep_index != index:
                raise ValueError("irrep entry index mismatch")
            generator = int(entry.group(2))
            inverse = entry.group(3) == "true"
            row, column = int(entry.group(4)) - 1, int(entry.group(5)) - 1
            matrices[(generator, inverse)][row, column] = cyclotomic_complex(entry.group(6))
            entry_count += 1
        trace = TRACE_CHECK_RE.fullmatch(line.strip())
        if trace:
            trace_checks.append(trace.group(2) == "true")
    if len(trace_checks) != 4 or not all(trace_checks):
        raise ArithmeticError(f"character trace check failed for irrep {index}")
    adjacency = sum(matrices.values(), np.zeros((degree, degree), dtype=np.complex128))
    eigenvalues = np.linalg.eigvals(adjacency)
    imaginary_residual = float(np.max(np.abs(eigenvalues.imag)))
    if imaginary_residual > 2.0e-7:
        raise ArithmeticError("non-real adjacency block eigenvalue exceeds tolerance")
    real_values = np.sort(eigenvalues.real)
    characteristic_residual = float(np.max(np.abs(np.polyval(np.poly(real_values), real_values))))
    temporary = block_path.with_name(block_path.name + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["status"] = "COMPLETE"
        handle.attrs["rep_index"] = index
        handle.attrs["degree"] = degree
        handle.attrs["action_sha256"] = action_hash
        handle.attrs["exact_irrep_sha256"] = sha256_file(irrep_path)
        handle.attrs["exact_entry_count"] = entry_count
        handle.attrs["trace_checks_passed"] = True
        handle.attrs["imaginary_residual"] = imaginary_residual
        handle.attrs["characteristic_residual"] = characteristic_residual
        for generator in range(1, 5):
            handle.create_dataset(f"generator_{generator}", data=matrices[(generator, False)])
            handle.create_dataset(f"generator_{generator}_inverse", data=matrices[(generator, True)])
        handle.create_dataset("adjacency_matrix", data=adjacency)
        handle.create_dataset("adjacency_eigenvalues", data=real_values)
    os.replace(temporary, block_path)
    return {
        "rep_index": index,
        "degree": degree,
        "entry_count": entry_count,
        "imaginary_residual": imaginary_residual,
        "characteristic_residual": characteristic_residual,
        "irrep_file": irrep_path.name,
        "irrep_sha256": sha256_file(irrep_path),
        "block_file": block_path.name,
        "block_sha256": sha256_file(block_path),
    }


def _valid_block(path: Path, index: int, action_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return (
                handle.attrs.get("status") == "COMPLETE"
                and int(handle.attrs.get("rep_index", -1)) == index
                and handle.attrs.get("action_sha256") == action_hash
                and "adjacency_eigenvalues" in handle
            )
    except OSError:
        return False


def _irreps_and_blocks(
    *,
    metadata: dict[str, object],
    action_hash: str,
    table: dict[str, object],
    workspace: Path,
    raw_dir: Path,
    log_dir: Path,
    state_path: Path,
    config: dict[str, object],
) -> tuple[list[dict[str, object]], list[StageResult]]:
    irrep_dir = raw_dir / "irreps"
    block_dir = raw_dir / "blocks"
    irrep_dir.mkdir(parents=True, exist_ok=True)
    block_dir.mkdir(parents=True, exist_ok=True)
    profiles: list[StageResult] = []
    records: list[dict[str, object]] = []
    degrees = [int(value) for value in table["degrees"]]
    for index, expected_degree in enumerate(degrees, start=1):
        irrep_path = irrep_dir / f"irrep_{index:04d}.txt"
        block_path = block_dir / f"block_{index:04d}.h5"
        if not _valid_irrep(irrep_path, index):
            part = irrep_dir / f"irrep_{index:04d}_attempt_{_attempt_number(log_dir, f'irrep_{index:04d}'):03d}.part"
            result = _execute_gap_stage(
                config=config,
                script_text=_irrep_script(index, part),
                stage=f"irrep_{index:04d}",
                group_name=str(metadata["tower_id"]) + f"_L{metadata['level']}",
                log_dir=log_dir,
                state_path=state_path,
                timeout_seconds=_stage_timeout(config, "individual_irrep"),
                workspace=workspace,
            )
            profiles.append(result)
            if not _valid_irrep(part, index):
                raise RuntimeError(f"GAP completed without a valid exact irrep artifact for index {index}")
            os.replace(part, irrep_path)
            _state_update(state_path, current_stage="individual_irreps", last_completed_irrep=index)
        if not _valid_block(block_path, index, action_hash):
            record = _build_block(irrep_path, block_path, index, action_hash)
            if record["degree"] != expected_degree:
                raise ArithmeticError(f"degree mismatch for irrep {index}")
            _state_update(state_path, current_stage="hamiltonian_blocks", last_completed_block=index)
        with h5py.File(block_path, "r") as handle:
            records.append(
                {
                    "rep_index": index,
                    "degree": int(handle.attrs["degree"]),
                    "entry_count": int(handle.attrs["exact_entry_count"]),
                    "imaginary_residual": float(handle.attrs["imaginary_residual"]),
                    "characteristic_residual": float(handle.attrs["characteristic_residual"]),
                    "irrep_file": irrep_path.name,
                    "irrep_sha256": str(handle.attrs["exact_irrep_sha256"]),
                    "block_file": block_path.name,
                    "block_sha256": sha256_file(block_path),
                }
            )
    _state_update(
        state_path,
        current_stage="complete",
        last_completed_irrep=len(degrees),
        last_completed_block=len(degrees),
        status="COMPLETE",
    )
    return records, profiles


def _failure_payload(
    *,
    run_id: str,
    action_path: Path,
    metadata: dict[str, object],
    state_path: Path,
    error: Exception,
) -> dict[str, object]:
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    result: StageResult | None = error.result if isinstance(error, StageExecutionError) else None
    return {
        "task_id": "B-07",
        "run_id": run_id,
        "status": "FAIL_IMPLEMENTATION",
        "exact_group": action_path.stem,
        "tower_id": metadata.get("tower_id"),
        "level": metadata.get("level"),
        "computation_stage": state.get("current_stage", "unknown"),
        "elapsed_seconds": result.elapsed_seconds if result else None,
        "peak_memory_bytes": result.peak_job_memory_bytes if result else None,
        "last_completed_irreducible_block": int(state.get("last_completed_block", 0)),
        "last_completed_irrep": int(state.get("last_completed_irrep", 0)),
        "timeout_tree_terminated": result.tree_terminated if result else None,
        "last_stdout_line": result.last_stdout_line if result else "",
        "last_stderr_line": result.last_stderr_line if result else "",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def prepare_wedderburn(
    root: Path, run_dir: Path, run_id: str, config: dict[str, object]
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, Path]]:
    gate_dir = root / str(config["tower_gate"]["raw_directory"])
    raw_root = run_dir / "raw" / "representation"
    log_root = run_dir / "logs" / "gap_irreps"
    raw_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    reference_bound = 8.0 * float(config["reference_adjacency"]["markov_spectral_radius_upper"])
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for action_path in sorted(gate_dir.glob("*.npz")):
        permutations, metadata = load_action(action_path)
        action_hash = sha256_file(action_path)
        group_raw = raw_root / action_path.stem
        group_logs = log_root / action_path.stem
        group_raw.mkdir(parents=True, exist_ok=True)
        group_logs.mkdir(parents=True, exist_ok=True)
        state_path = group_raw / "stage_state.json"
        _state_update(
            state_path,
            task_id="B-07",
            run_id=run_id,
            action=action_path.name,
            action_sha256=action_hash,
            tower_id=metadata["tower_id"],
            level=metadata["level"],
            current_stage="group_audit",
            last_completed_irrep=0,
            last_completed_block=0,
        )
        try:
            audit, audit_profile = _group_audit(
                permutations, metadata, action_hash, group_raw, group_logs, state_path, config
            )
            table, workspace, table_profile = _character_table(
                permutations, metadata, action_hash, group_raw, group_logs, state_path, config
            )
            block_records, irrep_profiles = _irreps_and_blocks(
                metadata=metadata,
                action_hash=action_hash,
                table=table,
                workspace=workspace,
                raw_dir=group_raw,
                log_dir=group_logs,
                state_path=state_path,
                config=config,
            )
        except Exception as error:
            failure = _failure_payload(
                run_id=run_id,
                action_path=action_path,
                metadata=metadata,
                state_path=state_path,
                error=error,
            )
            failure_path = run_dir / "certificates" / "b07_failure.json"
            atomic_json(failure_path, failure)
            _state_update(state_path, status="FAIL_IMPLEMENTATION", failure_certificate=str(failure_path))
            raise B07StageFailure(failure) from error
        profiles = [profile.to_dict() for profile in [audit_profile, table_profile] if profile is not None]
        profiles.extend(profile.to_dict() for profile in irrep_profiles)
        diagnostic = {
            **metadata,
            "action_sha256": action_hash,
            "order": int(audit["computed_order"]),
            "representation_count": int(table["character_count"]),
            "sum_degree_squares": int(table["sum_degree_squares"]),
            "degree_square_identity": int(table["sum_degree_squares"]) == int(audit["computed_order"]),
            "character_table_elapsed_seconds": float(table["character_table_elapsed_seconds"]),
            "stage_profiles": profiles,
            "last_completed_irrep": len(block_records),
            "last_completed_block": len(block_records),
            "raw_group_directory": group_raw.relative_to(run_dir).as_posix(),
        }
        diagnostics.append(diagnostic)
        block_dir = group_raw / "blocks"
        for record in block_records:
            with h5py.File(block_dir / str(record["block_file"]), "r") as handle:
                eigenvalues = np.asarray(handle["adjacency_eigenvalues"])
            degree = int(record["degree"])
            for eigen_index, eigenvalue in enumerate(eigenvalues):
                rows.append(
                    {
                        "tower_id": metadata["tower_id"],
                        "level": metadata["level"],
                        "quotient_order": diagnostic["order"],
                        "rep_index": int(record["rep_index"]),
                        "degree": degree,
                        "block_eigen_index": eigen_index,
                        "adjacency_eigenvalue": float(eigenvalue),
                        "regular_multiplicity": degree,
                        "retained_operator_tempered": abs(float(eigenvalue)) <= reference_bound + 2.0e-8,
                        "imaginary_residual": float(record["imaginary_residual"]),
                        "characteristic_residual": float(record["characteristic_residual"]),
                        "raw_irrep_sha256": record["irrep_sha256"],
                        "raw_block_sha256": record["block_sha256"],
                    }
                )
    frame = pd.DataFrame(rows)
    derived = run_dir / "derived" / "wedderburn_block_spectra.parquet"
    frame.to_parquet(derived, index=False)
    diagnostic_path = run_dir / "certificates" / "wedderburn_precompute.json"
    write_json(
        diagnostic_path,
        {
            "task_id": "B-07",
            "run_id": run_id,
            "status": "PRECOMPUTED",
            "backend": "native Windows GAP supervised by Job Object",
            "streaming_logs": True,
            "heartbeat_metadata": True,
            "complete_tree_timeout": True,
            "resumable_stages": ["group_audit", "character_table", "individual_irreps", "hamiltonian_blocks"],
            "diagnostics": diagnostics,
            "reference_adjacency_bound": reference_bound,
        },
    )
    return frame, diagnostics, {"raw": raw_root, "derived": derived, "certificate": diagnostic_path}


def run(config: dict[str, object], run_dir: Path, run_id: str, root: Path, context: dict[str, object]):
    frame: pd.DataFrame = context["blocks"]
    diagnostics: list[dict[str, object]] = context["wedderburn_diagnostics"]
    records: list[dict[str, object]] = []
    passed = True
    for diagnostic in diagnostics:
        subset = frame[(frame.tower_id == diagnostic["tower_id"]) & (frame.level == diagnostic["level"])]
        irreps = subset.drop_duplicates("rep_index")
        trace0 = int(sum(int(row.degree) ** 2 for row in irreps.itertuples()))
        trace1 = float(np.sum(subset.adjacency_eigenvalue * subset.regular_multiplicity))
        trace2 = float(np.sum(subset.adjacency_eigenvalue**2 * subset.regular_multiplicity))
        expected_trace2 = 8.0 * int(diagnostic["order"])
        record = {
            **diagnostic,
            "dimension_recombination": trace0,
            "trace_adjacency": trace1,
            "trace_adjacency_squared": trace2,
            "expected_trace_adjacency_squared": expected_trace2,
            "trace2_residual": abs(trace2 - expected_trace2),
        }
        record_pass = (
            bool(diagnostic["degree_square_identity"])
            and trace0 == int(diagnostic["order"])
            and abs(trace1) < 2.0e-6 * int(diagnostic["order"])
            and abs(trace2 - expected_trace2) < 2.0e-6 * expected_trace2
        )
        record["passed"] = record_pass
        passed = passed and record_pass
        records.append(record)
    certificate = run_dir / "certificates" / "b07_wedderburn_exact.json"
    payload: dict[str, object] = {
        "task_id": "B-07",
        "run_id": run_id,
        "status": "PASS_CERTIFIED" if passed else "FAIL_IMPLEMENTATION",
        "quotients": records,
        "fourier_recombination_theorem": "Spec Reg(A)=union_rho Spec(sum_s rho(s)), each block eigenvalue repeated dim(rho)",
        "complete": True,
        "all_irreps_persisted_individually": True,
        "restart_supported": True,
    }
    metadata = context.get("certificate_metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("B-07 certificate_metadata must be a mapping")
    protected = sorted(set(payload).intersection(metadata))
    if protected:
        raise ValueError(f"B-07 certificate_metadata overrides protected fields: {protected}")
    payload.update(metadata)
    write_json(certificate, payload)
    return ("PASS_CERTIFIED" if passed else "FAIL_IMPLEMENTATION"), {
        "raw": context["wedderburn_outputs"]["raw"],
        "derived": context["wedderburn_outputs"]["derived"],
        "certificate": certificate,
    }
