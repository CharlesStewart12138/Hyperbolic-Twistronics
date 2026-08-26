from __future__ import annotations

import ast
import json
import math
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from audit.data_io import write_json
from bulk.finite_cover_model import load_action


BEGIN_RE = re.compile(r"REP_BEGIN index=(\d+) degree=(\d+)")
ENTRY_RE = re.compile(r"ENTRY rep=(\d+) row=(\d+) col=(\d+) value=(.*)")


def cyclotomic_complex(text: str) -> complex:
    expression = text.strip().replace("^", "**")
    expression = re.sub(r"E\((\d+)\)", r"z(\1)", expression)
    if not re.fullmatch(r"[0-9z()+\-*/. ]+", expression):
        raise ValueError(f"unsupported GAP cyclotomic expression: {text}")
    return complex(
        eval(
            expression,
            {"__builtins__": {}},
            {"z": lambda n: np.exp(2j * np.pi / int(n))},
        )
    )


def gap_script(permutations: np.ndarray) -> str:
    lines = ["SetInfoLevel(InfoWarning,0);;", "SizeScreen([1000000,1000000]);;"]
    names = []
    for number, permutation in enumerate(permutations[:4], start=1):
        name = f"p{number}"
        names.append(name)
        images = ",".join(str(int(value) + 1) for value in permutation)
        lines.append(f"{name}:=PermList([{images}]);;")
    lines.extend(
        [
            f"gens:=[{','.join(names)}];; G:=Group(gens);;",
            'Print("GROUP_ORDER=",Size(G),"\\n");',
            "reps:=IrreducibleRepresentationsDixon(G);;",
            'Print("REP_COUNT=",Length(reps),"\\n");',
            "for i in [1..Length(reps)] do",
            "  d:=DimensionOfMatrixGroup(Image(reps[i]));;",
            '  Print("REP_BEGIN index=",i," degree=",d,"\\n");',
            "  adj:=Sum(List(gens,g->Image(reps[i],g)+Image(reps[i],g^-1)));;",
            "  for r in [1..d] do for c in [1..d] do",
            "    if not IsZero(adj[r][c]) then",
            '      Print("ENTRY rep=",i," row=",r," col=",c," value=",String(adj[r][c]),"\\n");',
            "    fi;",
            "  od; od;",
            '  Print("REP_END\\n");',
            "od;",
            'Print("SUM_SQUARES=",Sum(reps,r->DimensionOfMatrixGroup(Image(r))^2),"\\n");',
            "QUIT;",
        ]
    )
    return "\n".join(lines) + "\n"


def run_gap_export(
    action_path: Path, output_text: Path, input_script: Path, config: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    permutations, metadata = load_action(action_path)
    input_script.write_text(gap_script(permutations), encoding="ascii")
    backend = config["gap_backend"]
    cyg_script = "/cygdrive/" + input_script.drive[0].lower() + input_script.as_posix()[2:]
    command = [
        str(backend["gap_bash"]),
        "--login",
        "-c",
        f'"{backend["gap_binary_cygwin"]}" -b -q < "{cyg_script}"',
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=int(backend["timeout_seconds"]),
    )
    output_text.write_text(completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or "SUM_SQUARES=" not in completed.stdout:
        raise RuntimeError(f"GAP irreducible representation export failed for {action_path.name}")
    degrees: dict[int, int] = {}
    entries: dict[int, list[tuple[int, int, complex]]] = {}
    for line in completed.stdout.splitlines():
        begin = BEGIN_RE.fullmatch(line.strip())
        if begin:
            index, degree = int(begin.group(1)), int(begin.group(2))
            degrees[index] = degree
            entries[index] = []
            continue
        entry = ENTRY_RE.fullmatch(line.strip())
        if entry:
            index, row, column = map(int, entry.groups()[:3])
            entries[index].append((row - 1, column - 1, cyclotomic_complex(entry.group(4))))
    blocks: list[dict[str, object]] = []
    for index in sorted(degrees):
        degree = degrees[index]
        matrix = np.zeros((degree, degree), dtype=np.complex128)
        for row, column, value in entries[index]:
            matrix[row, column] = value
        eigenvalues = np.linalg.eigvals(matrix)
        imaginary_residual = float(np.max(np.abs(eigenvalues.imag)))
        if imaginary_residual > 2.0e-7:
            raise ArithmeticError("non-real adjacency block eigenvalue exceeds tolerance")
        real_values = np.sort(eigenvalues.real)
        characteristic_residual = float(
            np.max(np.abs(np.polyval(np.poly(real_values), real_values)))
        )
        blocks.append(
            {
                "rep_index": index,
                "degree": degree,
                "matrix": matrix,
                "eigenvalues": real_values,
                "imaginary_residual": imaginary_residual,
                "characteristic_residual": characteristic_residual,
            }
        )
    diagnostics = {
        **metadata,
        "gap_returncode": completed.returncode,
        "representation_count": len(blocks),
        "sum_degree_squares": sum(int(block["degree"]) ** 2 for block in blocks),
        "gap_output": output_text.name,
    }
    return blocks, diagnostics


def prepare_wedderburn(
    root: Path, run_dir: Path, run_id: str, config: dict[str, object]
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, Path]]:
    gate_dir = root / str(config["tower_gate"]["raw_directory"])
    raw_dir = run_dir / "raw" / "representation"
    log_dir = run_dir / "logs" / "gap_irreps"
    raw_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    reference_bound = 8.0 * float(config["reference_adjacency"]["markov_spectral_radius_upper"])
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for action_path in sorted(gate_dir.glob("*.npz")):
        output_text = raw_dir / f"{action_path.stem}_gap_irreps.txt"
        input_script = log_dir / f"{action_path.stem}_input.g"
        blocks, diagnostic = run_gap_export(action_path, output_text, input_script, config)
        h5_path = raw_dir / f"{action_path.stem}_adjacency_blocks.h5"
        with h5py.File(h5_path, "w") as handle:
            handle.attrs["run_id"] = run_id
            handle.attrs["task_id"] = "B-07"
            handle.attrs["tower_id"] = diagnostic["tower_id"]
            handle.attrs["level"] = diagnostic["level"]
            for block in blocks:
                group = handle.create_group(f"rep_{int(block['rep_index']):04d}")
                group.attrs["degree"] = int(block["degree"])
                group.create_dataset("adjacency_matrix", data=block["matrix"])
                group.create_dataset("adjacency_eigenvalues", data=block["eigenvalues"])
        for block in blocks:
            degree = int(block["degree"])
            for eigen_index, eigenvalue in enumerate(block["eigenvalues"]):
                rows.append(
                    {
                        "tower_id": diagnostic["tower_id"],
                        "level": diagnostic["level"],
                        "quotient_order": diagnostic["order"],
                        "rep_index": int(block["rep_index"]),
                        "degree": degree,
                        "block_eigen_index": eigen_index,
                        "adjacency_eigenvalue": float(eigenvalue),
                        "regular_multiplicity": degree,
                        "retained_operator_tempered": abs(float(eigenvalue)) <= reference_bound + 2.0e-8,
                        "imaginary_residual": block["imaginary_residual"],
                        "characteristic_residual": block["characteristic_residual"],
                    }
                )
        diagnostic["block_h5"] = h5_path.name
        diagnostic["degree_square_identity"] = diagnostic["sum_degree_squares"] == diagnostic["order"]
        diagnostics.append(diagnostic)
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
            "diagnostics": diagnostics,
            "reference_adjacency_bound": reference_bound,
        },
    )
    return frame, diagnostics, {"raw": raw_dir, "derived": derived, "certificate": diagnostic_path}


def run(config: dict[str, object], run_dir: Path, run_id: str, root: Path, context: dict[str, object]):
    frame: pd.DataFrame = context["blocks"]
    diagnostics: list[dict[str, object]] = context["wedderburn_diagnostics"]
    records = []
    passed = True
    for diagnostic in diagnostics:
        subset = frame[
            (frame.tower_id == diagnostic["tower_id"]) & (frame.level == diagnostic["level"])
        ]
        moment0 = int(np.sum(subset.degree * subset.regular_multiplicity / subset.degree))
        trace0 = int(sum(int(row.degree) ** 2 for row in subset.drop_duplicates("rep_index").itertuples()))
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
            diagnostic["degree_square_identity"]
            and trace0 == int(diagnostic["order"])
            and abs(trace1) < 2.0e-6 * diagnostic["order"]
            and abs(trace2 - expected_trace2) < 2.0e-6 * expected_trace2
        )
        record["passed"] = record_pass
        passed = passed and record_pass
        records.append(record)
    certificate = run_dir / "certificates" / "b07_wedderburn_exact.json"
    write_json(
        certificate,
        {
            "task_id": "B-07",
            "run_id": run_id,
            "status": "PASS_CERTIFIED" if passed else "FAIL_IMPLEMENTATION",
            "quotients": records,
            "fourier_recombination_theorem": "Spec Reg(A)=union_rho Spec(sum_s rho(s)), each block eigenvalue repeated dim(rho)",
            "complete": True,
        },
    )
    return ("PASS_CERTIFIED" if passed else "FAIL_IMPLEMENTATION"), {
        "raw": context["wedderburn_outputs"]["raw"],
        "derived": context["wedderburn_outputs"]["derived"],
        "certificate": certificate,
    }
