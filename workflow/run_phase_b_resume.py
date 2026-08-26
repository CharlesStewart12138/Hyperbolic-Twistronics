from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


TASKS = [f"B-{number:02d}" for number in range(1, 16)]
BLOCKING = {"FAIL_THEORY", "FAIL_IMPLEMENTATION"}


def add_path(root: Path) -> None:
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def update_manifest(root: Path, run_id: str, statuses: dict[str, str], outputs: dict[str, dict[str, Path]]) -> None:
    path = root / "TASK_MANIFEST.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        task = row["task_id"]
        if task not in statuses:
            continue
        row["status"] = statuses[task]
        row["run_id"] = run_id
        for field, key in (("raw_output", "raw"), ("derived_output", "derived"), ("certificate", "certificate")):
            value = outputs.get(task, {}).get(key)
            if value is not None:
                row[field] = value.relative_to(root).as_posix()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validation_rows(root: Path, run_id: str, statuses: dict[str, str], outputs: dict[str, dict[str, Path]]) -> list[dict[str, object]]:
    names = {
        "B-01": "finite-cover sparse spectra",
        "B-02": "lifted Weyl no loss",
        "B-03": "retained-sector no pollution",
        "B-04": "edge and gap transport",
        "B-05": "operator-tempered classification",
        "B-06": "cross-tower independence",
        "B-07": "complete resumable Wedderburn decomposition",
        "B-08": "single-character incompleteness",
        "B-09": "public projector crosscheck",
        "B-10": "common-space embedding",
        "B-11": "balanced full-shell tails",
        "B-12": "full-shell spectral inheritance",
        "B-13": "C0/C1/C2 derivative tiers",
        "B-14": "open-patch boundary control",
        "B-15": "injectivity-radius audit",
    }
    rows: list[dict[str, object]] = []
    for task in TASKS:
        out = outputs.get(task, {})
        rows.append(
            {
                "theorem_id": "Theorems 128-133 / Definitions 44-45",
                "claim_name": names[task],
                "claim_layer": "finite covers and thermodynamic bulk",
                "model_level": "retained scalar bilayer surface-group cover family",
                "code_id": task,
                "run_id": run_id,
                "validation_type": statuses.get(task, "NOT_STARTED"),
                "parameter_set": "configs/phase_b_resume.yaml",
                "status": statuses.get(task, "NOT_STARTED"),
                "raw_data_file": out["raw"].relative_to(root).as_posix() if "raw" in out else "MISSING",
                "derived_data_file": out["derived"].relative_to(root).as_posix() if "derived" in out else "MISSING",
                "certificate_file": out["certificate"].relative_to(root).as_posix() if "certificate" in out else "MISSING",
                "notes": "B-01 is byte-preserved from the frozen predecessor run."
                if task == "B-01"
                else ("Full regular pollution is preserved; bulk conclusions use only retained certified blocks." if task in {"B-03", "B-05", "B-08"} else ""),
            }
        )
    return rows


def preserve_b01(root: Path, run_dir: Path, run_id: str, config: dict[str, object]):
    from audit.data_io import write_json
    from audit.run_manifest import sha256_file

    source_id = str(config["preserved_b01"]["run_id"])
    source = root / "results" / source_id
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    certificate_path = source / "certificates" / "b01_finite_cover_spectra.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if manifest.get("task_statuses", {}).get("B-01") != "PASS_CONVERGED" or certificate.get("status") != "PASS_CONVERGED":
        raise RuntimeError("preserved B-01 source is not PASS_CONVERGED")
    raw_dir = source / "raw" / "finite_cover_spectra"
    derived = source / "derived" / "b01_finite_cover_edge_spectra.parquet"
    raw_hashes = {path.name: sha256_file(path) for path in sorted(raw_dir.glob("*.npz"))}
    reuse_certificate = run_dir / "certificates" / "b01_preserved_reuse.json"
    payload = {
        "task_id": "B-01",
        "run_id": run_id,
        "status": "PASS_CONVERGED",
        "execution": "NOT_RERUN",
        "preserved_source_run_id": source_id,
        "preserved_source_manifest_status": manifest.get("status"),
        "source_certificate": certificate_path.relative_to(root).as_posix(),
        "source_certificate_sha256": sha256_file(certificate_path),
        "source_derived": derived.relative_to(root).as_posix(),
        "source_derived_sha256": sha256_file(derived),
        "source_raw_sha256": raw_hashes,
        "source_files_modified": False,
    }
    if reuse_certificate.exists():
        existing = json.loads(reuse_certificate.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing B-01 reuse certificate is incompatible")
    else:
        write_json(reuse_certificate, payload)
    return "PASS_CONVERGED", {"raw": raw_dir, "derived": derived, "certificate": reuse_certificate}


def save_progress(run_dir: Path, statuses: dict[str, str], outputs: dict[str, dict[str, Path]], errors: dict[str, str]) -> None:
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_statuses": statuses,
        "outputs": {task: {key: str(value) for key, value in paths.items()} for task, paths in outputs.items()},
        "errors": errors,
    }
    path = run_dir / "logs" / "phase_b_progress.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def checkpoint(root: Path, run_id: str, run_dir: Path, statuses: dict[str, str], outputs: dict[str, dict[str, Path]]) -> str:
    blockers = {task: status for task, status in statuses.items() if status in BLOCKING}
    inconclusive = {task: status for task, status in statuses.items() if status == "INCONCLUSIVE"}
    state = "PHASE_B_BLOCKED" if blockers else ("PHASE_B_COMPLETE" if len(statuses) == len(TASKS) else "PHASE_B_PARTIAL")
    lines = [
        "# Phase B checkpoint (resumed execution)",
        "",
        f"- Run ID: `{run_id}`",
        f"- State: `{state}`",
        "- Phase I/G/S rerun: `false`",
        "- Previous failed Phase-B run modified: `false`",
        "- B-01 recomputed: `false`",
        "- Non-Abelian tower gate: `PASS_CERTIFIED`",
        "- Abelian control towers used for bulk claims: `false`",
        "",
        "| Task | Status | Certificate |",
        "|---|---|---|",
    ]
    for task in TASKS:
        certificate = outputs.get(task, {}).get("certificate")
        relative = certificate.relative_to(root).as_posix() if certificate else "-"
        lines.append(f"| {task} | {statuses.get(task, 'NOT_STARTED')} | `{relative}` |")
    lines.extend(["", f"Blockers: `{json.dumps(blockers, sort_keys=True)}`", f"Inconclusive: `{json.dumps(inconclusive, sort_keys=True)}`", ""])
    text = "\n".join(lines)
    (root / "reports" / "checkpoint_B.md").write_text(text, encoding="utf-8")
    (run_dir / "derived" / "checkpoint_B.md").write_text(text, encoding="utf-8")
    if "B-07" in blockers:
        next_task = "B-07"
    elif state == "PHASE_B_COMPLETE":
        next_task = "D-01"
    else:
        next_task = next((task for task in TASKS if task not in statuses), "B-01")
    project_path = root / "PROJECT_STATE.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project.update(
        {
            "state": state,
            "current_phase": "B",
            "latest_run_id": run_id,
            "latest_run_directory": run_dir.relative_to(root).as_posix(),
            "phase_b_task_statuses": statuses,
            "next_task": next_task,
            "preserved_phase_b_failed_run_id": "0b4258f5846301f260d77c64f2f0dfe91223d082a5342044235db030dbb12203",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    project_path.write_text(json.dumps(project, indent=2, sort_keys=True), encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--resume-run-id", help="resume an externally interrupted RUNNING immutable attempt")
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)

    from audit.run_manifest import finalize_run, initialize_run
    from bulk.common_space_embedding import run as b10
    from bulk.cross_tower_independence import run as b06
    from bulk.derivative_tiers import run as b13
    from bulk.edge_gap_transport import run as b04
    from bulk.full_shell_balance import run as b11
    from bulk.full_shell_spectral_inheritance import run as b12
    from bulk.injectivity_radius_audit import run as b15
    from bulk.lifted_weyl_no_loss import run as b02
    from bulk.no_pollution_certificate import run as b03
    from bulk.open_patch_control import run as b14
    from bulk.operator_tempered_test import run as b05
    from representation.character_incompleteness import run as b08
    from representation.public_projector_crosscheck import run as b09
    from representation.wedderburn_resumable import B07StageFailure, prepare_wedderburn, run as b07

    config = yaml.safe_load((root / "configs" / "phase_b_resume.yaml").read_text(encoding="utf-8"))
    gate_path = root / str(config["tower_gate"]["certificate"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_CERTIFIED":
        raise SystemExit("non-Abelian tower gate is not certified")
    previous = root / "results" / str(config["preserved_failed_run"])
    previous_manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
    if previous_manifest.get("task_statuses", {}).get("B-07") != "FAIL_IMPLEMENTATION":
        raise SystemExit("declared frozen predecessor does not contain the expected B-07 failure")
    gate_run = root / "results" / str(config["tower_gate"]["run_id"])
    gate_levels = pd.read_parquet(gate_run / "derived" / "nonabelian_tower_levels.parquet")
    actions = sorted((root / str(config["tower_gate"]["raw_directory"])).glob("*.npz"))

    if args.resume_run_id:
        run_id = args.resume_run_id
        run_dir = root / "results" / run_id
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "RUNNING":
            raise SystemExit("only a RUNNING attempt can be resumed")
    else:
        run_id, run_dir = initialize_run(root)

    context: dict[str, object] = {"gate": gate, "gate_levels": gate_levels, "actions": actions}
    statuses: dict[str, str] = {}
    outputs: dict[str, dict[str, Path]] = {}
    errors: dict[str, str] = {}

    def execute(task: str, function) -> bool:
        try:
            status, out = function(config, run_dir, run_id, root, context)
            statuses[task] = status
            outputs[task] = out
        except Exception:
            statuses[task] = "FAIL_IMPLEMENTATION"
            errors[task] = traceback.format_exc()
        save_progress(run_dir, statuses, outputs, errors)
        return statuses[task] not in BLOCKING

    try:
        status, out = preserve_b01(root, run_dir, run_id, config)
        statuses["B-01"] = status
        outputs["B-01"] = out
        save_progress(run_dir, statuses, outputs, errors)
        try:
            blocks, diagnostics, wedderburn_outputs = prepare_wedderburn(root, run_dir, run_id, config)
            context.update(
                {
                    "blocks": blocks,
                    "wedderburn_diagnostics": diagnostics,
                    "wedderburn_outputs": wedderburn_outputs,
                }
            )
            status, out = b07(config, run_dir, run_id, root, context)
            statuses["B-07"] = status
            outputs["B-07"] = out
        except B07StageFailure as error:
            statuses["B-07"] = "FAIL_IMPLEMENTATION"
            failure = run_dir / "certificates" / "b07_failure.json"
            outputs["B-07"] = {"raw": run_dir / "raw" / "representation", "certificate": failure}
            errors["B-07"] = traceback.format_exc()
        except Exception:
            statuses["B-07"] = "FAIL_IMPLEMENTATION"
            errors["B-07"] = traceback.format_exc()
        save_progress(run_dir, statuses, outputs, errors)
        if statuses.get("B-07") == "PASS_CERTIFIED":
            functions = [
                ("B-02", b02),
                ("B-03", b03),
                ("B-04", b04),
                ("B-05", b05),
                ("B-06", b06),
                ("B-08", b08),
                ("B-09", b09),
                ("B-10", b10),
                ("B-11", b11),
                ("B-12", b12),
                ("B-13", b13),
                ("B-14", b14),
                ("B-15", b15),
            ]
            for task, function in functions:
                if not execute(task, function):
                    break
    except Exception:
        if "B-07" not in statuses:
            statuses["B-07"] = "FAIL_IMPLEMENTATION"
            errors["B-07"] = traceback.format_exc()
        save_progress(run_dir, statuses, outputs, errors)

    report = {
        "run_id": run_id,
        "atomic_order": ["B-01", "B-07", "B-02", "B-03", "B-04", "B-05", "B-06", "B-08", "B-09", "B-10", "B-11", "B-12", "B-13", "B-14", "B-15"],
        "task_statuses": statuses,
        "errors": errors,
        "preserved_failed_run": config["preserved_failed_run"],
        "preserved_b01_run": config["preserved_b01"]["run_id"],
        "phase_i_g_s_rerun": False,
        "b01_rerun": False,
    }
    (run_dir / "logs" / "phase_b_execution_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(validation_rows(root, run_id, statuses, outputs)).to_parquet(
        run_dir / "validation_matrix.parquet", index=False
    )
    update_manifest(root, run_id, statuses, outputs)
    state = checkpoint(root, run_id, run_dir, statuses, outputs)
    finalize_run(run_dir, "COMPLETE" if state == "PHASE_B_COMPLETE" else "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 0 if state == "PHASE_B_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
