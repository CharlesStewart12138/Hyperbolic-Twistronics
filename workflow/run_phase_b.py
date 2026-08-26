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
        "B-01": "finite-cover sparse spectra", "B-02": "lifted Weyl no loss", "B-03": "retained-sector no pollution", "B-04": "edge and gap transport", "B-05": "operator-tempered classification", "B-06": "cross-tower independence", "B-07": "complete Wedderburn decomposition", "B-08": "single-character incompleteness", "B-09": "public projector crosscheck", "B-10": "common-space embedding", "B-11": "balanced full-shell tails", "B-12": "full-shell spectral inheritance", "B-13": "C0/C1/C2 derivative tiers", "B-14": "open-patch boundary control", "B-15": "injectivity-radius audit",
    }
    rows = []
    for task in TASKS:
        out = outputs.get(task, {})
        rows.append({"theorem_id": "Theorems 128-133 / Definitions 44-45", "claim_name": names[task], "claim_layer": "finite covers and thermodynamic bulk", "model_level": "retained scalar bilayer surface-group cover family", "code_id": task, "run_id": run_id, "validation_type": statuses.get(task, "NOT_STARTED"), "parameter_set": "configs/phase_b.yaml", "status": statuses.get(task, "NOT_STARTED"), "raw_data_file": out["raw"].relative_to(root).as_posix() if "raw" in out else "MISSING", "derived_data_file": out["derived"].relative_to(root).as_posix() if "derived" in out else "MISSING", "certificate_file": out["certificate"].relative_to(root).as_posix() if "certificate" in out else "MISSING", "notes": "Full regular pollution is preserved; bulk conclusions use only retained certified blocks." if task in {"B-03", "B-05", "B-08"} else ""})
    return rows


def checkpoint(root: Path, run_id: str, run_dir: Path, statuses: dict[str, str], outputs: dict[str, dict[str, Path]]) -> str:
    blockers = {task: status for task, status in statuses.items() if status in BLOCKING}
    inconclusive = {task: status for task, status in statuses.items() if status == "INCONCLUSIVE"}
    state = "PHASE_B_BLOCKED" if blockers else ("PHASE_B_COMPLETE" if len(statuses) == len(TASKS) else "PHASE_B_PARTIAL")
    lines = ["# Phase B checkpoint", "", f"- Run ID: `{run_id}`", f"- State: `{state}`", "- Phase I/G/S rerun: `false`", "- Non-Abelian tower gate: `PASS_CERTIFIED`", "- Abelian control towers used for bulk claims: `false`", "", "| Task | Status | Certificate |", "|---|---|---|"]
    for task in TASKS:
        cert = outputs.get(task, {}).get("certificate")
        relative = cert.relative_to(root).as_posix() if cert else "-"
        lines.append(f"| {task} | {statuses.get(task, 'NOT_STARTED')} | `{relative}` |")
    lines.extend(["", "Only complete irreducible blocks satisfying the proven reduced surface-group spectral interval are retained. Full regular outliers, the trivial character, missing public projectors, and all inconclusive cross-tower tests remain visible.", "", f"Blockers: `{json.dumps(blockers, sort_keys=True)}`", f"Inconclusive: `{json.dumps(inconclusive, sort_keys=True)}`", ""])
    text = "\n".join(lines)
    (root / "reports" / "checkpoint_B.md").write_text(text, encoding="utf-8")
    (run_dir / "derived" / "checkpoint_B.md").write_text(text, encoding="utf-8")
    state_path = root / "PROJECT_STATE.json"
    project = json.loads(state_path.read_text(encoding="utf-8"))
    project.update({"state": state, "current_phase": "B", "latest_run_id": run_id, "latest_run_directory": run_dir.relative_to(root).as_posix(), "phase_b_task_statuses": statuses, "next_task": "D-01" if state == "PHASE_B_COMPLETE" else next((task for task in TASKS if statuses.get(task) in BLOCKING or task not in statuses), "B-01"), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    state_path.write_text(json.dumps(project, indent=2, sort_keys=True), encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)
    from audit.run_manifest import finalize_run, initialize_run
    from bulk.common_space_embedding import run as b10
    from bulk.cross_tower_independence import run as b06
    from bulk.derivative_tiers import run as b13
    from bulk.edge_gap_transport import run as b04
    from bulk.finite_cover_spectra import run as b01
    from bulk.full_shell_balance import run as b11
    from bulk.full_shell_spectral_inheritance import run as b12
    from bulk.injectivity_radius_audit import run as b15
    from bulk.lifted_weyl_no_loss import run as b02
    from bulk.no_pollution_certificate import run as b03
    from bulk.open_patch_control import run as b14
    from bulk.operator_tempered_test import run as b05
    from representation.character_incompleteness import run as b08
    from representation.public_projector_crosscheck import run as b09
    from representation.wedderburn_exact import prepare_wedderburn, run as b07

    config = yaml.safe_load((root / "configs" / "phase_b.yaml").read_text(encoding="utf-8"))
    gate_path = root / str(config["tower_gate"]["certificate"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_CERTIFIED":
        raise SystemExit("non-Abelian tower gate is not certified")
    gate_run = root / "results" / str(config["tower_gate"]["run_id"])
    gate_levels = pd.read_parquet(gate_run / "derived" / "nonabelian_tower_levels.parquet")
    actions = sorted((root / str(config["tower_gate"]["raw_directory"])).glob("*.npz"))
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
        return statuses[task] not in BLOCKING

    if execute("B-01", b01):
        try:
            blocks, diagnostics, wedderburn_outputs = prepare_wedderburn(root, run_dir, run_id, config)
            context.update({"blocks": blocks, "wedderburn_diagnostics": diagnostics, "wedderburn_outputs": wedderburn_outputs})
        except Exception:
            statuses["B-07"] = "FAIL_IMPLEMENTATION"
            errors["B-07"] = traceback.format_exc()
    functions = [("B-02", b02), ("B-03", b03), ("B-04", b04), ("B-05", b05), ("B-06", b06), ("B-07", b07), ("B-08", b08), ("B-09", b09), ("B-10", b10), ("B-11", b11), ("B-12", b12), ("B-13", b13), ("B-14", b14), ("B-15", b15)]
    if not any(status in BLOCKING for status in statuses.values()):
        for task, function in functions:
            if task in statuses:
                break
            if not execute(task, function):
                break
    report = {"run_id": run_id, "atomic_order": TASKS, "task_statuses": statuses, "errors": errors, "preserved_tower_gate_run": config["tower_gate"]["run_id"], "phase_i_g_s_rerun": False}
    (run_dir / "logs" / "phase_b_execution_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(validation_rows(root, run_id, statuses, outputs)).to_parquet(run_dir / "validation_matrix.parquet", index=False)
    update_manifest(root, run_id, statuses, outputs)
    state = checkpoint(root, run_id, run_dir, statuses, outputs)
    finalize_run(run_dir, "COMPLETE" if state == "PHASE_B_COMPLETE" else "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 0 if state == "PHASE_B_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

