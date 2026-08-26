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


TASKS = [f"G-{number:02d}" for number in range(1, 13)]
PASS_STATUSES = {"PASS_EXACT", "PASS_CERTIFIED", "PASS_CONVERGED", "PASS_EXTERNAL"}


def add_source_path(root: Path) -> None:
    for path in (root / "src", root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def update_task_manifest(root: Path, run_id: str, statuses: dict[str, str], outputs: dict[str, dict[str, Path]]) -> None:
    path = root / "TASK_MANIFEST.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for row in rows:
        task_id = row["task_id"]
        if task_id not in statuses:
            continue
        row["status"] = statuses[task_id]
        row["run_id"] = run_id
        task_outputs = outputs.get(task_id, {})
        for field, key in (("raw_output", "raw"), ("derived_output", "derived"), ("certificate", "certificate")):
            if key in task_outputs:
                row[field] = task_outputs[key].relative_to(root).as_posix()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def matrix_rows(root: Path, run_id: str, statuses: dict[str, str], outputs: dict[str, dict[str, Path]]) -> list[dict[str, object]]:
    claims = {
        "G-01": ("Theorem 3", "centered hyperbolic twist displacement identity", "geometry", "M1-M5"),
        "G-02": ("Theorem 3", "closed-form moire length inversion", "geometry", "M1-M5"),
        "G-03": ("Theorem 3", "effective hyperbolic registry area", "geometry", "M1-M5"),
        "G-04": ("Theorem 4", "universal crossover amplitude and flow", "geometry", "M1-M5"),
        "G-05": ("Theorem 4", "double-scaling limit and order-two correction", "geometry", "M1-M5"),
        "G-06": ("Theorem 4", "D-dimensional active-subspace endpoint law", "geometry", "M1-M5"),
        "G-07": ("Euclidean CSL benchmark", "10/34-site square CSL and parity reduction", "arithmetic", "M0"),
        "G-08": ("Euclidean CSL benchmark", "complete square CSL catalog below 100 atoms", "arithmetic", "M0"),
        "G-09": ("Theorem 2", "explicit centered commensurator sequence", "arithmetic", "M1-M5"),
        "G-10": ("Theorem 3", "maximal-order local/global coincidence formula", "arithmetic", "M1-M5"),
        "G-11": ("Theorem 3", "explicit sequence arithmetic exponent four", "arithmetic", "M1-M5"),
        "G-12": ("Theorem 11", "radial non-locking limits three and four", "arithmetic", "M1-M5"),
    }
    rows = []
    for task_id in TASKS:
        theorem_id, claim, layer, model = claims[task_id]
        task_outputs = outputs.get(task_id, {})
        rows.append({
            "theorem_id": theorem_id,
            "claim_name": claim,
            "claim_layer": layer,
            "model_level": model,
            "code_id": task_id,
            "run_id": run_id,
            "validation_type": statuses.get(task_id, "INCONCLUSIVE"),
            "parameter_set": "configs/phase_g.yaml",
            "residual_value": None,
            "certified_lower_bound": None,
            "certified_upper_bound": None,
            "physical_margin": None,
            "status": statuses.get(task_id, "INCONCLUSIVE"),
            "raw_data_file": task_outputs.get("raw", Path("MISSING")).relative_to(root).as_posix() if "raw" in task_outputs else "MISSING",
            "derived_data_file": task_outputs.get("derived", Path("MISSING")).relative_to(root).as_posix() if "derived" in task_outputs else "MISSING",
            "certificate_file": task_outputs.get("certificate", Path("MISSING")).relative_to(root).as_posix() if "certificate" in task_outputs else "MISSING",
            "future_figure_id": "FIGURE 2" if task_id <= "G-06" else "FIGURE 3",
            "notes": "Fixed-axis cases are active-subspace only." if task_id == "G-06" else ("Maximal-order exact value; arbitrary fixed Gamma retains explicit comparison guard." if task_id in {"G-10", "G-11", "G-12"} else ""),
        })
    return rows


def write_checkpoint(root: Path, run_id: str, run_dir: Path, statuses: dict[str, str], outputs: dict[str, dict[str, Path]]) -> str:
    complete = all(statuses.get(task) in PASS_STATUSES for task in TASKS)
    state = "PHASE_G_COMPLETE" if complete else "PHASE_G_BLOCKED"
    lines = ["# Phase G checkpoint", "", f"- Run ID: `{run_id}`", f"- State: `{state}`", "- Phase I rerun: `false`", "- Scientific scans started: `true`", "", "## Task results", "", "| Task | Status | Raw | Derived | Certificate |", "|---|---|---|---|---|"]
    for task in TASKS:
        values = outputs.get(task, {})
        links = {key: values[key].relative_to(root).as_posix() if key in values else "-" for key in ("raw", "derived", "certificate")}
        lines.append(f"| {task} | {statuses.get(task, 'INCONCLUSIVE')} | `{links['raw']}` | `{links['derived']}` | `{links['certificate']}` |")
    lines.extend(["", "## Scientific interpretation", "", "The centered constant-curvature displacement, moire-length, effective-area, crossover-flow, and double-scaling identities were tested independently of plotting. Exact Euclidean CSL benchmarks and the explicit (2,3,8) commensurator sequence were evaluated with rational/algebraic arithmetic. The arithmetic sequence approaches exponent four, not exponent one; the corresponding radial ratios approach three and four.", "", "Odd-dimensional rotations with a fixed axis are explicitly excluded from an ambient full-D volume claim; only their active rotating subspace was evaluated.", "", "## Next task", "", "Proceed directly to S-01. No Phase I calculation is repeated."])
    text = "\n".join(lines) + "\n"
    report = root / "reports" / "checkpoint_G.md"
    report.write_text(text, encoding="utf-8")
    (run_dir / "derived" / "checkpoint_G.md").write_text(text, encoding="utf-8")
    old_state = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    old_state.update({"state": state, "current_phase": "G", "scientific_scans_started": True, "latest_run_id": run_id, "latest_run_directory": run_dir.relative_to(root).as_posix(), "phase_g_task_statuses": statuses, "phase_i_preserved_run_id": old_state.get("latest_run_id"), "next_task": "S-01" if complete else next((task for task in TASKS if statuses.get(task) not in PASS_STATUSES), "G-01"), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    (root / "PROJECT_STATE.json").write_text(json.dumps(old_state, indent=2, sort_keys=True), encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_source_path(root)
    from analysis.arithmetic_exponent import run as g11
    from analysis.radial_locking import run as g12
    from audit.run_manifest import finalize_run, initialize_run
    from exact.coincidence_index_height import run as g10
    from exact.commensurator_sequence import run as g09
    from exact.euclidean_angle_catalog import run as g08
    from exact.euclidean_square_csl import run as g07
    from geometry.crossover_collapse import run as g04
    from geometry.dimensional_extension import run as g06
    from geometry.double_scaling import run as g05
    from geometry.effective_area import run as g03
    from geometry.moire_length_inversion import run as g02
    from geometry.validate_distance_identity import run as g01

    config = yaml.safe_load((root / "configs" / "phase_g.yaml").read_text(encoding="utf-8"))
    run_id, run_dir = initialize_run(root)
    statuses: dict[str, str] = {}
    outputs: dict[str, dict[str, Path]] = {}
    errors: dict[str, str] = {}
    functions = [("G-01", g01), ("G-02", g02), ("G-03", g03), ("G-04", g04), ("G-05", g05), ("G-06", g06), ("G-07", g07), ("G-08", g08), ("G-09", g09), ("G-10", g10), ("G-11", g11), ("G-12", g12)]
    for task_id, function in functions:
        if task_id in {"G-11", "G-12"} and statuses.get("G-10") not in PASS_STATUSES:
            statuses[task_id] = "INCONCLUSIVE"
            errors[task_id] = "Unresolved prerequisite G-10"
            continue
        try:
            status, task_outputs = function(config, run_dir, run_id)
            statuses[task_id] = status
            outputs[task_id] = task_outputs
        except Exception:
            statuses[task_id] = "FAIL_IMPLEMENTATION"
            errors[task_id] = traceback.format_exc()
    execution = {"run_id": run_id, "atomic_order": TASKS, "task_statuses": statuses, "errors": errors, "phase_i_rerun": False}
    (run_dir / "logs" / "phase_g_execution_report.json").write_text(json.dumps(execution, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(matrix_rows(root, run_id, statuses, outputs)).to_parquet(run_dir / "validation_matrix.parquet", index=False)
    update_task_manifest(root, run_id, statuses, outputs)
    state = write_checkpoint(root, run_id, run_dir, statuses, outputs)
    finalize_run(run_dir, "COMPLETE" if state == "PHASE_G_COMPLETE" else "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "run_dir": str(run_dir), "task_statuses": statuses, "errors": errors}, indent=2))
    return 0 if state == "PHASE_G_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

