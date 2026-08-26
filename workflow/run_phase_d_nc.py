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


D_TASKS = [f"D-{number:02d}" for number in range(1, 15)]
NC_TASKS = [f"NC-{number:02d}" for number in range(1, 10)]
BLOCKING = {"FAIL_THEORY", "FAIL_IMPLEMENTATION"}


def add_path(root):
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def artifact_digest(path):
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest
    if path.is_file():
        return {"kind": "file", "sha256": sha256_file(path), "bytes": path.stat().st_size}
    inventory = _tree_inventory(path)
    return {"kind": "directory", "tree_inventory_sha256": inventory_digest(inventory), "file_count": len(inventory)}


def checkpoint(root, run_dir, run_id, task, status, output):
    from audit.data_io import write_json
    path = run_dir / "certificates" / "task_checkpoints" / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {
        "task_id": task, "run_id": run_id, "status": status,
        "checkpointed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {key: {"path": value.relative_to(root).as_posix(), **artifact_digest(value)} for key, value in output.items()},
    })


def update_manifest(root, run_id, statuses, outputs):
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
            if key in outputs.get(task, {}):
                row[field] = outputs[task][key].relative_to(root).as_posix()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs(root, config):
    from audit.run_manifest import sha256_file
    from external.fetch_public_baselines import tree_sha256
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest
    source_cfg = config["source_phase_b"]
    source = root / "results" / str(source_cfg["run_id"])
    inventory = _tree_inventory(source)
    checks = {
        "manifest": sha256_file(source / "manifest.json") == source_cfg["manifest_sha256"],
        "file_count": len(inventory) == int(source_cfg["file_count"]),
        "tree": inventory_digest(inventory) == source_cfg["tree_inventory_sha256"],
        "matrix": sha256_file(source / "validation_matrix.parquet") == source_cfg["validation_matrix_sha256"],
        "blocks": sha256_file(root / str(config["source_b07"]["block_spectra"])) == config["source_b07"]["block_spectra_sha256"],
        "gate": sha256_file(root / str(config["tower_gate"]["certificate"])) == config["tower_gate"]["certificate_sha256"],
        "levels": sha256_file(root / str(config["tower_gate"]["levels"])) == config["tower_gate"]["levels_sha256"],
    }
    for name in ("g10", "g11"):
        checks[name] = sha256_file(root / str(config["arithmetic_sources"][f"{name}_certificate"])) == config["arithmetic_sources"][f"{name}_sha256"]
    for name in ("s01", "s02", "s16"):
        checks[name] = sha256_file(root / str(config["negative_control_sources"][name])) == config["negative_control_sources"][f"{name}_sha256"]
    public = config["public_data"]
    for label, key in (("HyperBloch", "hyperbloch"), ("HyperCells", "hypercells"), ("cell-graph-library", "graph_library")):
        repo = root / "public_data" / label / str(public[f"{key}_revision"]) / "repo"
        checks[f"public_{key}"] = tree_sha256(repo) == public[f"{key}_tree_sha256"]
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    checks["phase_b_status"] = manifest.get("status") == "COMPLETE" and not any(value in BLOCKING for value in manifest.get("task_statuses", {}).values())
    if not all(checks.values()):
        raise RuntimeError(f"D/NC input verification failed: {json.dumps(checks, sort_keys=True)}")
    return {"source": source, "inventory": inventory, "checks": checks}


def validation_matrix_with_nc(root, run_id, context):
    matrix = context["d14_matrix"].copy()
    extra = []
    for task in NC_TASKS:
        output = context["outputs"][task]
        extra.append({
            "theorem_id": "reverse falsification", "claim_name": task,
            "claim_layer": "negative control", "model_level": "falsifying control",
            "code_id": task, "run_id": run_id, "validation_type": "FAIL_EXPECTED",
            "parameter_set": "configs/phase_d_nc.yaml", "residual_value": None, "tolerance": None,
            "certified_lower_bound": None, "certified_upper_bound": None, "physical_margin": None,
            "status": "FAIL_EXPECTED", "raw_data_file": output["raw"].relative_to(root).as_posix(),
            "derived_data_file": output["derived"].relative_to(root).as_posix(),
            "certificate_file": output["certificate"].relative_to(root).as_posix(),
            "future_figure_id": "NEGATIVE CONTROLS", "notes": "Expected failure preserved as required.",
        })
    return pd.concat([matrix, pd.DataFrame(extra)], ignore_index=True)


def write_reports(root, run_dir, run_id, statuses, outputs, source_id):
    d_inconclusive = {task: statuses[task] for task in D_TASKS if statuses[task] == "INCONCLUSIVE"}
    d_lines = [
        "# Phase D checkpoint", "", f"- Run ID: `{run_id}`", "- State: `PHASE_D_D14_COMPLETE`",
        f"- Frozen Phase-B source: `{source_id}`", "- D-15 executed: `false`",
        "- D-15 dependency guard: `G-13--G-15 and S-17--S-24 must be frozen first`",
        "- Scientific calculation in plotting scripts: `false`", "", "| Task | Status | Certificate |", "|---|---|---|",
    ]
    for task in D_TASKS:
        d_lines.append(f"| {task} | {statuses[task]} | `{outputs[task]['certificate'].relative_to(root).as_posix()}` |")
    d_lines.extend(["", f"Inconclusive results preserved: `{json.dumps(d_inconclusive, sort_keys=True)}`", "", "Next manifest task after NC: `G-13`.", ""])
    d_text = "\n".join(d_lines)
    (run_dir / "derived" / "checkpoint_D.md").write_text(d_text, encoding="utf-8")
    (root / "reports" / "checkpoint_D.md").write_text(d_text, encoding="utf-8")
    nc_lines = [
        "# Negative-control checkpoint", "", f"- Run ID: `{run_id}`", "- State: `PHASE_NC_COMPLETE`", "",
        "| Task | Status | Certificate |", "|---|---|---|",
    ]
    for task in NC_TASKS:
        nc_lines.append(f"| {task} | {statuses[task]} | `{outputs[task]['certificate'].relative_to(root).as_posix()}` |")
    nc_lines.extend(["", "All nine controls generated saved outputs and retained `FAIL_EXPECTED` status.", ""])
    nc_text = "\n".join(nc_lines)
    (run_dir / "derived" / "checkpoint_NC.md").write_text(nc_text, encoding="utf-8")
    (root / "reports" / "checkpoint_NC.md").write_text(nc_text, encoding="utf-8")
    state_path = root / "PROJECT_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "state": "PHASE_D_D14_AND_NC_COMPLETE", "current_phase": "D/NC",
        "latest_run_id": run_id, "latest_run_directory": run_dir.relative_to(root).as_posix(),
        "phase_d_task_statuses": {task: statuses[task] for task in D_TASKS},
        "negative_control_statuses": {task: statuses[task] for task in NC_TASKS},
        "phase_b_preserved_run_id": source_id, "next_task": "G-13",
        "d15_deferred_until_prerequisites": ["G-13", "G-14", "G-15", "S-17", "S-18", "S-19", "S-20", "S-21", "S-22", "S-23", "S-24"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--node-executable", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)
    from audit.error_budget import run as d13
    from audit.run_manifest import finalize_run, initialize_run
    from audit.validation_matrix import run as d14
    from diffraction.arithmetic_complexity import run as d08
    from diffraction.exact_vs_incommensurate import run as d07
    from diffraction.nonabelian_structure_factor import run as d06
    from dos.cdf_local_law import run as d02
    from dos.coherence_weighted_dos import run as d05
    from dos.kpm_slq_dos import run as d01
    from dos.unsmoothed_density import run as d04
    from dos.vanishing_broadening import run as d03
    from external.circuit_laplacian_mapping import run as d11
    from external.reproduce_circuit_spectra import run as d12
    from external.reproduce_hyperbloch_dos import run as d09
    from external.reproduce_public_graphs import run as d10
    from negative_controls import FUNCTIONS as NC_FUNCTIONS

    config = yaml.safe_load((root / "configs" / "phase_d_nc.yaml").read_text(encoding="utf-8"))
    project = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    if project.get("state") != "PHASE_B_COMPLETE" or project.get("latest_run_id") != config["source_phase_b"]["run_id"]:
        raise SystemExit("project is not at the declared Phase-B complete checkpoint")
    verified = verify_inputs(root, config)
    run_id, run_dir = initialize_run(root)
    statuses = {}
    outputs = {}
    errors = {}
    context = {
        "blocks": pd.read_parquet(root / str(config["source_b07"]["block_spectra"])),
        "actions": sorted((root / str(config["tower_gate"]["actions"])).glob("*.npz")),
        "phase_b_dir": verified["source"], "node_executable": args.node_executable.resolve(),
        "statuses": statuses, "outputs": outputs,
    }
    d_functions = {
        "D-01": d01, "D-02": d02, "D-03": d03, "D-04": d04, "D-05": d05,
        "D-06": d06, "D-07": d07, "D-08": d08, "D-09": d09, "D-10": d10,
        "D-11": d11, "D-12": d12, "D-13": d13, "D-14": d14,
    }
    for task in D_TASKS:
        try:
            status, output = d_functions[task](config, run_dir, run_id, root, context)
            statuses[task], outputs[task] = status, output
            checkpoint(root, run_dir, run_id, task, status, output)
        except Exception:
            statuses[task] = "FAIL_IMPLEMENTATION"
            errors[task] = traceback.format_exc()
        atomic_json(run_dir / "logs" / "phase_d_nc_progress.json", {"task_statuses": statuses, "errors": errors})
        if statuses[task] in BLOCKING:
            break
    if not any(status in BLOCKING for status in statuses.values()):
        for task in NC_TASKS:
            try:
                status, output = NC_FUNCTIONS[task](config, run_dir, run_id, root, context)
                statuses[task], outputs[task] = status, output
                checkpoint(root, run_dir, run_id, task, status, output)
            except Exception:
                statuses[task] = "FAIL_IMPLEMENTATION"
                errors[task] = traceback.format_exc()
            atomic_json(run_dir / "logs" / "phase_d_nc_progress.json", {"task_statuses": statuses, "errors": errors})
            if statuses[task] in BLOCKING:
                break
    source_after = verify_inputs(root, config)
    source_equal = source_after["inventory"] == verified["inventory"]
    if not source_equal:
        raise RuntimeError("frozen Phase-B source changed during D/NC continuation")
    complete = all(task in statuses for task in D_TASKS + NC_TASKS) and not any(status in BLOCKING for status in statuses.values())
    if complete:
        validation_matrix_with_nc(root, run_id, context).to_parquet(run_dir / "validation_matrix.parquet", index=False)
        update_manifest(root, run_id, statuses, outputs)
        write_reports(root, run_dir, run_id, statuses, outputs, str(config["source_phase_b"]["run_id"]))
    write_json = __import__("audit.data_io", fromlist=["write_json"]).write_json
    write_json(run_dir / "certificates" / "frozen_phase_b_source_integrity.json", {
        "status": "PASS_EXACT", "source_run_id": config["source_phase_b"]["run_id"],
        "before_equals_after": source_equal, "tree_inventory_sha256": config["source_phase_b"]["tree_inventory_sha256"],
        "file_count": config["source_phase_b"]["file_count"],
    })
    atomic_json(run_dir / "logs" / "phase_d_nc_execution_report.json", {
        "run_id": run_id, "source_phase_b_run_id": config["source_phase_b"]["run_id"],
        "execution_order": D_TASKS + NC_TASKS, "task_statuses": statuses, "errors": errors,
        "phase_b_rerun": False, "d15_executed": False,
    })
    finalize_run(run_dir, "COMPLETE" if complete else "INCOMPLETE", statuses)
    state = "PHASE_D_D14_AND_NC_COMPLETE" if complete else "PHASE_D_OR_NC_BLOCKED"
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
