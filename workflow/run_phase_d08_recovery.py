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


REUSED_TASKS = [f"D-{number:02d}" for number in range(1, 8)]
D_EXECUTION = [f"D-{number:02d}" for number in range(8, 16)]
D_TASKS = [f"D-{number:02d}" for number in range(1, 16)]
NC_TASKS = [f"NC-{number:02d}" for number in range(1, 10)]
BLOCKING = {"FAIL_THEORY", "FAIL_IMPLEMENTATION"}
D08_REQUIRED_PASS = "PASS_CERTIFIED"


def add_path(root: Path) -> None:
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_configs(root: Path):
    from audit.run_manifest import sha256_file
    recovery_path = root / "configs" / "phase_d08_recovery.yaml"
    recovery = yaml.safe_load(recovery_path.read_text(encoding="utf-8"))
    base_path = root / str(recovery["base_config"]["path"])
    if sha256_file(base_path) != recovery["base_config"]["sha256"]:
        raise RuntimeError("frozen Phase-D base config hash changed")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    return base, recovery, recovery_path


def artifact_digest(path: Path) -> dict:
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest
    if path.is_file():
        return {"kind": "file", "sha256": sha256_file(path), "bytes": path.stat().st_size}
    inventory = _tree_inventory(path)
    return {"kind": "directory", "tree_inventory_sha256": inventory_digest(inventory), "file_count": len(inventory)}


def verify_frozen_predecessor(root: Path, recovery: dict):
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest
    expected = recovery["frozen_predecessor"]
    source = root / "results" / str(expected["run_id"])
    manifest_hash = sha256_file(source / "manifest.json")
    inventory = _tree_inventory(source)
    tree_hash = inventory_digest(inventory)
    if manifest_hash != expected["manifest_sha256"] or tree_hash != expected["tree_inventory_sha256"] or len(inventory) != int(expected["file_count"]):
        raise RuntimeError("frozen D-08 predecessor inventory changed")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected_statuses = {task: row["status"] for task, row in expected["reused_tasks"].items()}
    expected_statuses["D-08"] = expected["frozen_d08_status"]
    if manifest.get("status") != "INCOMPLETE" or manifest.get("task_statuses") != expected_statuses:
        raise RuntimeError("frozen D-08 predecessor manifest status changed")
    d08_checkpoint = source / "certificates" / "task_checkpoints" / "D-08.json"
    d08_certificate = source / "certificates" / "d08_arithmetic_complexity.json"
    if sha256_file(d08_checkpoint) != expected["d08_checkpoint_sha256"] or sha256_file(d08_certificate) != expected["d08_certificate_sha256"]:
        raise RuntimeError("frozen D-08 failure artifacts changed")
    frozen_d08 = json.loads(d08_certificate.read_text(encoding="utf-8"))
    if frozen_d08.get("status") != "FAIL_THEORY" or abs(float(frozen_d08.get("fitted_tail_exponent")) - 167.1438203092077) > 1.0e-10:
        raise RuntimeError("frozen pathological D-08 record changed")
    outputs = {}
    checkpoint_records = {}
    for task, row in expected["reused_tasks"].items():
        checkpoint_path = source / "certificates" / "task_checkpoints" / f"{task}.json"
        if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"frozen predecessor checkpoint changed: {task}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("status") != row["status"]:
            raise RuntimeError(f"frozen predecessor task status changed: {task}")
        task_outputs = {}
        for key, record in checkpoint["outputs"].items():
            path = root / record["path"]
            actual = artifact_digest(path)
            if any(actual.get(name) != record.get(name) for name in actual):
                raise RuntimeError(f"frozen predecessor output digest changed: {task}/{key}")
            task_outputs[key] = path
        outputs[task] = task_outputs
        checkpoint_records[task] = {
            "checkpoint": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "status": row["status"], "outputs": checkpoint["outputs"],
        }
    return {
        "source": source, "manifest": manifest, "manifest_hash": manifest_hash,
        "inventory": inventory, "tree_hash": tree_hash, "outputs": outputs,
        "statuses": expected_statuses, "checkpoint_records": checkpoint_records,
        "frozen_d08": frozen_d08,
    }


def restore_context(base: dict, root: Path, args, phase_b_verified, predecessor, statuses, outputs):
    d01 = json.loads(outputs["D-01"]["certificate"].read_text(encoding="utf-8"))
    d05 = json.loads(outputs["D-05"]["certificate"].read_text(encoding="utf-8"))
    return {
        "blocks": pd.read_parquet(root / str(base["source_b07"]["block_spectra"])),
        "actions": sorted((root / str(base["tower_gate"]["actions"])).glob("*.npz")),
        "phase_b_dir": phase_b_verified["source"],
        "node_executable": args.node_executable.resolve() if args.node_executable else None,
        "statuses": statuses, "outputs": outputs,
        "d01_summary": pd.DataFrame(d01["records"]),
        "d02_records": pd.read_parquet(outputs["D-02"]["derived"]),
        "d02_status": statuses["D-02"],
        "d03_terms": pd.read_parquet(outputs["D-03"]["derived"]),
        "d03_status": statuses["D-03"],
        "d04_status": statuses["D-04"],
        "d05_errors": d05["error_components"],
        "d06_summary": pd.read_parquet(outputs["D-06"]["derived"]),
        "d07_frame": pd.read_parquet(outputs["D-07"]["derived"]),
    }


def checkpoint(root: Path, run_dir: Path, run_id: str, task: str, status: str, output: dict[str, Path]):
    from audit.data_io import write_json
    path = run_dir / "certificates" / "task_checkpoints" / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {
        "task_id": task, "run_id": run_id, "status": status,
        "checkpointed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {key: {"path": value.relative_to(root).as_posix(), **artifact_digest(value)} for key, value in output.items()},
    })


def save_progress(root: Path, run_dir: Path, run_id: str, statuses, outputs, errors, stage: str):
    atomic_json(run_dir / "logs" / "phase_d08_recovery_progress.json", {
        "run_id": run_id, "stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_statuses": statuses,
        "outputs": {task: {key: value.relative_to(root).as_posix() for key, value in out.items()} for task, out in outputs.items()},
        "errors": errors,
    })


def update_task_manifest(root: Path, run_id: str, statuses, outputs, predecessor_id: str):
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
        if task in REUSED_TASKS:
            note = f"hash-only reuse from frozen run {predecessor_id}; not rerun"
            if note not in row["notes"]:
                row["notes"] = f"{row['notes']} {note}".strip()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(root: Path, run_dir: Path, run_id: str, statuses, outputs, state: str, errors, predecessor_id: str):
    d08 = outputs.get("D-08", {}).get("certificate")
    d08_data = json.loads(d08.read_text(encoding="utf-8")) if d08 and d08.exists() else {}
    lines = [
        "# Phase D-08 recovery checkpoint", "", f"- Run ID: `{run_id}`", f"- State: `{state}`",
        f"- Frozen predecessor: `{predecessor_id}`", "- Frozen predecessor D-08 label modified: `false`",
        "- D-01--D-07 execution: `NOT_RERUN_HASH_ONLY_REUSE`",
        "- Estimator: `dyadic_scale_separated_v1`, fixed `c=2`", "- Post-hoc window selection: `false`",
        "- D-15 user-authorized order override recorded: `true`", "", "| Task | Status | Certificate |", "|---|---|---|",
    ]
    for task in D_TASKS + NC_TASKS:
        cert = outputs.get(task, {}).get("certificate")
        lines.append(f"| {task} | {statuses.get(task, 'NOT_STARTED')} | `{cert.relative_to(root).as_posix() if cert else '-'}` |")
    if d08_data:
        lines.extend([
            "", "## D-08 estimator outcome", "",
            f"- primary extrapolation: `{d08_data.get('primary_extrapolated_exponent')}`",
            f"- primary tail center: `{d08_data.get('primary_tail_center')}`",
            f"- upper-envelope extrapolation: `{d08_data.get('upper_envelope_extrapolated_exponent')}`",
            f"- lower-envelope extrapolation: `{d08_data.get('lower_envelope_extrapolated_exponent')}`",
            f"- G-11 frozen extrapolation: `{d08_data.get('g11_frozen_extrapolation')}`",
            f"- acceptance checks: `{json.dumps(d08_data.get('acceptance_checks', {}), sort_keys=True)}`",
        ])
    lines.extend(["", f"Errors: `{json.dumps(errors, sort_keys=True)}`", ""])
    text = "\n".join(lines)
    (run_dir / "derived" / "checkpoint_D08_recovery.md").write_text(text, encoding="utf-8")
    (root / "reports" / "checkpoint_D08_recovery.md").write_text(text, encoding="utf-8")


def update_project_state(root: Path, run_dir: Path, run_id: str, statuses, state: str, next_task: str, predecessor_id: str, blocker=None):
    path = root / "PROJECT_STATE.json"
    project = json.loads(path.read_text(encoding="utf-8"))
    prior = project.pop("blocker", None)
    history = project.setdefault("historical_blockers", [])
    if prior and prior not in history:
        history.append(prior)
    if blocker:
        project["blocker"] = blocker
    project.update({
        "state": state, "current_phase": "D/NC", "latest_run_id": run_id,
        "latest_run_directory": run_dir.relative_to(root).as_posix(),
        "phase_d_task_statuses": {task: statuses[task] for task in D_TASKS if task in statuses},
        "negative_control_statuses": {task: statuses[task] for task in NC_TASKS if task in statuses},
        "d08_frozen_predecessor_run_id": predecessor_id,
        "next_task": next_task, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(path, project)


def verify_current_identity(root: Path, run_id: str):
    from audit.run_manifest import build_identity
    current_id, _ = build_identity(root)
    if current_id != run_id:
        raise RuntimeError("code/config identity changed between D-08 and continuation stages")


def finalize_matrix(root: Path, run_dir: Path, run_id: str, context, statuses, outputs):
    frame = context["d14_matrix"].copy()
    mask = frame.code_id == "D-15"
    frame.loc[mask, "run_id"] = run_id
    frame.loc[mask, "validation_type"] = statuses["D-15"]
    frame.loc[mask, "status"] = statuses["D-15"]
    frame.loc[mask, "raw_data_file"] = outputs["D-15"]["raw"].relative_to(root).as_posix()
    frame.loc[mask, "derived_data_file"] = outputs["D-15"]["derived"].relative_to(root).as_posix()
    frame.loc[mask, "certificate_file"] = outputs["D-15"]["certificate"].relative_to(root).as_posix()
    for task in NC_TASKS:
        output = outputs[task]
        mask = frame.code_id == task
        if not bool(mask.any()):
            raise RuntimeError(f"D-14 validation matrix omitted required negative control {task}")
        frame.loc[mask, "theorem_id"] = "reverse falsification"
        frame.loc[mask, "claim_name"] = task
        frame.loc[mask, "claim_layer"] = "negative control"
        frame.loc[mask, "model_level"] = "falsifying control"
        frame.loc[mask, "run_id"] = run_id
        frame.loc[mask, "validation_type"] = statuses[task]
        frame.loc[mask, "parameter_set"] = context.get("parameter_set", "configs/phase_d08_recovery.yaml")
        frame.loc[mask, "status"] = statuses[task]
        frame.loc[mask, "raw_data_file"] = output["raw"].relative_to(root).as_posix()
        frame.loc[mask, "derived_data_file"] = output["derived"].relative_to(root).as_posix()
        frame.loc[mask, "certificate_file"] = output["certificate"].relative_to(root).as_posix()
        frame.loc[mask, "future_figure_id"] = "NEGATIVE CONTROLS"
        frame.loc[mask, "notes"] = "Expected failure preserved as required."
    if frame.code_id.duplicated().any():
        raise RuntimeError("final validation matrix contains duplicate task rows")
    frame.to_parquet(run_dir / "validation_matrix.parquet", index=False)


def finalize_integrity(root: Path, run_dir: Path, recovery, before, phase_b_before, base):
    from audit.data_io import write_json
    after = verify_frozen_predecessor(root, recovery)
    from workflow.run_phase_d_nc import verify_inputs
    phase_b_after = verify_inputs(root, base)
    unchanged = after["inventory"] == before["inventory"] and phase_b_after["inventory"] == phase_b_before["inventory"]
    write_json(run_dir / "certificates" / "frozen_predecessor_integrity.json", {
        "status": "PASS_EXACT" if unchanged else "FAIL_IMPLEMENTATION",
        "predecessor_run_id": recovery["frozen_predecessor"]["run_id"],
        "predecessor_before_equals_after": after["inventory"] == before["inventory"],
        "predecessor_tree_inventory_sha256": before["tree_hash"],
        "phase_b_before_equals_after": phase_b_after["inventory"] == phase_b_before["inventory"],
        "phase_b_tree_inventory_sha256": base["source_phase_b"]["tree_inventory_sha256"],
    })
    if not unchanged:
        raise RuntimeError("a frozen predecessor changed during recovery")


def stage_d08(root: Path, args, base, recovery, recovery_path):
    from audit.data_io import write_json
    from audit.run_manifest import finalize_run, initialize_run, sha256_file
    from diffraction.arithmetic_complexity_recovery import run as d08_recovery
    from workflow.run_phase_d_nc import verify_inputs
    project = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    predecessor_id = str(recovery["frozen_predecessor"]["run_id"])
    if project.get("state") != "PHASE_D_BLOCKED_AT_D08" or project.get("latest_run_id") != predecessor_id:
        raise SystemExit("project is not at the declared frozen D-08 blocked checkpoint")
    if project.get("blocker", {}).get("classification") != "FAIL_IMPLEMENTATION":
        raise SystemExit("D-08 root-cause audit is not FAIL_IMPLEMENTATION")
    predecessor = verify_frozen_predecessor(root, recovery)
    phase_b = verify_inputs(root, base)
    run_id, run_dir = initialize_run(root)
    statuses = {task: recovery["frozen_predecessor"]["reused_tasks"][task]["status"] for task in REUSED_TASKS}
    outputs = dict(predecessor["outputs"])
    errors = {}
    prereg = run_dir / "certificates" / "d08_estimator_preregistration.json"
    write_json(prereg, {
        "task_id": "D-08", "run_id": run_id, "status": "PREREGISTERED_BEFORE_OUTCOME",
        "estimator_definition": recovery["d08_preregistered_estimator"],
        "recovery_config": recovery_path.relative_to(root).as_posix(),
        "recovery_config_sha256": sha256_file(recovery_path),
        "estimator_code": "src/diffraction/arithmetic_complexity_recovery.py",
        "estimator_code_sha256": sha256_file(root / "src" / "diffraction" / "arithmetic_complexity_recovery.py"),
        "frozen_predecessor_manifest_sha256": predecessor["manifest_hash"],
        "frozen_predecessor_tree_inventory_sha256": predecessor["tree_hash"],
        "outcome_inspected_before_freeze": False, "posthoc_changes_permitted": False,
    })
    reuse = run_dir / "certificates" / "d01_d07_verified_hash_reuse.json"
    write_json(reuse, {
        "run_id": run_id, "status": "PASS_EXACT", "execution": "NOT_RERUN",
        "source_run_id": predecessor_id, "source_manifest_sha256": predecessor["manifest_hash"],
        "source_tree_inventory_sha256": predecessor["tree_hash"],
        "tasks": predecessor["checkpoint_records"], "source_modified": False,
    })
    for task in REUSED_TASKS:
        checkpoint(root, run_dir, run_id, task, statuses[task], outputs[task])
    context = restore_context(base, root, args, phase_b, predecessor, statuses, outputs)
    try:
        status, output = d08_recovery(base, recovery, run_dir, run_id, root, context)
        statuses["D-08"], outputs["D-08"] = status, output
        checkpoint(root, run_dir, run_id, "D-08", status, output)
    except Exception:
        statuses["D-08"] = "FAIL_IMPLEMENTATION"
        errors["D-08"] = traceback.format_exc()
    save_progress(root, run_dir, run_id, statuses, outputs, errors, "D08_COMPLETE")
    update_task_manifest(root, run_id, statuses, outputs, predecessor_id)
    if statuses["D-08"] == D08_REQUIRED_PASS:
        state = "PHASE_D_D08_RECOVERY_PASS_AWAITING_CONTINUATION"
        write_report(root, run_dir, run_id, statuses, outputs, state, errors, predecessor_id)
        update_project_state(root, run_dir, run_id, statuses, state, "D-09", predecessor_id)
        print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
        return 0
    finalize_integrity(root, run_dir, recovery, predecessor, phase_b, base)
    status = statuses["D-08"]
    state = "PHASE_D_BLOCKED_AT_D08_RECOVERY" if status in BLOCKING else "PHASE_D_D08_RECOVERY_INCONCLUSIVE"
    blocker = {"task_id": "D-08", "classification": status if status in BLOCKING else "UNRESOLVED_PREREQUISITE", "recovery_run_id": run_id}
    write_report(root, run_dir, run_id, statuses, outputs, state, errors, predecessor_id)
    update_project_state(root, run_dir, run_id, statuses, state, "D-08", predecessor_id, blocker)
    atomic_json(run_dir / "logs" / "phase_d08_recovery_execution_report.json", {
        "run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors,
        "frozen_predecessor_modified": False, "d01_d07_rerun": False,
    })
    finalize_run(run_dir, "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 2


def stage_continue(root: Path, args, base, recovery):
    from audit.run_manifest import finalize_run
    from audit.error_budget import run as d13
    from audit.validation_matrix import run as d14
    from external.circuit_laplacian_mapping import run as d11
    from external.reproduce_circuit_spectra import run as d12
    from external.reproduce_hyperbloch_dos import run as d09
    from external.reproduce_public_graphs import run as d10
    from negative_controls import FUNCTIONS as nc_functions
    from plots.export_figure_data import run as d15
    from workflow.run_phase_d_nc import verify_inputs
    if not args.run_id:
        raise SystemExit("--run-id is required for --stage continue")
    run_id = args.run_id
    verify_current_identity(root, run_id)
    run_dir = root / "results" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    project = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "RUNNING" or project.get("state") != "PHASE_D_D08_RECOVERY_PASS_AWAITING_CONTINUATION" or project.get("latest_run_id") != run_id:
        raise SystemExit("D-08 recovery run is not awaiting continuation")
    progress = json.loads((run_dir / "logs" / "phase_d08_recovery_progress.json").read_text(encoding="utf-8"))
    statuses = dict(progress["task_statuses"])
    errors = dict(progress["errors"])
    if statuses.get("D-08") != D08_REQUIRED_PASS:
        raise SystemExit("D-08 recovery did not pass")
    predecessor = verify_frozen_predecessor(root, recovery)
    phase_b = verify_inputs(root, base)
    outputs = dict(predecessor["outputs"])
    outputs["D-08"] = {
        "raw": run_dir / "raw" / "d08_arithmetic_complexity_recovery",
        "derived": run_dir / "derived" / "d08_arithmetic_complexity_recovery",
        "certificate": run_dir / "certificates" / "d08_arithmetic_complexity_recovery.json",
    }
    context = restore_context(base, root, args, phase_b, predecessor, statuses, outputs)
    context["d08_recovery"] = json.loads(outputs["D-08"]["certificate"].read_text(encoding="utf-8"))
    functions = {"D-09": d09, "D-10": d10, "D-11": d11, "D-12": d12, "D-13": d13, "D-14": d14, "D-15": d15}
    for task in D_EXECUTION[1:]:
        try:
            status, output = functions[task](base, run_dir, run_id, root, context)
            statuses[task], outputs[task] = status, output
            checkpoint(root, run_dir, run_id, task, status, output)
        except Exception:
            statuses[task] = "FAIL_IMPLEMENTATION"
            errors[task] = traceback.format_exc()
        save_progress(root, run_dir, run_id, statuses, outputs, errors, f"AFTER_{task}")
        if statuses[task] in BLOCKING:
            break
    if not any(status in BLOCKING for status in statuses.values()) and all(task in statuses for task in D_TASKS):
        for task in NC_TASKS:
            try:
                status, output = nc_functions[task](base, run_dir, run_id, root, context)
                statuses[task], outputs[task] = status, output
                checkpoint(root, run_dir, run_id, task, status, output)
            except Exception:
                statuses[task] = "FAIL_IMPLEMENTATION"
                errors[task] = traceback.format_exc()
            save_progress(root, run_dir, run_id, statuses, outputs, errors, f"AFTER_{task}")
            if statuses[task] in BLOCKING:
                break
    complete = all(task in statuses for task in D_TASKS + NC_TASKS) and not any(status in BLOCKING for status in statuses.values())
    if complete:
        finalize_matrix(root, run_dir, run_id, context, statuses, outputs)
    update_task_manifest(root, run_id, statuses, outputs, str(recovery["frozen_predecessor"]["run_id"]))
    finalize_integrity(root, run_dir, recovery, predecessor, phase_b, base)
    if complete:
        state, next_task, blocker = "PHASE_D_D15_AND_NC_COMPLETE", "G-13", None
    else:
        failed = next((task for task in D_TASKS + NC_TASKS if statuses.get(task) in BLOCKING), "UNKNOWN")
        state, next_task = "PHASE_D_OR_NC_BLOCKED", failed
        blocker = {"task_id": failed, "classification": statuses.get(failed, "FAIL_IMPLEMENTATION"), "recovery_run_id": run_id}
    write_report(root, run_dir, run_id, statuses, outputs, state, errors, str(recovery["frozen_predecessor"]["run_id"]))
    update_project_state(root, run_dir, run_id, statuses, state, next_task, str(recovery["frozen_predecessor"]["run_id"]), blocker)
    atomic_json(run_dir / "logs" / "phase_d08_recovery_execution_report.json", {
        "run_id": run_id, "state": state, "execution_order": recovery["execution_order"],
        "task_statuses": statuses, "errors": errors, "d01_d07_rerun": False,
        "frozen_predecessor_modified": False, "d15_executed": "D-15" in statuses,
        "d15_user_authorized_dependency_override": True,
    })
    finalize_run(run_dir, "COMPLETE" if complete else "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("d08", "continue"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--node-executable", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)
    base, recovery, recovery_path = load_configs(root)
    if args.stage == "d08":
        return stage_d08(root, args, base, recovery, recovery_path)
    if not args.node_executable:
        raise SystemExit("--node-executable is required for continuation")
    return stage_continue(root, args, base, recovery)


if __name__ == "__main__":
    raise SystemExit(main())
