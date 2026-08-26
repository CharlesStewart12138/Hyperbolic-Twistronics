from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml


REUSED_TASKS = [f"D-{number:02d}" for number in range(1, 9)]
D_TASKS = [f"D-{number:02d}" for number in range(1, 16)]
D_EXECUTION = [f"D-{number:02d}" for number in range(9, 16)]
NC_TASKS = [f"NC-{number:02d}" for number in range(1, 10)]
BLOCKING = {"FAIL_THEORY", "FAIL_IMPLEMENTATION"}


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
    recovery_path = root / "configs" / "phase_d09_parser_recovery.yaml"
    recovery = yaml.safe_load(recovery_path.read_text(encoding="utf-8"))
    base_path = root / str(recovery["base_config"]["path"])
    if sha256_file(base_path) != recovery["base_config"]["sha256"]:
        raise RuntimeError("frozen Phase-D base config hash changed")
    return yaml.safe_load(base_path.read_text(encoding="utf-8")), recovery, recovery_path


def verify_ancestor_chain(root: Path):
    from workflow import run_phase_d08_theorem_contract as theorem
    _, recovery, _ = theorem.load_configs(root)
    return theorem.verify_frozen_predecessor(root, recovery)


def verify_frozen_predecessor(root: Path, recovery: dict):
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest
    from workflow import run_phase_d08_recovery as legacy
    expected = recovery["frozen_predecessor"]
    source = root / "results" / str(expected["run_id"])
    manifest_path = source / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    inventory = _tree_inventory(source)
    tree_hash = inventory_digest(inventory)
    if manifest_hash != expected["manifest_sha256"] or tree_hash != expected["tree_inventory_sha256"] or len(inventory) != int(expected["file_count"]):
        raise RuntimeError("frozen D-09 predecessor inventory changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_statuses = {task: row["status"] for task, row in expected["reused_tasks"].items()}
    expected_statuses["D-09"] = expected["frozen_d09_status"]
    if manifest.get("status") != "INCOMPLETE" or manifest.get("task_statuses") != expected_statuses:
        raise RuntimeError("frozen D-09 predecessor task labels changed")
    report_path = root / str(expected["execution_report"])
    if sha256_file(report_path) != expected["execution_report_sha256"]:
        raise RuntimeError("frozen D-09 execution report changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("task_statuses", {}).get("D-09") != "FAIL_IMPLEMENTATION":
        raise RuntimeError("frozen D-09 failure status changed")
    if expected["frozen_d09_exception"] not in report.get("errors", {}).get("D-09", ""):
        raise RuntimeError("frozen D-09 exception record changed")
    outputs, checkpoint_records = {}, {}
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
            actual = legacy.artifact_digest(path)
            if any(actual.get(name) != record.get(name) for name in actual):
                raise RuntimeError(f"frozen predecessor output digest changed: {task}/{key}")
            task_outputs[key] = path
        outputs[task] = task_outputs
        checkpoint_records[task] = {
            "checkpoint": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "status": row["status"],
            "outputs": checkpoint["outputs"],
        }
    ancestor = verify_ancestor_chain(root)
    return {
        "source": source,
        "manifest": manifest,
        "manifest_hash": manifest_hash,
        "inventory": inventory,
        "tree_hash": tree_hash,
        "outputs": outputs,
        "statuses": {task: expected_statuses[task] for task in REUSED_TASKS},
        "checkpoint_records": checkpoint_records,
        "frozen_d09_report": report,
        "ancestor_tree_hash": ancestor["tree_hash"],
    }


def save_progress(root: Path, run_dir: Path, run_id: str, statuses, outputs, errors, stage: str) -> None:
    atomic_json(run_dir / "logs" / "phase_d09_parser_recovery_progress.json", {
        "run_id": run_id,
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_statuses": statuses,
        "outputs": {task: {key: value.relative_to(root).as_posix() for key, value in output.items()} for task, output in outputs.items()},
        "errors": errors,
    })


def update_task_manifest(root: Path, run_id: str, statuses, outputs, predecessor_id: str) -> None:
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
            note = f"hash-only reuse from frozen D-09 predecessor {predecessor_id}; not rerun"
            if note not in row["notes"]:
                row["notes"] = f"{row['notes']} {note}".strip()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_project_state(root: Path, run_dir: Path, run_id: str, statuses, state: str, next_task: str, predecessor_id: str, blocker=None) -> None:
    path = root / "PROJECT_STATE.json"
    project = json.loads(path.read_text(encoding="utf-8"))
    prior = project.pop("blocker", None)
    history = project.setdefault("historical_blockers", [])
    if prior and prior not in history:
        history.append(prior)
    if blocker:
        project["blocker"] = blocker
    project.update({
        "state": state,
        "current_phase": "D/NC",
        "latest_run_id": run_id,
        "latest_run_directory": run_dir.relative_to(root).as_posix(),
        "phase_d_task_statuses": {task: statuses[task] for task in D_TASKS if task in statuses},
        "negative_control_statuses": {task: statuses[task] for task in NC_TASKS if task in statuses},
        "d09_frozen_predecessor_run_id": predecessor_id,
        "next_task": next_task,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(path, project)


def write_report(root: Path, run_dir: Path, run_id: str, statuses, outputs, state: str, errors, predecessor_id: str) -> None:
    d09_path = outputs.get("D-09", {}).get("certificate")
    d09 = json.loads(d09_path.read_text(encoding="utf-8")) if d09_path and d09_path.exists() else {}
    lines = [
        "# Phase D-09 parser recovery checkpoint",
        "",
        f"- Run ID: `{run_id}`",
        f"- State: `{state}`",
        f"- Frozen predecessor: `{predecessor_id}`",
        "- Frozen predecessor modified: `false`",
        "- Frozen predecessor D-09 failure relabelled: `false`",
        "- D-01--D-08 execution: `NOT_RERUN_HASH_ONLY_REUSE`",
        "- Parser schema: `hypercells_tess_content_v1`",
        "- Parser contract frozen before scientific rerun: `true`",
        "",
        "| Task | Status | Certificate |",
        "|---|---|---|",
    ]
    for task in D_TASKS + NC_TASKS:
        certificate = outputs.get(task, {}).get("certificate")
        lines.append(f"| {task} | {statuses.get(task, 'NOT_STARTED')} | `{certificate.relative_to(root).as_posix() if certificate else '-'}` |")
    if d09:
        lines.extend(["", "## D-09 parser outcome", ""])
        for record in d09.get("records", []):
            lines.append(
                f"- `{Path(record['source']).name}`: vertices `{record['vertex_count']}`, "
                f"edges `{record['edge_count']}`, degree `{record['minimum_degree']}..{record['maximum_degree']}`, "
                f"parser `{record['parser_schema_version']}`"
            )
        lines.extend([
            f"- Raw external files preserved: `{d09.get('raw_external_files_preserved')}`",
            f"- Parser diagnostics preserved: `{d09.get('parser_diagnostics_preserved')}`",
        ])
    lines.extend(["", f"Errors: `{json.dumps(errors, sort_keys=True)}`", ""])
    text = "\n".join(lines)
    (run_dir / "derived" / "checkpoint_D09_parser_recovery.md").write_text(text, encoding="utf-8")
    (root / "reports" / "checkpoint_D09_parser_recovery.md").write_text(text, encoding="utf-8")


def finalize_integrity(root: Path, run_dir: Path, recovery: dict, predecessor_before, ancestor_before, phase_b_before, base: dict) -> None:
    from audit.data_io import write_json
    from workflow.run_phase_d_nc import verify_inputs
    predecessor_after = verify_frozen_predecessor(root, recovery)
    ancestor_after = verify_ancestor_chain(root)
    phase_b_after = verify_inputs(root, base)
    checks = {
        "d09_predecessor_before_equals_after": predecessor_after["inventory"] == predecessor_before["inventory"],
        "d08_ancestor_before_equals_after": ancestor_after["inventory"] == ancestor_before["inventory"],
        "phase_b_before_equals_after": phase_b_after["inventory"] == phase_b_before["inventory"],
    }
    write_json(run_dir / "certificates" / "frozen_predecessor_integrity.json", {
        "status": "PASS_EXACT" if all(checks.values()) else "FAIL_IMPLEMENTATION",
        "frozen_d09_predecessor_run_id": recovery["frozen_predecessor"]["run_id"],
        "frozen_d09_predecessor_tree_inventory_sha256": predecessor_before["tree_hash"],
        "frozen_d08_ancestor_tree_inventory_sha256": ancestor_before["tree_hash"],
        "phase_b_tree_inventory_sha256": base["source_phase_b"]["tree_inventory_sha256"],
        **checks,
    })
    if not all(checks.values()):
        raise RuntimeError("a frozen predecessor changed during D-09 parser recovery")


def run_recovery(root: Path, args, base: dict, recovery: dict, recovery_path: Path) -> int:
    from audit.data_io import write_json
    from audit.error_budget import run as d13
    from audit.run_manifest import finalize_run, initialize_run, sha256_file
    from audit.validation_matrix import run as d14
    from external.circuit_laplacian_mapping import run as d11
    from external.reproduce_circuit_spectra import run as d12
    from external.reproduce_hyperbloch_dos import run as d09
    from external.reproduce_public_graphs import run as d10
    from negative_controls import FUNCTIONS as nc_functions
    from plots.export_figure_data import run as d15
    from workflow import run_phase_d08_recovery as legacy
    from workflow.run_phase_d_nc import verify_inputs

    predecessor_id = str(recovery["frozen_predecessor"]["run_id"])
    project = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    if project.get("state") != "PHASE_D_OR_NC_BLOCKED" or project.get("latest_run_id") != predecessor_id:
        raise SystemExit("project is not at the declared frozen D-09 blocked checkpoint")
    blocker = project.get("blocker", {})
    if blocker.get("task_id") != "D-09" or blocker.get("classification") != "FAIL_IMPLEMENTATION":
        raise SystemExit("current blocker is not the declared D-09 FAIL_IMPLEMENTATION")

    predecessor = verify_frozen_predecessor(root, recovery)
    ancestor = verify_ancestor_chain(root)
    phase_b = verify_inputs(root, base)
    run_id, run_dir = initialize_run(root)
    statuses = dict(predecessor["statuses"])
    outputs = dict(predecessor["outputs"])
    errors = {}

    write_json(run_dir / "certificates" / "d09_parser_preregistration.json", {
        "task_id": "D-09",
        "run_id": run_id,
        "status": "PARSER_CONTRACT_FROZEN_BEFORE_SCIENTIFIC_RERUN",
        "parser_contract": recovery["parser_contract"],
        "recovery_config": recovery_path.relative_to(root).as_posix(),
        "recovery_config_sha256": sha256_file(recovery_path),
        "parser_code": "src/external/reproduce_hyperbloch_dos.py",
        "parser_code_sha256": sha256_file(root / "src" / "external" / "reproduce_hyperbloch_dos.py"),
        "regression_tests": "tests/test_d09_parser_recovery.py",
        "regression_tests_sha256": sha256_file(root / "tests" / "test_d09_parser_recovery.py"),
        "scientific_rerun_executed_before_freeze": False,
        "posthoc_parser_changes_permitted": False,
    })
    write_json(run_dir / "certificates" / "d01_d08_verified_hash_reuse.json", {
        "run_id": run_id,
        "status": "PASS_EXACT",
        "execution": "D01_D08_NOT_RERUN",
        "source_run_id": predecessor_id,
        "source_manifest_sha256": predecessor["manifest_hash"],
        "source_tree_inventory_sha256": predecessor["tree_hash"],
        "tasks": predecessor["checkpoint_records"],
        "frozen_d09_failure_report": recovery["frozen_predecessor"]["execution_report"],
        "frozen_d09_failure_report_sha256": recovery["frozen_predecessor"]["execution_report_sha256"],
        "source_modified": False,
    })
    for task in REUSED_TASKS:
        legacy.checkpoint(root, run_dir, run_id, task, statuses[task], outputs[task])

    context = legacy.restore_context(base, root, args, phase_b, predecessor, statuses, outputs)
    context["parameter_set"] = recovery_path.relative_to(root).as_posix()
    context["d09_parser_contract"] = recovery["parser_contract"]
    context["d08_theorem_contract"] = json.loads(outputs["D-08"]["certificate"].read_text(encoding="utf-8"))
    functions = {"D-09": d09, "D-10": d10, "D-11": d11, "D-12": d12, "D-13": d13, "D-14": d14, "D-15": d15}
    for task in D_EXECUTION:
        try:
            status, output = functions[task](base, run_dir, run_id, root, context)
            statuses[task], outputs[task] = status, output
            legacy.checkpoint(root, run_dir, run_id, task, status, output)
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
                legacy.checkpoint(root, run_dir, run_id, task, status, output)
            except Exception:
                statuses[task] = "FAIL_IMPLEMENTATION"
                errors[task] = traceback.format_exc()
            save_progress(root, run_dir, run_id, statuses, outputs, errors, f"AFTER_{task}")
            if statuses[task] in BLOCKING:
                break

    complete = all(task in statuses for task in D_TASKS + NC_TASKS) and not any(status in BLOCKING for status in statuses.values())
    if complete:
        legacy.finalize_matrix(root, run_dir, run_id, context, statuses, outputs)
    update_task_manifest(root, run_id, statuses, outputs, predecessor_id)
    finalize_integrity(root, run_dir, recovery, predecessor, ancestor, phase_b, base)
    if complete:
        state, next_task, blocker_record = "PHASE_D_D15_AND_NC_COMPLETE", "G-13", None
    else:
        failed = next((task for task in D_TASKS + NC_TASKS if statuses.get(task) in BLOCKING), "UNKNOWN")
        state, next_task = "PHASE_D_OR_NC_BLOCKED", failed
        blocker_record = {"task_id": failed, "classification": statuses.get(failed, "FAIL_IMPLEMENTATION"), "recovery_run_id": run_id}
    write_report(root, run_dir, run_id, statuses, outputs, state, errors, predecessor_id)
    update_project_state(root, run_dir, run_id, statuses, state, next_task, predecessor_id, blocker_record)
    atomic_json(run_dir / "logs" / "phase_d09_parser_recovery_execution_report.json", {
        "run_id": run_id,
        "state": state,
        "execution_order": recovery["execution_order"],
        "task_statuses": statuses,
        "errors": errors,
        "d01_d08_rerun": False,
        "d09_rerun_from_scratch": True,
        "frozen_d09_failure_relabelled": False,
        "frozen_predecessor_modified": False,
        "parser_contract_frozen_before_scientific_rerun": True,
        "d15_executed": "D-15" in statuses,
        "d15_user_authorized_dependency_override": True,
    })
    finalize_run(run_dir, "COMPLETE" if complete else "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 0 if complete else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--node-executable", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)
    base, recovery, recovery_path = load_configs(root)
    return run_recovery(root, args, base, recovery, recovery_path)


if __name__ == "__main__":
    raise SystemExit(main())
