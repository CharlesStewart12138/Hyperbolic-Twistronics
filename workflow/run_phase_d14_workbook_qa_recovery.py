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


D_TASKS = [f"D-{number:02d}" for number in range(1, 16)]
NC_TASKS = [f"NC-{number:02d}" for number in range(1, 10)]
ALL_TASKS = D_TASKS + NC_TASKS
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
    recovery_path = root / "configs" / "phase_d14_workbook_qa_recovery.yaml"
    recovery = yaml.safe_load(recovery_path.read_text(encoding="utf-8"))
    base_path = root / str(recovery["base_config"]["path"])
    if sha256_file(base_path) != recovery["base_config"]["sha256"]:
        raise RuntimeError("frozen Phase-D base config hash changed")
    return yaml.safe_load(base_path.read_text(encoding="utf-8")), recovery, recovery_path


def verify_ancestor_chain(root: Path):
    from workflow import run_phase_d09_parser_recovery as d09_recovery
    _, recovery, _ = d09_recovery.load_configs(root)
    return d09_recovery.verify_frozen_predecessor(root, recovery)


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
        raise RuntimeError("frozen D-14 QA predecessor inventory changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("task_statuses") != expected["expected_task_statuses"]:
        raise RuntimeError("frozen D-14 QA predecessor task labels changed")
    execution_report = root / str(expected["execution_report"])
    if sha256_file(execution_report) != expected["execution_report_sha256"]:
        raise RuntimeError("frozen D-09 recovery execution report changed")
    defect = recovery["frozen_d14_qa_defect"]
    old_certificate = root / str(defect["old_d14_certificate"])
    old_verification = root / str(defect["old_verification"])
    if sha256_file(old_certificate) != defect["old_d14_certificate_sha256"] or sha256_file(old_verification) != defect["old_verification_sha256"]:
        raise RuntimeError("frozen D-14 QA defect artifacts changed")
    old_audit = json.loads(old_verification.read_text(encoding="utf-8"))
    if f'"Passed / certified / external",{defect["observed_pass_count"]}' not in old_audit.get("summary_inspection", ""):
        raise RuntimeError("frozen D-14 zero-pass visual defect changed")
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
        "checkpoint_records": checkpoint_records,
        "ancestor_tree_hash": ancestor["tree_hash"],
    }


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
        if task != "D-14":
            note = f"hash-only reuse from frozen D-14 QA predecessor {predecessor_id}; not rerun"
            if note not in row["notes"]:
                row["notes"] = f"{row['notes']} {note}".strip()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_project_state(root: Path, run_dir: Path, run_id: str, statuses, state: str, blocker=None) -> None:
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
        "phase_d_task_statuses": {task: statuses[task] for task in D_TASKS},
        "negative_control_statuses": {task: statuses[task] for task in NC_TASKS},
        "d14_qa_frozen_predecessor_run_id": "ed350b4ade947be9a310532f8fd8e643030e35205b74ea14bf01ca4c48844ec4",
        "next_task": "G-13" if not blocker else "D-14",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(path, project)


def write_report(root: Path, run_dir: Path, run_id: str, statuses, outputs, state: str, errors, predecessor_id: str) -> None:
    certificate_path = outputs.get("D-14", {}).get("certificate")
    certificate = json.loads(certificate_path.read_text(encoding="utf-8")) if certificate_path and certificate_path.exists() else {}
    lines = [
        "# Phase D-14 workbook QA recovery checkpoint",
        "",
        f"- Run ID: `{run_id}`",
        f"- State: `{state}`",
        f"- Frozen predecessor: `{predecessor_id}`",
        "- Frozen predecessor modified: `false`",
        "- Tasks rerun: `D-14 only`",
        "- All other D/NC tasks: `VERIFIED_HASH_ONLY_REUSE`",
        "- Old D-14 PASS record relabelled: `false`",
        "- QA defect: wildcard pass count rendered zero and provenance dates rendered as serial numbers",
        "",
        f"- New D-14 status: `{statuses.get('D-14')}`",
        f"- Summary reconciled: `{certificate.get('summary_reconciled')}`",
        f"- Summary expected: `{json.dumps(certificate.get('summary_expected'), sort_keys=True)}`",
        f"- Summary actual: `{json.dumps(certificate.get('summary_actual'), sort_keys=True)}`",
        f"- Provenance date format: `{certificate.get('provenance_date_number_format')}`",
        f"- Formula error count: `{certificate.get('formula_error_count')}`",
        f"- Rendered sheet count: `{certificate.get('rendered_sheet_count')}`",
        "",
        f"Errors: `{json.dumps(errors, sort_keys=True)}`",
        "",
    ]
    text = "\n".join(lines)
    (run_dir / "derived" / "checkpoint_D14_workbook_qa_recovery.md").write_text(text, encoding="utf-8")
    (root / "reports" / "checkpoint_D14_workbook_qa_recovery.md").write_text(text, encoding="utf-8")


def finalize_integrity(root: Path, run_dir: Path, recovery: dict, predecessor_before, ancestor_before) -> None:
    from audit.data_io import write_json
    predecessor_after = verify_frozen_predecessor(root, recovery)
    ancestor_after = verify_ancestor_chain(root)
    checks = {
        "d14_qa_predecessor_before_equals_after": predecessor_after["inventory"] == predecessor_before["inventory"],
        "d09_ancestor_before_equals_after": ancestor_after["inventory"] == ancestor_before["inventory"],
    }
    write_json(run_dir / "certificates" / "frozen_predecessor_integrity.json", {
        "status": "PASS_EXACT" if all(checks.values()) else "FAIL_IMPLEMENTATION",
        "frozen_d14_qa_predecessor_tree_inventory_sha256": predecessor_before["tree_hash"],
        "frozen_d09_ancestor_tree_inventory_sha256": ancestor_before["tree_hash"],
        **checks,
    })
    if not all(checks.values()):
        raise RuntimeError("a frozen predecessor changed during D-14 QA recovery")


def run_recovery(root: Path, args, base: dict, recovery: dict, recovery_path: Path) -> int:
    from audit.data_io import write_json
    from audit.run_manifest import finalize_run, initialize_run, sha256_file
    from audit.validation_matrix import run as d14
    from workflow import run_phase_d08_recovery as legacy
    from workflow.run_phase_d_nc import verify_inputs

    predecessor_id = str(recovery["frozen_predecessor"]["run_id"])
    project = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    if project.get("state") != "PHASE_D_D15_AND_NC_COMPLETE" or project.get("latest_run_id") != predecessor_id:
        raise SystemExit("project is not at the declared completed D/NC checkpoint")
    predecessor = verify_frozen_predecessor(root, recovery)
    ancestor = verify_ancestor_chain(root)
    phase_b = verify_inputs(root, base)
    run_id, run_dir = initialize_run(root)
    statuses = dict(recovery["frozen_predecessor"]["expected_task_statuses"])
    outputs = dict(predecessor["outputs"])
    errors = {}
    write_json(run_dir / "certificates" / "d14_workbook_qa_preregistration.json", {
        "task_id": "D-14",
        "run_id": run_id,
        "status": "QA_CONTRACT_FROZEN_BEFORE_RECOVERY_RERUN",
        "qa_contract": recovery["d14_qa_contract"],
        "frozen_defect": recovery["frozen_d14_qa_defect"],
        "recovery_config": recovery_path.relative_to(root).as_posix(),
        "recovery_config_sha256": sha256_file(recovery_path),
        "builder_code_sha256": sha256_file(root / "src" / "audit" / "theorem_validation_workbook.mjs"),
        "acceptance_code_sha256": sha256_file(root / "src" / "audit" / "validation_matrix.py"),
        "recovery_outcome_inspected_before_freeze": False,
        "posthoc_acceptance_changes_permitted": False,
    })
    write_json(run_dir / "certificates" / "all_except_d14_verified_hash_reuse.json", {
        "run_id": run_id,
        "status": "PASS_EXACT",
        "execution": "ALL_TASKS_EXCEPT_D14_NOT_RERUN",
        "source_run_id": predecessor_id,
        "source_manifest_sha256": predecessor["manifest_hash"],
        "source_tree_inventory_sha256": predecessor["tree_hash"],
        "tasks": predecessor["checkpoint_records"],
        "source_modified": False,
    })
    for task in recovery["frozen_predecessor"]["reused_tasks"]:
        legacy.checkpoint(root, run_dir, run_id, task, statuses[task], outputs[task])
    context = legacy.restore_context(base, root, args, phase_b, predecessor, statuses, outputs)
    context["parameter_set"] = recovery_path.relative_to(root).as_posix()
    context["d09_records"] = json.loads(outputs["D-09"]["certificate"].read_text(encoding="utf-8"))["records"]
    context["d10_records"] = json.loads(outputs["D-10"]["certificate"].read_text(encoding="utf-8"))["records"]
    context["error_budget"] = pd.read_parquet(outputs["D-13"]["derived"])
    try:
        status, output = d14(base, run_dir, run_id, root, context)
        statuses["D-14"], outputs["D-14"] = status, output
        legacy.checkpoint(root, run_dir, run_id, "D-14", status, output)
        context["d14_matrix"].to_parquet(run_dir / "validation_matrix.parquet", index=False)
    except Exception:
        statuses["D-14"] = "FAIL_IMPLEMENTATION"
        errors["D-14"] = traceback.format_exc()
    update_task_manifest(root, run_id, statuses, outputs, predecessor_id)
    finalize_integrity(root, run_dir, recovery, predecessor, ancestor)
    passed = statuses.get("D-14") == "PASS_CERTIFIED"
    state = "PHASE_D_D15_AND_NC_COMPLETE" if passed else "PHASE_D_OR_NC_BLOCKED"
    blocker = None if passed else {"task_id": "D-14", "classification": statuses.get("D-14", "FAIL_IMPLEMENTATION"), "recovery_run_id": run_id}
    write_report(root, run_dir, run_id, statuses, outputs, state, errors, predecessor_id)
    update_project_state(root, run_dir, run_id, statuses, state, blocker)
    atomic_json(run_dir / "logs" / "phase_d14_workbook_qa_recovery_execution_report.json", {
        "run_id": run_id,
        "state": state,
        "task_statuses": statuses,
        "errors": errors,
        "tasks_rerun": ["D-14"],
        "all_other_tasks_hash_only_reused": True,
        "frozen_predecessor_modified": False,
        "old_d14_record_relabelled": False,
    })
    finalize_run(run_dir, "COMPLETE" if passed else "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 0 if passed else 2


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
