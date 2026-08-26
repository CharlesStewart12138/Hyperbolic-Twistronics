from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml


REUSED_TASKS = [f"D-{number:02d}" for number in range(1, 8)]
D_TASKS = [f"D-{number:02d}" for number in range(1, 16)]
D_EXECUTION = [f"D-{number:02d}" for number in range(8, 16)]
NC_TASKS = [f"NC-{number:02d}" for number in range(1, 10)]
BLOCKING = {"FAIL_THEORY", "FAIL_IMPLEMENTATION"}
D08_REQUIRED_PASS = "PASS_CERTIFIED"


def add_path(root: Path) -> None:
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_configs(root: Path):
    from audit.run_manifest import sha256_file
    recovery_path = root / "configs" / "phase_d08_theorem_contract_recovery.yaml"
    recovery = yaml.safe_load(recovery_path.read_text(encoding="utf-8"))
    base_path = root / str(recovery["base_config"]["path"])
    if sha256_file(base_path) != recovery["base_config"]["sha256"]:
        raise RuntimeError("frozen Phase-D base config hash changed")
    return yaml.safe_load(base_path.read_text(encoding="utf-8")), recovery, recovery_path


def verify_frozen_predecessor(root: Path, recovery: dict):
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest
    from workflow import run_phase_d08_recovery as legacy
    expected = recovery["frozen_predecessor"]
    source = root / "results" / str(expected["run_id"])
    manifest_hash = sha256_file(source / "manifest.json")
    inventory = _tree_inventory(source)
    tree_hash = inventory_digest(inventory)
    if manifest_hash != expected["manifest_sha256"] or tree_hash != expected["tree_inventory_sha256"] or len(inventory) != int(expected["file_count"]):
        raise RuntimeError("frozen D-08 inconclusive predecessor inventory changed")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected_statuses = {task: row["status"] for task, row in expected["reused_tasks"].items()}
    expected_statuses["D-08"] = expected["frozen_d08_status"]
    if manifest.get("status") != "INCOMPLETE" or manifest.get("task_statuses") != expected_statuses:
        raise RuntimeError("frozen D-08 inconclusive manifest status changed")
    d08_checkpoint = source / "certificates" / "task_checkpoints" / "D-08.json"
    d08_certificate = root / str(expected["d08_certificate"])
    d08_preregistration = source / "certificates" / "d08_estimator_preregistration.json"
    if sha256_file(d08_checkpoint) != expected["d08_checkpoint_sha256"]:
        raise RuntimeError("frozen D-08 checkpoint changed")
    if sha256_file(d08_certificate) != expected["d08_certificate_sha256"]:
        raise RuntimeError("frozen D-08 certificate changed")
    if sha256_file(d08_preregistration) != expected["d08_preregistration_sha256"]:
        raise RuntimeError("frozen D-08 preregistration changed")
    frozen_d08 = json.loads(d08_certificate.read_text(encoding="utf-8"))
    metrics = expected["frozen_metrics"]
    if frozen_d08.get("status") != "INCONCLUSIVE":
        raise RuntimeError("frozen D-08 status changed")
    for key, source_key in (
        ("fixed_ratio_extrapolation", "primary_extrapolated_exponent"),
        ("fixed_ratio_tail_center", "primary_tail_center"),
        ("dyadic_upper_envelope_extrapolation", "upper_envelope_extrapolated_exponent"),
        ("dyadic_lower_envelope_extrapolation", "lower_envelope_extrapolated_exponent"),
    ):
        if abs(float(metrics[key]) - float(frozen_d08[source_key])) > 1.0e-12:
            raise RuntimeError(f"frozen predecessor metric changed: {key}")
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
            "status": row["status"], "outputs": checkpoint["outputs"],
        }
    return {
        "source": source, "manifest": manifest, "manifest_hash": manifest_hash,
        "inventory": inventory, "tree_hash": tree_hash, "outputs": outputs,
        "statuses": expected_statuses, "checkpoint_records": checkpoint_records,
        "frozen_d08": frozen_d08,
    }


def save_progress(root: Path, run_dir: Path, run_id: str, statuses, outputs, errors, stage: str):
    from workflow import run_phase_d08_recovery as legacy
    legacy.atomic_json(run_dir / "logs" / "phase_d08_theorem_contract_progress.json", {
        "run_id": run_id, "stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_statuses": statuses,
        "outputs": {task: {key: value.relative_to(root).as_posix() for key, value in out.items()} for task, out in outputs.items()},
        "errors": errors,
    })


def write_report(root: Path, run_dir: Path, run_id: str, statuses, outputs, state: str, errors, predecessor_id: str):
    d08_path = outputs.get("D-08", {}).get("certificate")
    d08 = json.loads(d08_path.read_text(encoding="utf-8")) if d08_path and d08_path.exists() else {}
    lines = [
        "# Phase D-08 theorem-contract recovery checkpoint", "", f"- Run ID: `{run_id}`", f"- State: `{state}`",
        f"- Frozen predecessor: `{predecessor_id}`", "- Frozen predecessor labels modified: `false`",
        "- Theorem contract: `A_LOG_ASYMPTOTIC_EXPONENT`", "- Strong regular variation registered: `false`",
        "- D-01--D-07 execution: `NOT_RERUN_HASH_ONLY_REUSE`", "- Post-hoc window selection: `false`",
        "- D-15 user-authorized order override recorded: `true`", "", "| Task | Status | Certificate |", "|---|---|---|",
    ]
    for task in D_TASKS + NC_TASKS:
        cert = outputs.get(task, {}).get("certificate")
        lines.append(f"| {task} | {statuses.get(task, 'NOT_STARTED')} | `{cert.relative_to(root).as_posix() if cert else '-'}` |")
    if d08:
        lines.extend([
            "", "## Theorem-matched D-08 outcome", "",
            f"- normalized residual extrapolation: `{d08.get('normalized_residual_extrapolation')}`",
            f"- upper residual extrapolation: `{d08.get('upper_normalized_residual_extrapolation')}`",
            f"- lower residual extrapolation: `{d08.get('lower_normalized_residual_extrapolation')}`",
            f"- fixed-ratio c=2 diagnostic: `{d08.get('fixed_ratio_c2_extrapolation')}` (not an acceptance condition)",
            f"- G-11 frozen extrapolation: `{d08.get('g11_frozen_extrapolation')}`",
            f"- supports exponent four: `{d08.get('data_support_log_asymptotic_exponent_four')}`",
            f"- supports smooth doubling regular variation: `{d08.get('data_support_strong_regular_variation_under_doubling')}`",
            f"- acceptance checks: `{json.dumps(d08.get('acceptance_checks', {}), sort_keys=True)}`",
        ])
    lines.extend(["", f"Errors: `{json.dumps(errors, sort_keys=True)}`", ""])
    text = "\n".join(lines)
    (run_dir / "derived" / "checkpoint_D08_theorem_contract.md").write_text(text, encoding="utf-8")
    (root / "reports" / "checkpoint_D08_theorem_contract.md").write_text(text, encoding="utf-8")


def finalize_integrity(root: Path, run_dir: Path, recovery, predecessor_before, phase_b_before, base):
    from audit.data_io import write_json
    from workflow.run_phase_d_nc import verify_inputs
    predecessor_after = verify_frozen_predecessor(root, recovery)
    phase_b_after = verify_inputs(root, base)
    unchanged = predecessor_after["inventory"] == predecessor_before["inventory"] and phase_b_after["inventory"] == phase_b_before["inventory"]
    write_json(run_dir / "certificates" / "frozen_predecessor_integrity.json", {
        "status": "PASS_EXACT" if unchanged else "FAIL_IMPLEMENTATION",
        "predecessor_run_id": recovery["frozen_predecessor"]["run_id"],
        "predecessor_before_equals_after": predecessor_after["inventory"] == predecessor_before["inventory"],
        "predecessor_tree_inventory_sha256": predecessor_before["tree_hash"],
        "phase_b_before_equals_after": phase_b_after["inventory"] == phase_b_before["inventory"],
        "phase_b_tree_inventory_sha256": base["source_phase_b"]["tree_inventory_sha256"],
    })
    if not unchanged:
        raise RuntimeError("a frozen predecessor changed during theorem-contract recovery")


def stage_d08(root: Path, args, base, recovery, recovery_path):
    from audit.data_io import write_json
    from audit.run_manifest import finalize_run, initialize_run, sha256_file
    from diffraction.arithmetic_complexity_theorem_contract import run as d08_theorem_matched
    from workflow import run_phase_d08_recovery as legacy
    from workflow.run_phase_d_nc import verify_inputs
    project = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    predecessor_id = str(recovery["frozen_predecessor"]["run_id"])
    if project.get("state") != "PHASE_D_D08_RECOVERY_INCONCLUSIVE" or project.get("latest_run_id") != predecessor_id:
        raise SystemExit("project is not at the declared D-08 inconclusive checkpoint")
    if project.get("blocker", {}).get("classification") != "UNRESOLVED_PREREQUISITE":
        raise SystemExit("D-08 inconclusive prerequisite state changed")
    predecessor = verify_frozen_predecessor(root, recovery)
    phase_b = verify_inputs(root, base)
    manuscript = recovery["theorem_contract"]["manuscript"]
    if sha256_file(root / manuscript["path"]) != manuscript["sha256"]:
        raise RuntimeError("frozen manuscript hash changed")
    run_id, run_dir = initialize_run(root)
    statuses = {task: recovery["frozen_predecessor"]["reused_tasks"][task]["status"] for task in REUSED_TASKS}
    outputs = dict(predecessor["outputs"])
    errors = {}
    write_json(run_dir / "certificates" / "d08_theorem_contract_audit.json", {
        "task_id": "D-08", "run_id": run_id, "status": "CONTRACT_FROZEN_BEFORE_NEW_OUTCOME",
        "contract": recovery["theorem_contract"],
        "audit_conclusion": "A_LOG_ASYMPTOTIC_EXPONENT",
        "registered_theorem": "q_j=|theta_j|^(-4+o(1)) equivalently log(q_j)/log(|theta_j|^-1)->4",
        "stronger_regular_variation_statement_registered": False,
        "manuscript_sha256": manuscript["sha256"],
        "g10_sha256": recovery["theorem_contract"]["g10"]["sha256"],
        "g11_sha256": recovery["theorem_contract"]["g11"]["sha256"],
        "frozen_fixed_ratio_diagnostic_sha256": recovery["frozen_predecessor"]["d08_certificate_sha256"],
        "new_theorem_matched_outcome_inspected_before_freeze": False,
    })
    write_json(run_dir / "certificates" / "d08_theorem_matched_preregistration.json", {
        "task_id": "D-08", "run_id": run_id, "status": "PREREGISTERED_BEFORE_NEW_OUTCOME",
        "test_definition": recovery["d08_preregistered_theorem_matched_test"],
        "recovery_config": recovery_path.relative_to(root).as_posix(),
        "recovery_config_sha256": sha256_file(recovery_path),
        "estimator_code": "src/diffraction/arithmetic_complexity_theorem_contract.py",
        "estimator_code_sha256": sha256_file(root / "src" / "diffraction" / "arithmetic_complexity_theorem_contract.py"),
        "fixed_ratio_acceptance_role_for_theorem_A": "NONE",
        "posthoc_changes_permitted": False,
    })
    write_json(run_dir / "certificates" / "d01_d07_and_prior_d08_verified_hash_reuse.json", {
        "run_id": run_id, "status": "PASS_EXACT", "execution": "D01_D07_NOT_RERUN_D08_DIAGNOSTIC_NOT_RECALCULATED",
        "source_run_id": predecessor_id, "source_manifest_sha256": predecessor["manifest_hash"],
        "source_tree_inventory_sha256": predecessor["tree_hash"], "tasks": predecessor["checkpoint_records"],
        "frozen_d08_certificate_sha256": recovery["frozen_predecessor"]["d08_certificate_sha256"], "source_modified": False,
    })
    for task in REUSED_TASKS:
        legacy.checkpoint(root, run_dir, run_id, task, statuses[task], outputs[task])
    context = legacy.restore_context(base, root, args, phase_b, predecessor, statuses, outputs)
    context["parameter_set"] = recovery_path.relative_to(root).as_posix()
    try:
        status, output = d08_theorem_matched(base, recovery, run_dir, run_id, root, context)
        statuses["D-08"], outputs["D-08"] = status, output
        legacy.checkpoint(root, run_dir, run_id, "D-08", status, output)
    except Exception:
        statuses["D-08"] = "FAIL_IMPLEMENTATION"
        errors["D-08"] = traceback.format_exc()
    save_progress(root, run_dir, run_id, statuses, outputs, errors, "D08_THEOREM_MATCHED_COMPLETE")
    legacy.update_task_manifest(root, run_id, statuses, outputs, predecessor_id)
    if statuses["D-08"] == D08_REQUIRED_PASS:
        state = "PHASE_D_D08_THEOREM_MATCHED_PASS_AWAITING_CONTINUATION"
        write_report(root, run_dir, run_id, statuses, outputs, state, errors, predecessor_id)
        legacy.update_project_state(root, run_dir, run_id, statuses, state, "D-09", predecessor_id)
        print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
        return 0
    finalize_integrity(root, run_dir, recovery, predecessor, phase_b, base)
    status = statuses["D-08"]
    state = "PHASE_D_BLOCKED_AT_D08_THEOREM_MATCHED" if status in BLOCKING else "PHASE_D_D08_THEOREM_MATCHED_INCONCLUSIVE"
    blocker = {"task_id": "D-08", "classification": status if status in BLOCKING else "UNRESOLVED_PREREQUISITE", "recovery_run_id": run_id}
    write_report(root, run_dir, run_id, statuses, outputs, state, errors, predecessor_id)
    legacy.update_project_state(root, run_dir, run_id, statuses, state, "D-08", predecessor_id, blocker)
    legacy.atomic_json(run_dir / "logs" / "phase_d08_theorem_contract_execution_report.json", {
        "run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors,
        "frozen_predecessor_modified": False, "d01_d07_rerun": False, "fixed_ratio_recalculated": False,
    })
    finalize_run(run_dir, "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 2


def stage_continue(root: Path, args, base, recovery):
    from audit.error_budget import run as d13
    from audit.run_manifest import finalize_run
    from audit.validation_matrix import run as d14
    from external.circuit_laplacian_mapping import run as d11
    from external.reproduce_circuit_spectra import run as d12
    from external.reproduce_hyperbloch_dos import run as d09
    from external.reproduce_public_graphs import run as d10
    from negative_controls import FUNCTIONS as nc_functions
    from plots.export_figure_data import run as d15
    from workflow import run_phase_d08_recovery as legacy
    from workflow.run_phase_d_nc import verify_inputs
    if not args.run_id:
        raise SystemExit("--run-id is required for --stage continue")
    run_id = args.run_id
    legacy.verify_current_identity(root, run_id)
    run_dir = root / "results" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    project = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    expected_state = "PHASE_D_D08_THEOREM_MATCHED_PASS_AWAITING_CONTINUATION"
    if manifest.get("status") != "RUNNING" or project.get("state") != expected_state or project.get("latest_run_id") != run_id:
        raise SystemExit("D-08 theorem-matched run is not awaiting continuation")
    progress = json.loads((run_dir / "logs" / "phase_d08_theorem_contract_progress.json").read_text(encoding="utf-8"))
    statuses, errors = dict(progress["task_statuses"]), dict(progress["errors"])
    if statuses.get("D-08") != D08_REQUIRED_PASS:
        raise SystemExit("D-08 theorem-matched test did not pass")
    predecessor = verify_frozen_predecessor(root, recovery)
    phase_b = verify_inputs(root, base)
    outputs = dict(predecessor["outputs"])
    outputs["D-08"] = {
        "raw": run_dir / "raw" / "d08_theorem_matched",
        "derived": run_dir / "derived" / "d08_theorem_matched",
        "certificate": run_dir / "certificates" / "d08_theorem_matched.json",
    }
    context = legacy.restore_context(base, root, args, phase_b, predecessor, statuses, outputs)
    context["d08_theorem_contract"] = json.loads(outputs["D-08"]["certificate"].read_text(encoding="utf-8"))
    context["parameter_set"] = "configs/phase_d08_theorem_contract_recovery.yaml"
    functions = {"D-09": d09, "D-10": d10, "D-11": d11, "D-12": d12, "D-13": d13, "D-14": d14, "D-15": d15}
    for task in D_EXECUTION[1:]:
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
    legacy.update_task_manifest(root, run_id, statuses, outputs, str(recovery["frozen_predecessor"]["run_id"]))
    finalize_integrity(root, run_dir, recovery, predecessor, phase_b, base)
    if complete:
        state, next_task, blocker = "PHASE_D_D15_AND_NC_COMPLETE", "G-13", None
    else:
        failed = next((task for task in D_TASKS + NC_TASKS if statuses.get(task) in BLOCKING), "UNKNOWN")
        state, next_task = "PHASE_D_OR_NC_BLOCKED", failed
        blocker = {"task_id": failed, "classification": statuses.get(failed, "FAIL_IMPLEMENTATION"), "recovery_run_id": run_id}
    predecessor_id = str(recovery["frozen_predecessor"]["run_id"])
    write_report(root, run_dir, run_id, statuses, outputs, state, errors, predecessor_id)
    legacy.update_project_state(root, run_dir, run_id, statuses, state, next_task, predecessor_id, blocker)
    legacy.atomic_json(run_dir / "logs" / "phase_d08_theorem_contract_execution_report.json", {
        "run_id": run_id, "state": state, "execution_order": recovery["execution_order"],
        "task_statuses": statuses, "errors": errors, "d01_d07_rerun": False,
        "fixed_ratio_recalculated": False, "frozen_predecessor_modified": False,
        "d15_executed": "D-15" in statuses, "d15_user_authorized_dependency_override": True,
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
