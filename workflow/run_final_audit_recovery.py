from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml


SCIENTIFIC_TASKS = [
    "G-13",
    "G-14",
    "G-15",
    "S-17",
    "S-18",
    "S-19",
    "S-20",
    "S-21",
    "S-22",
    "S-23",
    "S-24",
]
TERMINAL = {
    "PASS_EXACT",
    "PASS_CERTIFIED",
    "PASS_CONVERGED",
    "PASS_EXTERNAL",
    "INCONCLUSIVE",
    "FAIL_EXPECTED",
}


def add_path(root: Path) -> None:
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def artifact_digest(path: Path) -> dict[str, object]:
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest

    if path.is_file():
        return {"kind": "file", "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if not path.is_dir():
        raise FileNotFoundError(path)
    inventory = _tree_inventory(path)
    return {
        "kind": "directory",
        "tree_inventory_sha256": inventory_digest(inventory),
        "file_count": len(inventory),
    }


def _read_manifest(root: Path) -> list[dict[str, str]]:
    with (root / "TASK_MANIFEST.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_frozen_scientific_run(root: Path, recovery: dict) -> dict[str, object]:
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest

    section = recovery["frozen_scientific_run"]
    run_id = str(section["run_id"])
    source = root / "results" / run_id
    inventory = _tree_inventory(source)
    checks: dict[str, bool] = {
        "source_directory_exists": source.is_dir(),
        "manifest_sha256": sha256_file(source / "manifest.json") == str(section["manifest_sha256"]),
        "tree_inventory_sha256": inventory_digest(inventory) == str(section["tree_inventory_sha256"]),
        "file_count": len(inventory) == int(section["file_count"]),
        "validation_matrix_sha256": sha256_file(root / str(section["validation_matrix"]))
        == str(section["validation_matrix_sha256"]),
        "failure_log_sha256": sha256_file(root / str(section["failure_log"]))
        == str(section["failure_log_sha256"]),
        "task_manifest_sha256": sha256_file(root / "TASK_MANIFEST.csv")
        == str(section["task_manifest_sha256"]),
    }

    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected_statuses = {str(key): str(value) for key, value in section["task_statuses"].items()}
    checks["frozen_manifest_status_is_original_incomplete"] = source_manifest.get("status") == "INCOMPLETE"
    checks["frozen_manifest_scientific_statuses"] = source_manifest.get("task_statuses") == expected_statuses

    task_rows = _read_manifest(root)
    current_statuses = {row["task_id"]: row["status"] for row in task_rows}
    blank_derived = {row["task_id"] for row in task_rows if not row["derived_output"]}
    allowed_blank = set(recovery["repair_contract"]["tasks_with_blank_legacy_derived_output"])
    checks.update(
        {
            "task_count_88": len(task_rows) == 88,
            "all_tasks_terminal": all(status in TERMINAL for status in current_statuses.values()),
            "scientific_statuses_unchanged": all(
                current_statuses.get(task) == status for task, status in expected_statuses.items()
            ),
            "legacy_blank_derived_exact_set": blank_derived == allowed_blank,
            "legacy_blank_derived_is_phase_i_only": allowed_blank
            == {f"I-{index:02d}" for index in range(1, 11)},
        }
    )

    checkpoint_audit: dict[str, object] = {}
    for task in SCIENTIFIC_TASKS:
        checkpoint = source / "certificates" / "task_checkpoints" / f"{task}.json"
        expected_checkpoint_sha = str(section["checkpoints"][task])
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        output_checks: dict[str, bool] = {}
        for key, expected in data["outputs"].items():
            output = (root / str(expected["path"])).resolve()
            try:
                output.relative_to(source.resolve())
                inside_source = True
            except ValueError:
                inside_source = False
            actual = artifact_digest(output)
            output_checks[key] = inside_source and actual == {
                field: expected[field]
                for field in actual
            }
        checkpoint_checks = {
            "checkpoint_sha256": sha256_file(checkpoint) == expected_checkpoint_sha,
            "run_id": data.get("run_id") == run_id,
            "task_id": data.get("task_id") == task,
            "status": data.get("status") == expected_statuses[task],
            "outputs": all(output_checks.values()),
        }
        checkpoint_audit[task] = {
            "checks": checkpoint_checks,
            "output_checks": output_checks,
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        checks[f"checkpoint_{task}"] = all(checkpoint_checks.values())

    final_targets = [
        root / "FINAL_VALIDATION_STATUS.json",
        root / "reports" / "validation_report.md",
        root / "reports" / "validation_report.tex",
        root / "reports" / "theorem_validation_matrix.xlsx",
        root / "reports" / "error_budget.parquet",
        root / "reports" / "publication_figure_data_index.csv",
        root / "figures" / "final_validation",
    ]
    checks["no_partial_final_outputs_before_recovery"] = not any(path.exists() for path in final_targets)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"frozen scientific recovery verification failed: {failed}")
    return {
        "source_run_id": run_id,
        "source_tree_inventory_sha256": inventory_digest(inventory),
        "source_file_count": len(inventory),
        "checks": checks,
        "checkpoint_audit": checkpoint_audit,
        "task_statuses": current_statuses,
    }


def update_blocked_state(root: Path, run_id: str, run_dir: Path) -> None:
    state_path = root / "PROJECT_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "state": "FINAL_AUDIT_BLOCKED",
            "current_phase": "FINAL_AUDIT",
            "latest_run_id": run_id,
            "latest_run_directory": run_dir.relative_to(root).as_posix(),
            "next_task": "FINAL_GLOBAL_AUDIT",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--node-executable", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)

    from audit.data_io import write_json
    from audit.final_global_audit import run as final_audit
    from audit.run_manifest import finalize_run, initialize_run, sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest

    config_path = root / "configs" / "final_remaining.yaml"
    amendment_path = root / "configs" / "final_remaining_preregistration_amendment.yaml"
    recovery_path = root / "configs" / "final_audit_recovery.yaml"
    workbook_recovery_path = root / "configs" / "final_audit_workbook_render_recovery.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    recovery = yaml.safe_load(recovery_path.read_text(encoding="utf-8"))
    workbook_recovery = yaml.safe_load(workbook_recovery_path.read_text(encoding="utf-8"))
    config["preregistration_amendment"] = amendment
    config["final_audit_recovery"] = recovery
    config["final_audit_workbook_render_recovery"] = workbook_recovery
    source_run_id = str(recovery["frozen_scientific_run"]["run_id"])
    config["scientific_source_run_dir"] = f"results/{source_run_id}"

    verified = verify_frozen_scientific_run(root, recovery)
    failed = workbook_recovery["frozen_failed_audit_run"]
    failed_dir = root / "results" / str(failed["run_id"])
    failed_inventory = _tree_inventory(failed_dir)
    failure_log = root / str(failed["failure_log"])
    failure_text = failure_log.read_text(encoding="utf-8")
    failed_checks = {
        "manifest_sha256": sha256_file(failed_dir / "manifest.json") == str(failed["manifest_sha256"]),
        "tree_inventory_sha256": inventory_digest(failed_inventory) == str(failed["tree_inventory_sha256"]),
        "file_count": len(failed_inventory) == int(failed["file_count"]),
        "failure_log_sha256": sha256_file(failure_log) == str(failed["failure_log_sha256"]),
        "expected_error_signature": str(failed["expected_error_signature"]) in failure_text,
    }
    if not all(failed_checks.values()):
        raise RuntimeError(f"frozen workbook-render failure verification failed: {failed_checks}")
    verified["frozen_workbook_render_failure"] = {
        "run_id": failed["run_id"],
        "checks": failed_checks,
        "tree_inventory_sha256": inventory_digest(failed_inventory),
        "file_count": len(failed_inventory),
    }
    run_id, run_dir = initialize_run(root)
    write_json(
        run_dir / "certificates" / "verified_final_scientific_reuse.json",
        {
            "run_id": run_id,
            "status": "PASS_CERTIFIED",
            "execution_scope": "FINAL_GLOBAL_AUDIT_ONLY",
            "scientific_tasks_rerun": False,
            "task_statuses_changed": False,
            "verified": verified,
            "config_sha256": sha256_file(config_path),
            "amendment_sha256": sha256_file(amendment_path),
            "recovery_contract_sha256": sha256_file(recovery_path),
            "workbook_render_recovery_contract_sha256": sha256_file(workbook_recovery_path),
        },
    )

    statuses = verified["task_statuses"]
    try:
        outputs = final_audit(config, run_dir, run_id, root, args.node_executable.resolve())
        matrix_path = run_dir / "derived" / "final_global_audit" / "theorem_validation_matrix.parquet"
        (run_dir / "validation_matrix.parquet").write_bytes(matrix_path.read_bytes())
        execution = {
            "run_id": run_id,
            "state": "PROJECT_COMPLETE",
            "execution_scope": "FINAL_GLOBAL_AUDIT_ONLY",
            "scientific_source_run_id": source_run_id,
            "scientific_tasks_rerun": False,
            "task_statuses_changed": False,
            "task_count": len(statuses),
            "terminal_task_count": sum(status in TERMINAL for status in statuses.values()),
            "final_status": str(outputs["status"]),
        }
        (run_dir / "logs" / "final_audit_recovery_execution.json").write_text(
            json.dumps(execution, indent=2, sort_keys=True), encoding="utf-8"
        )
        finalize_run(run_dir, "COMPLETE", statuses)
        print(json.dumps(execution, indent=2, sort_keys=True))
        return 0
    except Exception:
        failure = traceback.format_exc()
        (run_dir / "logs" / "final_audit_recovery_failure.log").write_text(failure, encoding="utf-8")
        write_json(
            run_dir / "certificates" / "final_audit_recovery_failure.json",
            {
                "run_id": run_id,
                "status": "FAIL_IMPLEMENTATION",
                "execution_scope": "FINAL_GLOBAL_AUDIT_ONLY",
                "scientific_tasks_rerun": False,
                "task_statuses_changed": False,
                "traceback_log": (run_dir / "logs" / "final_audit_recovery_failure.log")
                .relative_to(root)
                .as_posix(),
            },
        )
        update_blocked_state(root, run_id, run_dir)
        finalize_run(run_dir, "INCOMPLETE", {"FINAL_GLOBAL_AUDIT": "FAIL_IMPLEMENTATION"})
        print(failure, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
