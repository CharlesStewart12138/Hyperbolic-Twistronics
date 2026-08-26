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


TASKS = ["G-13", "G-14", "G-15", "S-17", "S-18", "S-19", "S-20", "S-21", "S-22", "S-23", "S-24"]
BLOCKERS = {"FAIL_THEORY", "FAIL_IMPLEMENTATION"}
TERMINAL = {"PASS_EXACT", "PASS_CERTIFIED", "PASS_CONVERGED", "PASS_EXTERNAL", "INCONCLUSIVE", "FAIL_EXPECTED"}


def add_path(root: Path) -> None:
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def artifact_digest(path: Path) -> dict[str, object]:
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest

    if path.is_file():
        return {"kind": "file", "sha256": sha256_file(path), "bytes": path.stat().st_size}
    inventory = _tree_inventory(path)
    return {"kind": "directory", "tree_inventory_sha256": inventory_digest(inventory), "file_count": len(inventory)}


def verify_run(root: Path, section: dict) -> dict[str, object]:
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest

    source = root / "results" / str(section["run_id"])
    inventory = _tree_inventory(source)
    checks = {
        "manifest": sha256_file(source / "manifest.json") == str(section["manifest_sha256"]),
        "tree": inventory_digest(inventory) == str(section["tree_inventory_sha256"]),
        "file_count": len(inventory) == int(section["file_count"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen run verification failed for {section['run_id']}: {checks}")
    return {"run_id": section["run_id"], "checks": checks, "tree_inventory_sha256": inventory_digest(inventory)}


def verify_inputs(root: Path, config: dict) -> dict[str, object]:
    from audit.run_manifest import sha256_file

    verified = {
        "frozen_predecessor": verify_run(root, config["frozen_predecessor"]),
        "phase_g_source": verify_run(root, config["phase_g_source"]),
        "phase_s_source": verify_run(root, config["phase_s_source"]),
        "nonabelian_tower_source": verify_run(root, config["nonabelian_tower_source"]),
    }
    file_pairs = []
    for section_name, keys in {
        "frozen_predecessor": ["validation_matrix"],
        "phase_g_source": ["g09_exact", "growth", "g10_certificate", "g11_certificate", "g12_certificate"],
        "phase_s_source": ["s04_certificate", "normal_forms", "s15_table", "s15_certificate", "s14_table"],
        "nonabelian_tower_source": ["certificate", "levels"],
    }.items():
        section = config[section_name]
        for key in keys:
            file_pairs.append((section_name, key, root / str(section[key]), str(section[f"{key}_sha256"])))
    file_checks = {}
    for section_name, key, path, expected in file_pairs:
        name = f"{section_name}.{key}"
        file_checks[name] = path.exists() and sha256_file(path) == expected
    file_checks["initial_task_manifest"] = sha256_file(root / "TASK_MANIFEST.csv") == str(
        config["frozen_predecessor"]["task_manifest_sha256"]
    )
    with (root / "TASK_MANIFEST.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    statuses = {row["task_id"]: row["status"] for row in manifest}
    file_checks["manifest_task_count"] = len(manifest) == 88
    file_checks["remaining_exactly_11"] = sum(status == "NOT_STARTED" for status in statuses.values()) == 11
    file_checks["remaining_set_exact"] = {task for task, status in statuses.items() if status == "NOT_STARTED"} == set(TASKS)
    file_checks["preserved_statuses"] = all(
        statuses.get(task) == expected for task, expected in config["final_audit"]["preserve_statuses"].items()
    )
    if not all(file_checks.values()):
        raise RuntimeError(f"final-stage input verification failed: {json.dumps(file_checks, sort_keys=True)}")
    verified["file_checks"] = file_checks
    return verified


def update_task_manifest(root: Path, run_id: str, task: str, status: str, output: dict[str, Path]) -> None:
    path = root / "TASK_MANIFEST.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    found = False
    for row in rows:
        if row["task_id"] != task:
            continue
        found = True
        row["status"] = status
        row["run_id"] = run_id
        for field, key in (("raw_output", "raw"), ("derived_output", "derived"), ("certificate", "certificate")):
            if key in output:
                row[field] = output[key].relative_to(root).as_posix()
        note = "finite active-fiber scope; no frozen S-07/S-08 bulk promotion" if task.startswith("S-") else "verified hash-only predecessor reuse"
        row["notes"] = (row.get("notes", "") + " " + note).strip()
    if not found:
        raise RuntimeError(f"task not found in manifest: {task}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_interim_matrix(root: Path, run_dir: Path, config: dict) -> None:
    frame = pd.read_parquet(root / str(config["frozen_predecessor"]["validation_matrix"])).copy()
    with (root / "TASK_MANIFEST.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = {row["task_id"]: row for row in csv.DictReader(handle)}
    for index, row in frame.iterrows():
        task = str(row["code_id"])
        source = manifest[task]
        frame.at[index, "status"] = source["status"]
        frame.at[index, "validation_type"] = source["status"]
        frame.at[index, "run_id"] = source["run_id"]
        frame.at[index, "raw_data_file"] = source["raw_output"] or "MISSING"
        frame.at[index, "derived_data_file"] = source["derived_output"] or "MISSING"
        frame.at[index, "certificate_file"] = source["certificate"] or "MISSING"
    frame.to_parquet(run_dir / "validation_matrix.parquet", index=False)


def checkpoint(root: Path, run_dir: Path, run_id: str, task: str, status: str, output: dict[str, Path]) -> Path:
    from audit.data_io import write_json

    path = run_dir / "certificates" / "task_checkpoints" / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "task_id": task,
            "run_id": run_id,
            "status": status,
            "checkpointed_at_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                key: {"path": value.relative_to(root).as_posix(), **artifact_digest(value)}
                for key, value in output.items()
            },
        },
    )
    return path


def update_project_state(root: Path, run_id: str, run_dir: Path, statuses: dict[str, str], next_task: str | None, state_name: str) -> None:
    path = root / "PROJECT_STATE.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state.update(
        {
            "state": state_name,
            "current_phase": "FINAL_REMAINING",
            "latest_run_id": run_id,
            "latest_run_directory": run_dir.relative_to(root).as_posix(),
            "final_remaining_task_statuses": statuses,
            "next_task": next_task,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--node-executable", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)
    from analysis.magic_complexity import run as g14
    from analysis.magic_subsequence_sampling import run as g13
    from audit.data_io import write_json
    from audit.final_global_audit import run as final_audit
    from audit.run_manifest import finalize_run, initialize_run, sha256_file
    from geometry.incommensurate_joint_limit import run as g15
    from spectral.bifurcation_certificates import run as s21
    from spectral.curvature_born_branch import run as s22
    from spectral.geometry_spectrum_factorization import run as s19
    from spectral.magic_landscape import run as s20
    from spectral.master_curve_collapse import run as s18
    from spectral.operational_magic_metrics import run as s17
    from spectral.reverse_falsification import run as s24
    from spectral.symmetry_vs_flatness import run as s23

    config_path = root / "configs" / "final_remaining.yaml"
    amendment_path = root / "configs" / "final_remaining_preregistration_amendment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    config["preregistration_amendment"] = amendment
    verified = verify_inputs(root, config)
    run_id, run_dir = initialize_run(root)
    write_json(
        run_dir / "certificates" / "verified_frozen_input_reuse.json",
        {
            "run_id": run_id,
            "status": "PASS_CERTIFIED",
            "verified": verified,
            "base_preregistration_sha256": sha256_file(config_path),
            "pre_outcome_amendment_sha256": sha256_file(amendment_path),
            "source_modified": False,
        },
    )
    functions = {
        "G-13": g13,
        "G-14": g14,
        "G-15": g15,
        "S-17": s17,
        "S-18": s18,
        "S-19": s19,
        "S-20": s20,
        "S-21": s21,
        "S-22": s22,
        "S-23": s23,
        "S-24": s24,
    }
    statuses: dict[str, str] = {}
    outputs: dict[str, dict[str, Path]] = {}
    errors: dict[str, str] = {}
    blocked = False
    for index, task in enumerate(TASKS):
        try:
            status, output = functions[task](config, run_dir, run_id, root)
        except Exception:
            status = "FAIL_IMPLEMENTATION"
            errors[task] = traceback.format_exc()
            failure_log = run_dir / "logs" / f"{task.lower()}_failure.log"
            failure_log.write_text(errors[task], encoding="utf-8")
            failure_certificate = run_dir / "certificates" / f"{task.lower()}_failure.json"
            write_json(
                failure_certificate,
                {
                    "task_id": task,
                    "run_id": run_id,
                    "status": status,
                    "classification": "FAIL_IMPLEMENTATION",
                    "traceback_log": failure_log.relative_to(root).as_posix(),
                },
            )
            output = {"certificate": failure_certificate}
        statuses[task] = status
        outputs[task] = output
        checkpoint(root, run_dir, run_id, task, status, output)
        update_task_manifest(root, run_id, task, status, output)
        write_interim_matrix(root, run_dir, config)
        next_task = TASKS[index + 1] if index + 1 < len(TASKS) else "FINAL_GLOBAL_AUDIT"
        update_project_state(root, run_id, run_dir, statuses, next_task, "FINAL_REMAINING_IN_PROGRESS")
        if status in BLOCKERS:
            blocked = True
            update_project_state(root, run_id, run_dir, statuses, task, "FINAL_REMAINING_BLOCKED")
            break
        if status not in TERMINAL:
            errors[task] = f"nonterminal status returned: {status}"
            blocked = True
            update_project_state(root, run_id, run_dir, statuses, task, "FINAL_REMAINING_BLOCKED")
            break

    final_outputs = None
    if not blocked and len(statuses) == len(TASKS):
        try:
            final_outputs = final_audit(config, run_dir, run_id, root, args.node_executable.resolve())
        except Exception:
            errors["FINAL_GLOBAL_AUDIT"] = traceback.format_exc()
            (run_dir / "logs" / "final_global_audit_failure.log").write_text(
                errors["FINAL_GLOBAL_AUDIT"], encoding="utf-8"
            )
            blocked = True
            update_project_state(root, run_id, run_dir, statuses, "FINAL_GLOBAL_AUDIT", "FINAL_AUDIT_BLOCKED")
    execution = {
        "run_id": run_id,
        "atomic_order": TASKS,
        "task_statuses": statuses,
        "errors": errors,
        "completed_D_or_NC_rerun": False,
        "final_global_audit_completed": final_outputs is not None,
    }
    (run_dir / "logs" / "final_remaining_execution_report.json").write_text(
        json.dumps(execution, indent=2, sort_keys=True), encoding="utf-8"
    )
    finalize_run(run_dir, "INCOMPLETE" if blocked else "COMPLETE", statuses)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "state": "BLOCKED" if blocked else "PROJECT_COMPLETE",
                "task_statuses": statuses,
                "errors": errors,
                "final_status": None if final_outputs is None else str(final_outputs["status"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
