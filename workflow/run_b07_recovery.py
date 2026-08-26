from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def add_path(root: Path) -> None:
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(
    root: Path,
    run_dir: Path,
    run_id: str,
    statuses: dict[str, str],
    outputs: dict[str, dict[str, Path]],
) -> str:
    b07_passed = statuses.get("B-07") == "PASS_CERTIFIED"
    state = "PHASE_B_B07_COMPLETE" if b07_passed else "PHASE_B_BLOCKED"
    lines = [
        "# Phase B B-07 recovery checkpoint",
        "",
        f"- Run ID: `{run_id}`",
        f"- State: `{state}`",
        "- Execution scope: `B-07_ONLY`",
        "- B-02--B-15 executed: `false`",
        "- Phase I/G/S rerun: `false`",
        "- B-01 recomputed: `false`",
        "- Both predecessor Phase-B runs modified: `false`",
        "",
        "| Task | Status | Certificate |",
        "|---|---|---|",
    ]
    for task in ("B-01", "B-07"):
        certificate = outputs.get(task, {}).get("certificate")
        relative = certificate.relative_to(root).as_posix() if certificate else "-"
        lines.append(f"| {task} | {statuses.get(task, 'NOT_STARTED')} | `{relative}` |")
    lines.extend(["", "B-02--B-15 remain `NOT_STARTED` in this recovery run.", ""])
    text = "\n".join(lines)
    (run_dir / "derived" / "checkpoint_B07_recovery.md").write_text(text, encoding="utf-8")
    (root / "reports" / "checkpoint_B.md").write_text(text, encoding="utf-8")
    project_path = root / "PROJECT_STATE.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project.update(
        {
            "state": state,
            "current_phase": "B",
            "latest_run_id": run_id,
            "latest_run_directory": run_dir.relative_to(root).as_posix(),
            "phase_b_task_statuses": statuses,
            "next_task": "B-02" if b07_passed else "B-07",
            "preserved_phase_b_failed_run_ids": [
                "0b4258f5846301f260d77c64f2f0dfe91223d082a5342044235db030dbb12203",
                "190214972dbfac41046435b98a5d715c1410ba894fc2de55ff5143b6def20ecd",
            ],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_json(project_path, project)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)

    from audit.data_io import write_json
    from audit.run_manifest import finalize_run, initialize_run
    from representation.b07_recovery_seed import (
        audit_frozen_runs,
        audit_materialization,
        seed_recovery_artifacts,
        verify_b01_hashes,
    )
    from representation.wedderburn_b07_recovery import (
        B07StageFailure,
        prepare_wedderburn,
        run as adjudicate_b07,
    )
    from workflow.run_phase_b_resume import preserve_b01, update_manifest, validation_rows

    config = yaml.safe_load((root / "configs" / "phase_b_b07_recovery.yaml").read_text(encoding="utf-8"))
    if config.get("execution_scope") != "B-07_ONLY":
        raise SystemExit("recovery configuration is not restricted to B-07")
    gate = json.loads((root / str(config["tower_gate"]["certificate"])).read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_CERTIFIED":
        raise SystemExit("non-Abelian tower gate is not certified")
    frozen_before = audit_frozen_runs(root, config)
    verify_b01_hashes(root, config)
    run_id, run_dir = initialize_run(root)
    statuses: dict[str, str] = {}
    outputs: dict[str, dict[str, Path]] = {}
    errors: dict[str, str] = {}
    try:
        status, output = preserve_b01(root, run_dir, run_id, config)
        statuses["B-01"] = status
        outputs["B-01"] = output
        seed_certificate = seed_recovery_artifacts(root, run_dir, run_id, config, frozen_before)
        materialization_certificate = audit_materialization(run_dir, config)
        blocks, diagnostics, precompute_outputs = prepare_wedderburn(root, run_dir, run_id, config)
        context = {
            "blocks": blocks,
            "wedderburn_diagnostics": diagnostics,
            "wedderburn_outputs": precompute_outputs,
        }
        status, output = adjudicate_b07(config, run_dir, run_id, root, context)
        statuses["B-07"] = status
        outputs["B-07"] = output
        b07_certificate = json.loads(output["certificate"].read_text(encoding="utf-8"))
        b07_certificate.update(
            {
                "recovery_seed_certificate": seed_certificate.relative_to(run_dir).as_posix(),
                "matrix_materialization_certificate": materialization_certificate.relative_to(run_dir).as_posix(),
                "execution_scope": "B-07_ONLY",
                "b02_through_b15_executed": False,
            }
        )
        write_json(output["certificate"], b07_certificate)
    except B07StageFailure as error:
        statuses.setdefault("B-01", "PASS_CONVERGED")
        statuses["B-07"] = str(error.payload.get("status", "FAIL_IMPLEMENTATION"))
        outputs["B-07"] = {
            "raw": run_dir / "raw" / "representation",
            "certificate": run_dir / "certificates" / "b07_failure.json",
        }
        errors["B-07"] = traceback.format_exc()
    except Exception:
        statuses.setdefault("B-01", "PASS_CONVERGED")
        statuses["B-07"] = "FAIL_IMPLEMENTATION"
        errors["B-07"] = traceback.format_exc()
        failure = run_dir / "certificates" / "b07_failure.json"
        write_json(
            failure,
            {
                "task_id": "B-07",
                "run_id": run_id,
                "status": "FAIL_IMPLEMENTATION",
                "error": errors["B-07"],
            },
        )
        outputs["B-07"] = {"raw": run_dir / "raw" / "representation", "certificate": failure}

    frozen_after = audit_frozen_runs(root, config)
    if frozen_after != frozen_before:
        raise RuntimeError("one or both frozen Phase-B predecessor runs changed during recovery")
    integrity_certificate = run_dir / "certificates" / "frozen_phase_b_predecessor_integrity.json"
    write_json(
        integrity_certificate,
        {
            "status": "PASS_EXACT",
            "before_equals_after": True,
            "predecessors": frozen_after,
        },
    )
    report = {
        "run_id": run_id,
        "execution_scope": "B-07_ONLY",
        "atomic_order": ["B-01_HASH_REUSE", "B-07_RECOVERY_FROM_IRREP_0004"],
        "task_statuses": statuses,
        "errors": errors,
        "b02_through_b15_executed": False,
        "frozen_predecessor_integrity_certificate": integrity_certificate.relative_to(root).as_posix(),
    }
    atomic_json(run_dir / "logs" / "b07_recovery_execution_report.json", report)
    rows = validation_rows(root, run_id, statuses, outputs)
    for row in rows:
        row["parameter_set"] = "configs/phase_b_b07_recovery.yaml"
        if row["code_id"] not in {"B-01", "B-07"}:
            row["status"] = "NOT_STARTED"
            row["validation_type"] = "NOT_STARTED"
            row["notes"] = "Excluded by the B-07-only recovery scope."
    pd.DataFrame(rows).to_parquet(run_dir / "validation_matrix.parquet", index=False)
    update_manifest(root, run_id, statuses, outputs)
    state = save_checkpoint(root, run_dir, run_id, statuses, outputs)
    finalize_run(run_dir, "COMPLETE" if state == "PHASE_B_B07_COMPLETE" else "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 0 if state == "PHASE_B_B07_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
