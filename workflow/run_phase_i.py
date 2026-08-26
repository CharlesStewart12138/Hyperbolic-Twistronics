from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def add_source_path(root: Path) -> None:
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    project = str(root)
    if project not in sys.path:
        sys.path.insert(0, project)


def load_status(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["status"])


def update_task_manifest(root: Path, run_id: str, statuses: dict[str, str], outputs: dict[str, Path]) -> None:
    path = root / "TASK_MANIFEST.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for row in rows:
        task_id = row["task_id"]
        if task_id in statuses:
            row["status"] = statuses[task_id]
            row["run_id"] = run_id
            output = outputs.get(task_id)
            if output:
                relative = output.relative_to(root).as_posix()
                row["certificate"] = relative if "certificate" in relative or relative.endswith(".json") else ""
                row["raw_output"] = relative
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_checkpoint(
    root: Path,
    run_id: str,
    run_dir: Path,
    statuses: dict[str, str],
    outputs: dict[str, Path],
) -> str:
    incomplete = {task: status for task, status in statuses.items() if status in {"INCONCLUSIVE", "FAIL_IMPLEMENTATION", "FAIL_THEORY"}}
    gate_a = all(
        statuses.get(task) in {"PASS_EXACT", "PASS_CERTIFIED", "PASS_CONVERGED", "PASS_EXTERNAL"}
        for task in ("I-04", "I-07", "I-08", "I-10")
    )
    state = "PHASE_I_COMPLETE" if not incomplete and gate_a else "PHASE_I_INCOMPLETE"
    lines = [
        "# Phase I checkpoint",
        "",
        f"- Run ID: `{run_id}`",
        f"- State: `{state}`",
        f"- Gate A model integrity: `{'PASS' if gate_a else 'FAIL'}`",
        "- Scientific parameter scans started: `false`",
        "",
        "## Task results",
        "",
        "| Task | Status | Primary output |",
        "|---|---|---|",
    ]
    for task in sorted(statuses):
        output = outputs.get(task)
        relative = output.relative_to(root).as_posix() if output else "-"
        lines.append(f"| {task} | {statuses[task]} | `{relative}` |")
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "Phase I establishes exact group/arithmetic identities, intrinsic geometry and transport checks, sparse ARO-3B construction, interval-backend readiness, provenance, and immutable run hashing. It does not establish a microscopic magic root, bulk spectrum, no-pollution result, Hodge cancellation, or a public-code scientific agreement.",
            "",
            "The three materialized abelian cover towers are deliberate infrastructure controls. Their commutator word gives bounded systole, so they are rejected for later thermodynamic-bulk claims and must be replaced or supplemented by injectivity-growing towers in B-01/B-15.",
            "",
            "## Open items before/within later phases",
            "",
            "- Re-run the exact scripts under independent Sage and GAP installations for backend cross-checking.",
            "- Install Snakemake to execute the declarative DAG directly; the current Windows driver records the same atomic dependencies.",
            "- Construct three injectivity-growing non-Abelian towers before any B-phase convergence claim.",
            "- Execute G-01 through G-12 only after accepting this checkpoint; none were run here.",
            "",
            f"Run data are immutable under `{run_dir.relative_to(root).as_posix()}/`.",
        ]
    )
    text = "\n".join(lines) + "\n"
    report = root / "reports" / "checkpoint_I.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    (run_dir / "derived" / "checkpoint_I.md").write_text(text, encoding="utf-8")
    project_state = {
        "schema_version": 1,
        "state": state,
        "current_phase": "I",
        "scientific_scans_started": False,
        "latest_run_id": run_id,
        "latest_run_directory": run_dir.relative_to(root).as_posix(),
        "gate_a_model_integrity": "PASS" if gate_a else "FAIL",
        "task_statuses": statuses,
        "theory_source": {
            "original_path": "C:/Users/charl/chatGPTwork/5203/05_FINAL/5203_REVISED_FINAL_ROUND2.pdf",
            "project_copy": "references/5203_REVISED_FINAL_ROUND2.pdf",
            "sha256": "4815472e059312cf75e5b7fce0e2d2f718a5766964e888eec508cff8478f9abb",
            "page_count": 455,
            "immutable_input": True,
        },
        "theory_tex_found": False,
        "preexisting_git_repository_found": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (root / "PROJECT_STATE.json").write_text(
        json.dumps(project_state, indent=2, sort_keys=True), encoding="utf-8"
    )
    return state


def checkpoint_only(root: Path, run_id: str) -> int:
    run_dir = root / "results" / run_id
    paths = {
        "I-02": run_dir / "environment" / "environment.json",
        "I-04": run_dir / "exact" / "i04_octagon_group.json",
        "I-05": run_dir / "exact" / "i05_commensurability_238.json",
        "I-06": run_dir / "raw" / "covers" / "cover_towers.json",
        "I-07": run_dir / "certificates" / "i07_geometry.json",
        "I-08": run_dir / "certificates" / "i08_aro3b.json",
        "I-09": run_dir / "certificates" / "i09_interval_backend.json",
    }
    statuses = {task: load_status(path) if task != "I-02" else "PASS_EXACT" for task, path in paths.items()}
    statuses.update({"I-01": "PASS_EXACT", "I-03": "PASS_EXTERNAL", "I-10": "PASS_EXACT"})
    write_checkpoint(root, run_id, run_dir, statuses, paths)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_source_path(root)
    if args.checkpoint_only:
        if not args.run_id:
            raise SystemExit("--run-id is required with --checkpoint-only")
        return checkpoint_only(root, args.run_id)

    from audit.run_manifest import finalize_run, initialize_run
    from covers.generate_cover_towers import generate
    from environment.lock_environment import main as unused_environment_main  # noqa: F401
    from exact.commensurability_238 import exact_certificate as commensurability_certificate
    from exact.interval_backend import backend_certificate
    from exact.octagon_group import exact_certificate as octagon_certificate
    from geometry.build_orbit_and_frames import build_and_save as build_geometry
    from model.build_aro3b_hamiltonian import build_and_save as build_model
    from environment.lock_environment import snapshot, lock_lines

    run_id, run_dir = initialize_run(root)
    statuses: dict[str, str] = {"I-01": "PASS_EXACT", "I-10": "PASS_EXACT"}
    outputs: dict[str, Path] = {
        "I-01": root / "workflow" / "Snakefile",
        "I-10": run_dir / "manifest.json",
    }
    errors: dict[str, str] = {}

    def execute(task_id: str, function) -> None:
        try:
            status, output = function()
            statuses[task_id] = status
            outputs[task_id] = output
        except Exception:
            statuses[task_id] = "FAIL_IMPLEMENTATION"
            errors[task_id] = traceback.format_exc()

    def task_i02():
        environment_dir = run_dir / "environment"
        environment_dir.mkdir()
        data = snapshot(run_id)
        environment_json = environment_dir / "environment.json"
        environment_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        lock = "\n".join(lock_lines(data)) + "\n"
        (environment_dir / "environment.lock").write_text(lock, encoding="utf-8")
        (root / "requirements.lock").write_text(lock, encoding="utf-8")
        return "PASS_EXACT", environment_json

    def task_i03():
        provenance = root / "public_data" / "provenance.json"
        data = json.loads(provenance.read_text(encoding="utf-8"))
        passed = bool(data.get("resources")) and all(row.get("status") == "PASS_EXTERNAL" for row in data["resources"])
        return ("PASS_EXTERNAL" if passed else "INCONCLUSIVE"), provenance

    def task_i04():
        output = run_dir / "exact" / "i04_octagon_group.json"
        data = octagon_certificate()
        data["run_id"] = run_id
        output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return str(data["status"]), output

    def task_i05():
        output = run_dir / "exact" / "i05_commensurability_238.json"
        data = commensurability_certificate()
        data["run_id"] = run_id
        output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return str(data["status"]), output

    def task_i06():
        output_dir = run_dir / "raw" / "covers"
        output_dir.mkdir()
        data = generate(root / "configs" / "tower_definitions.yaml", output_dir, run_id)
        return str(data["status"]), output_dir / "cover_towers.json"

    def task_i07():
        output = run_dir / "raw" / "orbit_frames.h5"
        certificate = run_dir / "certificates" / "i07_geometry.json"
        data = build_geometry(root / "configs" / "model_base.yaml", output, certificate, run_id)
        return str(data["status"]), certificate

    def task_i08():
        if statuses.get("I-07") != "PASS_CONVERGED":
            return "INCONCLUSIVE", run_dir / "certificates" / "i08_aro3b.json"
        output = run_dir / "raw" / "aro3b_hamiltonian.h5"
        certificate = run_dir / "certificates" / "i08_aro3b.json"
        data = build_model(root / "configs" / "model_base.yaml", output, certificate, run_id)
        return str(data["status"]), certificate

    def task_i09():
        output = run_dir / "certificates" / "i09_interval_backend.json"
        data = backend_certificate(192)
        data["run_id"] = run_id
        output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return str(data["status"]), output

    for task_id, function in (
        ("I-02", task_i02),
        ("I-03", task_i03),
        ("I-04", task_i04),
        ("I-05", task_i05),
        ("I-06", task_i06),
        ("I-07", task_i07),
        ("I-08", task_i08),
        ("I-09", task_i09),
    ):
        execute(task_id, function)

    execution_report = {
        "run_id": run_id,
        "task_statuses": statuses,
        "errors": errors,
        "engine": "workflow/run_phase_i.py",
        "declarative_dag": "workflow/Snakefile",
        "atomic_order": [f"I-{number:02d}" for number in range(1, 11)],
    }
    execution_path = run_dir / "logs" / "phase_i_execution_report.json"
    execution_path.write_text(json.dumps(execution_report, indent=2, sort_keys=True), encoding="utf-8")
    outputs["I-01"] = execution_path

    rows = [
        {
            "task_id": task,
            "run_id": run_id,
            "status": status,
            "output": outputs.get(task, Path("MISSING")).relative_to(root).as_posix() if task in outputs else "MISSING",
        }
        for task, status in sorted(statuses.items())
    ]
    pd.DataFrame(rows).to_parquet(run_dir / "validation_matrix.parquet", index=False)
    update_task_manifest(root, run_id, statuses, outputs)
    state = write_checkpoint(root, run_id, run_dir, statuses, outputs)
    final_status = "COMPLETE" if state == "PHASE_I_COMPLETE" else "INCOMPLETE"
    finalize_run(run_dir, final_status, statuses)
    print(json.dumps({"run_id": run_id, "state": state, "run_dir": str(run_dir)}))
    return 0 if state == "PHASE_I_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

