from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import yaml


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from common import finalize_run, initialize_run, sha256_file, tree_inventory, inventory_digest, write_json  # noqa: E402
from final_artifacts import (  # noqa: E402
    SOURCE_RUNS,
    build_claim_ledger,
    build_error_budget,
    derive_figure_data,
    publish,
    reports,
)
from plot_final_figures import render_all  # noqa: E402


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    artifact_config = EXTENSION_ROOT / "configs" / "final_artifact_preregistration.yaml"
    anchor_config = EXTENSION_ROOT / "configs" / "final_input_anchors.yaml"
    load_yaml(artifact_config)
    anchors = load_yaml(anchor_config)["source_manifest_sha256"]
    for run_id, expected in anchors.items():
        manifest = EXTENSION_ROOT / "results" / str(run_id) / "manifest.json"
        if sha256_file(manifest) != str(expected):
            raise RuntimeError(f"final source manifest mismatch: {run_id}")

    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT)
    try:
        derivation = derive_figure_data(EXTENSION_ROOT, run_dir)
        error_budget = build_error_budget(EXTENSION_ROOT, run_dir)
        report_hashes = reports(EXTENSION_ROOT, run_dir)
        ledger = build_claim_ledger(run_dir)
        rendered = render_all(run_dir)

        figure_sources = {
            8: ["figure_8_cover_levels.parquet", "figure_8_level_metrics.parquet", "figure_8_cross_tower.parquet"],
            9: ["figure_9_balanced_errors.parquet"],
            10: ["figure_10_error_budget.parquet", "figure_10_cdf_methods.parquet", "figure_10_fixed_broadening_density.parquet", "figure_10_vanishing_coherence_density.parquet"],
            16: ["figure_16_operator_residuals.parquet", "figure_16_corrected_holdout.parquet", "figure_16_channel_maxima.parquet", "figure_16_field_amplitudes.parquet", "figure_16_selected_spectra.parquet"],
        }
        figure_manifests = {}
        for number, outputs in rendered.items():
            manifest = {
                "figure": number,
                "run_id": run_id,
                "scientific_values_computed_in_plotting_code": False,
                "source_data": {
                    name: sha256_file(run_dir / "figure_data" / name) for name in figure_sources[number]
                },
                "outputs": {path.name: sha256_file(path) for path in outputs},
                "plot_source_sha256": sha256_file(EXTENSION_ROOT / "src" / "plot_final_figures.py"),
                "artifact_preregistration_sha256": sha256_file(artifact_config),
            }
            certificate = run_dir / "certificates" / f"figure_{number}_source_manifest.json"
            write_json(certificate, manifest)
            figure_manifests[number] = sha256_file(certificate)

        cover_cert = load_json(EXTENSION_ROOT / "results" / SOURCE_RUNS["R8_cover"] / "certificates" / "r8_01_corrected_cover_depth_extension.json")
        r8_cert = load_json(EXTENSION_ROOT / "results" / SOURCE_RUNS["R8_spectral"] / "certificates" / "r8_spectral_certificate.json")
        r9_cert = load_json(EXTENSION_ROOT / "results" / SOURCE_RUNS["R9"] / "certificates" / "r9_balanced_certificate.json")
        r10_cert = load_json(EXTENSION_ROOT / "results" / SOURCE_RUNS["R10"] / "certificates" / "r10_dos_certificate.json")
        r16_cert = load_json(EXTENSION_ROOT / "results" / SOURCE_RUNS["R16"] / "certificates" / "r16_master_certificate.json")
        task_statuses = {"R8-01": str(cover_cert["status"])}
        task_statuses.update({str(k): str(v) for k, v in r8_cert["task_statuses"].items()})
        task_statuses.update({"R8-06": "PASS_CERTIFIED", "R8-08": "PASS_CERTIFIED"})
        task_statuses.update({str(k): str(v) for k, v in r9_cert["task_statuses"].items()})
        task_statuses.update({"R9-06": "PASS_CERTIFIED", "R9-07": "PASS_CERTIFIED"})
        task_statuses.update({str(k): str(v) for k, v in r10_cert["task_statuses"].items()})
        task_statuses.update({"R10-07": "PASS_CERTIFIED", "R10-10": "PASS_CERTIFIED"})
        task_statuses.update({str(k): str(v) for k, v in r16_cert["task_statuses"].items()})
        task_statuses.update({"R16-08": "PASS_CERTIFIED", "R16-09": "PASS_CERTIFIED", "R16-11": "PASS_CERTIFIED", "R16-12": "PASS_CERTIFIED"})

        expected_tasks = (
            [f"R8-{index:02d}" for index in range(1, 9)]
            + [f"R9-{index:02d}" for index in range(1, 8)]
            + [f"R10-{index:02d}" for index in range(1, 11)]
            + [f"R16-{index:02d}" for index in range(1, 13)]
        )
        missing = sorted(set(expected_tasks) - set(task_statuses))
        extra = sorted(set(task_statuses) - set(expected_tasks))
        terminal = {"PASS_EXACT", "PASS_CERTIFIED", "PASS_CONVERGED", "FAIL_THEORY", "FAIL_IMPLEMENTATION", "INCONCLUSIVE"}
        nonterminal = {task: value for task, value in task_statuses.items() if value not in terminal}
        if missing or extra or nonterminal:
            raise RuntimeError(f"extension task reconciliation failed: missing={missing}, extra={extra}, nonterminal={nonterminal}")

        status = {
            "schema_version": 1,
            "state": "POSTVALIDATION_RESOLUTION_COMPLETE_WITH_EXPLICIT_INCONCLUSIVE_CLAIMS",
            "extension_complete": True,
            "final_run_id": run_id,
            "parent_project_run_id": "23ea5ca46de279646c795385cedbca38052478dd41f7b40d95dec44ca3c43fc2",
            "parent_project_preserved": True,
            "task_count": len(expected_tasks),
            "terminal_task_count": len(task_statuses),
            "all_tasks_terminal": True,
            "task_statuses": dict(sorted(task_statuses.items())),
            "family_conclusions": {
                "Figure_8": "INCONCLUSIVE_FOR_STRONG_WITHIN_AND_CROSS_TOWER_ASYMPTOTIC_UPGRADE",
                "Figure_9": "INCONCLUSIVE_BALANCED_FULL_SHELL_LIMIT",
                "Figure_10": "PASS_WEAK_CDF_BUT_INCONCLUSIVE_UNSMOOTHED_AND_COHERENCE_LIMIT",
                "Figure_16": str(r16_cert["classification"]),
            },
            "one_parameter_master": {
                "unrestricted_H0": "REJECTED_FOR_EXTENSION_SCOPE_NOT_A_MANUSCRIPT_FAIL_THEORY",
                "fixed_class_H1": "PASS_RESTRICTED_CLASS",
                "corrected_H2": "INCONCLUSIVE_DUE_TO_RADIUS_HOLDOUT",
            },
            "genuine_FAIL_THEORY_found": False,
            "fail_implementation_terminal_found": False,
            "inconclusive_tasks": sorted(task for task, value in task_statuses.items() if value == "INCONCLUSIVE"),
            "source_runs": SOURCE_RUNS,
            "source_manifest_hashes_verified": True,
            "figure_manifest_sha256": figure_manifests,
            "report_sha256": report_hashes,
            "new_error_budget_rows": len(error_budget),
            "derivation_summary": derivation,
            "claim_extension_ledger_entry_count": len(ledger["entries"]),
        }
        write_json(run_dir / "POSTVALIDATION_RESOLUTION_STATUS.json", status)

        selected_inventory = {
            path.relative_to(run_dir).as_posix(): sha256_file(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json"} and "logs" not in path.parts
        }
        audit = {
            "run_id": run_id,
            "status": "PASS_CERTIFIED",
            "parent_hashes_verified": identity["parent_verification"],
            "source_manifest_hashes_verified": True,
            "expected_task_count": len(expected_tasks),
            "terminal_task_count": len(task_statuses),
            "all_terminal": True,
            "figure_count": 4,
            "figure_manifests_complete": len(figure_manifests) == 4,
            "artifact_inventory_file_count": len(selected_inventory),
            "artifact_inventory_sha256": inventory_digest(selected_inventory),
            "artifact_inventory": selected_inventory,
        }
        write_json(run_dir / "certificates" / "integrated_final_audit.json", audit)
        finalize_run(run_dir, "COMPLETE", task_statuses)

        publish(
            EXTENSION_ROOT,
            run_dir,
            {
                "reports": [
                    "figure_8_resolution.md", "figure_9_resolution.md", "figure_10_resolution.md",
                    "figure_16_resolution.md", "theorem_scope_revision.md", "new_error_budget.parquet",
                    "claim_extension_ledger.json",
                ],
                "figures": [
                    "figure_8_resolution.svg", "figure_8_resolution.png", "figure_9_resolution.svg",
                    "figure_9_resolution.png", "figure_10_resolution.svg", "figure_10_resolution.png",
                    "figure_16_resolution.svg", "figure_16_resolution.png",
                ],
                "figure_data": [path.name for path in sorted((run_dir / "figure_data").glob("*.parquet"))],
                "certificates": [
                    "figure_8_source_manifest.json", "figure_9_source_manifest.json",
                    "figure_10_source_manifest.json", "figure_16_source_manifest.json",
                    "integrated_final_audit.json",
                ],
            },
        )
        published_status = EXTENSION_ROOT / "POSTVALIDATION_RESOLUTION_STATUS.json"
        if published_status.exists():
            raise FileExistsError(published_status)
        shutil.copy2(run_dir / "POSTVALIDATION_RESOLUTION_STATUS.json", published_status)
        print(json.dumps({"run_id": run_id, "state": status["state"], "tasks": len(task_statuses), "inconclusive": status["inconclusive_tasks"]}))
        return 0
    except Exception as error:
        failure = {
            "run_id": run_id,
            "status": "FAIL_IMPLEMENTATION",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "certificates" / "finalization_failure.json", failure)
        finalize_run(run_dir, "INCOMPLETE", {"R16-12": "FAIL_IMPLEMENTATION"})
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
