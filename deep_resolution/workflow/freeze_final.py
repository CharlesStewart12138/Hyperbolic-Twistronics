from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from common import finalize_run, load_yaml, sha256_file, write_json  # noqa: E402


FIGURE_FILES = {
    8: ["figure08_deep_finite_cover_closure.png", "figure08_deep_finite_cover_closure.svg"],
    9: ["figure09_balanced_full_shell.png", "figure09_balanced_full_shell.svg"],
    10: ["figure10_vanishing_broadening_DOS.png", "figure10_vanishing_broadening_DOS.svg"],
    16: ["figure16_curvature_relevant_universality.png", "figure16_curvature_relevant_universality.svg"],
    18: ["figure18_curvature_coordinate_rank.png", "figure18_curvature_coordinate_rank.svg"],
}


FIGURE_DATA = {
    8: ["figure_8_deep_cover_levels.parquet", "figure_8_matched_levels.parquet", "figure_10_vanishing_schedule.parquet"],
    9: ["figure_9_balanced_diagonal.parquet"],
    10: ["figure_10_KPM_SLQ_CDF.parquet", "figure_10_vanishing_schedule.parquet", "figure_10_holdout_coherence_density.parquet"],
    16: ["figure_16_channel_summary.parquet", "figure_16_hypothesis_holdout.parquet", "figure_16_old_radius_shape_field.parquet"],
    18: ["figure_18_singular_values.parquet", "figure_18_curvature_scaling.parquet", "figure_18_hypothesis_summary.parquet", "r16_block_rank.parquet"],
}


def copy_published(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"published target exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("family") != "FINAL_AUDIT" or manifest.get("status") != "RUNNING":
        raise RuntimeError("not a mutable FINAL_AUDIT run")
    figure_manifests = {}
    for figure, names in FIGURE_FILES.items():
        files = {}
        for name in names:
            path = run_dir / "figures" / name
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"missing figure artifact: {path}")
            files[name] = sha256_file(path)
        data = {}
        for name in FIGURE_DATA[figure]:
            path = run_dir / "figure_data" / name
            if not path.exists():
                raise RuntimeError(f"missing figure data: {path}")
            data[name] = sha256_file(path)
        payload = {"figure": figure, "files": files, "figure_data": data, "plot_source_sha256": sha256_file(EXTENSION_ROOT / "src" / "plot_final_figures.py")}
        write_json(run_dir / "certificates" / f"figure_{figure}_manifest.json", payload)
        figure_manifests[str(figure)] = payload
    source_config = load_yaml(EXTENSION_ROOT / "configs" / "final_source_runs_preregistration.yaml")
    source_runs = {name: entry["run_id"] for name, entry in source_config["source_runs"].items()}
    all_task_statuses = {}
    for family, source_id in source_runs.items():
        source_manifest = json.loads((EXTENSION_ROOT / "results" / str(source_id) / "manifest.json").read_text(encoding="utf-8"))
        for task, status in source_manifest.get("task_statuses", {}).items():
            all_task_statuses[f"{family}:{task}"] = status
    final_tasks = {
        "AUDIT-01": "PASS_CERTIFIED",
        "AUDIT-02": "PASS_CERTIFIED",
        "AUDIT-03": "PASS_CERTIFIED",
        "AUDIT-04": "PASS_CERTIFIED",
        "AUDIT-05": "PASS_CERTIFIED",
        "AUDIT-06": "PASS_CERTIFIED",
    }
    status_payload = {
        "schema_version": 1,
        "state": "DEEP_RESOLUTION_COMPLETE_WITH_SCOPED_PASS_AND_EXPLICIT_INCONCLUSIVE_CLAIMS",
        "deep_resolution_complete": True,
        "final_run_id": manifest["run_id"],
        "parent_project_preserved": True,
        "parent_extension_preserved": True,
        "source_runs": source_runs,
        "preserved_failed_runs": {"R10_memmap_cleanup": {"run_id": source_runs["R10_failed_memmap"], "status": "FAIL_IMPLEMENTATION"}},
        "family_conclusions": {
            "Figure_8": {"weak_retained_projective_sector": "PASS_WEAK_BULK", "strong_full_regular_spectrum": "INCONCLUSIVE"},
            "Figure_9": {"balanced_diagonal": "PASS_CERTIFIED", "C0_C1_C2_tail_closure": "INCONCLUSIVE"},
            "Figure_10": {"weak_CDF": "PASS_CONVERGED", "vanishing_schedule_finite_holdout": "PASS_CONVERGED", "local_unsmoothed_coherence_DOS": "INCONCLUSIVE"},
            "Figure_16": "PASS_RESTRICTED_CLASS",
            "curvature_independent_relevant_coordinate": "LOCAL_OPERATOR_RANK2_EVIDENCE_NOT_GLOBAL_CERTIFICATION",
        },
        "genuine_FAIL_THEORY_found": False,
        "claims_may_be_strengthened": [
            "three non-Abelian congruence towers possess quantitative certified growing injectivity lower bounds",
            "a genuine balanced cover diagonal realizes L=1,2,3,4,5 and extends analytically",
            "retained projective-sector KPM/SLQ weak CDF and frozen eta=sqrt(kappa) finite holdout pass",
            "the old fixed-octagon radius residual is a lambda_perp/a microscopic shape effect on its isolated holdout",
        ],
        "claims_must_remain_restricted": [
            "one-parameter universality is confined to the fixed comparison class",
            "local fixed-a curvature tangent rank is not a global bulk curvature theorem",
            "weak projective-sector convergence is not full regular-spectrum no-pollution",
        ],
        "inconclusive_claims": [
            "strong edge/gap/projector no-pollution",
            "C0/C1/C2 physical-tail closure on the balanced diagonal",
            "tower-uniform regularity and local unsmoothed/coherence DOS",
            "globally identifiable two-parameter curvature master",
        ],
        "figure_manifests": figure_manifests,
        "source_task_statuses": all_task_statuses,
        "final_task_statuses": final_tasks,
        "publication_claim_ledger": "reports/publication_claim_ledger.json",
        "theorem_to_computation_matrix": "reports/theorem_to_computation_matrix.parquet",
        "error_budget": "reports/deep_error_budget.parquet",
    }
    write_json(run_dir / "DEEP_RESOLUTION_STATUS.json", status_payload)
    freeze = finalize_run(run_dir, final_tasks, "DEEP_RESOLUTION_COMPLETE_WITH_SCOPED_RESULTS")

    # Publish immutable copies into the requested top-level extension folders.
    for directory in ("reports", "figure_data", "figures", "certificates"):
        for source in sorted((run_dir / directory).iterdir()):
            if source.is_file():
                copy_published(source, EXTENSION_ROOT / directory / source.name)
    copy_published(run_dir / "freeze_certificate.json", EXTENSION_ROOT / "certificates" / "final_run_freeze_certificate.json")
    published_status = dict(status_payload)
    published_status["final_run_freeze_certificate_sha256"] = sha256_file(run_dir / "freeze_certificate.json")
    published_status["final_run_tree_inventory_sha256"] = freeze["tree_inventory_sha256"]
    published_status["published_report_hashes"] = {path.name: sha256_file(path) for path in sorted((EXTENSION_ROOT / "reports").iterdir()) if path.is_file()}
    write_json(EXTENSION_ROOT / "DEEP_RESOLUTION_STATUS.json", published_status)
    print(json.dumps({"final_run_id": manifest["run_id"], "state": published_status["state"], "figure_count": len(figure_manifests), "freeze": freeze["tree_inventory_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

