from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from audit.data_io import write_json, write_zarr
from spectral.magic_active_shell import ActiveShellModel, load_baseline_variation, relative_change


METRIC_KEYS = ("bandwidth_W", "gap_Delta", "Omega_max", "rho_coh_max", "C_coh")


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    active = config["active_shell"]
    phase_s = yaml.safe_load((root / "configs" / "phase_s.yaml").read_text(encoding="utf-8"))
    variation, root_w = load_baseline_variation(root, config)
    natural = phase_s["natural_surface_model"]
    model = ActiveShellModel(
        root,
        phase_s,
        root / str(config["phase_s_source"]["normal_forms"]),
        curvature_radius=float(natural["curvature_radius"]),
        lambda_perp=float(natural["lambda_perp"]),
        cutoff=int(natural["normal_form_cutoff"]),
        variation=variation,
    )
    primary = model.path_spectrum(
        root_w,
        float(active["q_min"]),
        float(active["q_max"]),
        int(active["q_points_primary"]),
    )
    refined = model.path_spectrum(
        root_w,
        float(active["q_min"]),
        float(active["q_max"]),
        int(active["q_points_refined"]),
    )
    eta = float(active["lorentzian_eta"])
    primary_metrics = model.metrics(primary, eta)
    refined_metrics = model.metrics(refined, eta)
    convergence = {key: relative_change(primary_metrics[key], refined_metrics[key]) for key in METRIC_KEYS}
    raw = run_dir / "raw" / "operational_magic.zarr"
    write_zarr(
        raw,
        {
            "q": refined.q,
            "complete_eigenvalues": refined.eigenvalues,
            "target_energy": refined.target_energy,
            "target_index": refined.target_index,
            "target_coherence": refined.target_coherence,
            "consecutive_overlap": refined.consecutive_overlap,
            "external_gap": refined.external_gap,
        },
        {
            "task_id": "S-17",
            "run_id": run_id,
            "target_rule": active["target_tracking_rule"],
            "same_target_for_all_metrics": True,
            "model_scope": active["model_family"],
            "bulk_claim_permitted": False,
        },
    )
    derived = run_dir / "derived" / "operational_magic_metrics.parquet"
    pd.DataFrame(
        [
            {
                "root_w_over_t": root_w,
                "q_points_primary": int(active["q_points_primary"]),
                "q_points_refined": int(active["q_points_refined"]),
                **{f"primary_{key}": primary_metrics[key] for key in primary_metrics},
                **{f"refined_{key}": refined_metrics[key] for key in refined_metrics},
                **{f"relative_change_{key}": convergence[key] for key in convergence},
            }
        ]
    ).to_parquet(derived, index=False)
    passed = (
        refined_metrics["minimum_tracking_overlap"] >= float(active["minimum_tracking_overlap"])
        and refined_metrics["gap_Delta"] >= float(active["minimum_gap"])
        and refined_metrics["C_coh"] >= float(active["minimum_coherence"])
        and max(convergence.values()) <= float(active["maximum_refinement_relative_change"])
    )
    status = "PASS_CONVERGED" if passed else "INCONCLUSIVE"
    certificate = run_dir / "certificates" / "s17_operational_magic_metrics.json"
    write_json(
        certificate,
        {
            "task_id": "S-17",
            "run_id": run_id,
            "status": status,
            "preregistered_target": active["variation"],
            "root_w_over_t": root_w,
            "same_target_for_all_diagnostics": True,
            "metrics": refined_metrics,
            "grid_refinement_relative_changes": convergence,
            "maximum_refinement_relative_change": max(convergence.values()),
            "normal_form_count": model.normal_form_count,
            "scope": "fixed transported finite-rank ARO-3B character-sector active fiber; no infinite-regular or bulk promotion",
            "reason_if_inconclusive": "At least one preregistered band-tracking, gap, coherence, or refinement criterion did not close.",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
