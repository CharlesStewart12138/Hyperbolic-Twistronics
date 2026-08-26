from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = EXTENSION_ROOT.parent
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_ROOT / "src"))

from common import finalize_run, initialize_run, load_yaml, sha256_file, write_json  # noqa: E402
from projective_action import build_projective_action, symmetry_residual  # noqa: E402
from r10_dos import (  # noqa: E402
    cdf_distance,
    gaussian_slq_density_cdf,
    kpm_density_cdf,
    kpm_local_moments,
    lorentzian_density,
    regular_interval_audit,
    slq_local,
)
from tower_height import marked_generators  # noqa: E402


TOWERS = {
    "congruence_p7_r2": (7, 2),
    "congruence_p23_r11": (23, 11),
    "congruence_p31_r3": (31, 3),
}


def main() -> int:
    config_path = EXTENSION_ROOT / "configs" / "r10_preregistration.yaml"
    resource_path = EXTENSION_ROOT / "configs" / "r10_resource_amendment_preregistration.yaml"
    config = load_yaml(config_path)
    resource = load_yaml(resource_path)
    order = int(resource["KPM_polynomial_order"])
    depth = int(resource["SLQ_lanczos_depth"])
    sigma = float(resource["SLQ_smoothing"]["sigma"])
    grid_points = int(config["KPM"]["energy_grid_points"])
    grid = np.linspace(-0.9995, 0.9995, grid_points)
    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT, "R10_DOS")
    progress = run_dir / "logs" / "r10_progress.jsonl"
    outputs: dict[tuple[str, int], dict[str, object]] = {}
    audit_rows = []
    method_rows = []
    level_list: list[tuple[str, int]] = []
    for tower_id, entry in config["sector"]["towers"].items():
        for level in entry["levels"]:
            level_list.append((str(tower_id), int(level)))
    for case_index, (tower_id, level) in enumerate(level_list):
        p, root = TOWERS[tower_id]
        started = time.perf_counter()
        action = build_projective_action(p, root, level)
        action_seconds = time.perf_counter() - started
        symmetric = symmetry_residual(action)
        generators = marked_generators(p, root, level)
        exact = {
            "tower_id": tower_id,
            "p": p,
            "root_mod_p": root,
            "level": level,
            "modulus": action.modulus,
            "dimension": action.dimension,
            "marked_generators": [list(matrix) for matrix in generators],
            "bijection_pass": action.bijection_pass,
            "inverse_pair_pass": action.inverse_pair_pass,
            "symmetry_residual": symmetric,
        }
        write_json(run_dir / "exact" / f"r10_action_{tower_id}_level_{level}.json", exact)
        kpm_started = time.perf_counter()
        moments, kpm_audit = kpm_local_moments(action, order)
        kpm_density, kpm_cdf = kpm_density_cdf(moments, grid)
        kpm_seconds = time.perf_counter() - kpm_started
        slq_started = time.perf_counter()
        slq = slq_local(
            action,
            depth,
            temp_parent=run_dir / "logs",
            breakdown_tolerance=float(config["SLQ"]["breakdown_tolerance"]),
        )
        slq_density, slq_cdf = gaussian_slq_density_cdf(slq["nodes"], slq["weights"], grid, sigma)
        slq_seconds = time.perf_counter() - slq_started
        disagreement = cdf_distance(kpm_cdf, slq_cdf)
        raw_path = run_dir / "raw" / f"r10_{tower_id}_level_{level}.npz"
        with raw_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                grid=grid,
                kpm_moments=moments,
                kpm_density=kpm_density,
                kpm_cdf=kpm_cdf,
                slq_alpha=slq["alpha"],
                slq_beta=slq["beta"],
                slq_nodes=slq["nodes"],
                slq_weights=slq["weights"],
                slq_density=slq_density,
                slq_cdf=slq_cdf,
            )
        outputs[(tower_id, level)] = {
            "p": p,
            "root": root,
            "dimension": action.dimension,
            "moments": moments,
            "kpm_density": kpm_density,
            "kpm_cdf": kpm_cdf,
            "slq_density": slq_density,
            "slq_cdf": slq_cdf,
            "slq_nodes": slq["nodes"],
            "slq_weights": slq["weights"],
            "method_average_cdf": 0.5 * (kpm_cdf + slq_cdf),
            "method_disagreement": disagreement,
        }
        audit_rows.append(
            {
                "tower_id": tower_id,
                "p": p,
                "level": level,
                "modulus": action.modulus,
                "dimension": action.dimension,
                "action_seconds": action_seconds,
                "KPM_seconds": kpm_seconds,
                "SLQ_seconds": slq_seconds,
                "KPM_order": order,
                "SLQ_requested_depth": depth,
                "SLQ_actual_depth": int(slq["actual_depth"]),
                "KPM_SLQ_CDF_disagreement": disagreement,
                "symmetry_residual": symmetric,
                "inverse_pair_pass": action.inverse_pair_pass,
                "KPM_mu0_error": float(kpm_audit["mu0_error"]),
                "SLQ_weight_sum_error": float(slq["weight_sum_error"]),
                "SLQ_orthogonality_residual": float(slq["maximum_orthogonality_residual"]),
                "raw_sha256": sha256_file(raw_path),
            }
        )
        for index, energy in enumerate(grid):
            method_rows.append(
                {
                    "tower_id": tower_id,
                    "level": level,
                    "energy": float(energy),
                    "KPM_density": float(kpm_density[index]),
                    "KPM_CDF": float(kpm_cdf[index]),
                    "SLQ_density": float(slq_density[index]),
                    "SLQ_CDF": float(slq_cdf[index]),
                }
            )
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"case_index": case_index, "tower_id": tower_id, "level": level, "dimension": action.dimension, "action_seconds": action_seconds, "KPM_seconds": kpm_seconds, "SLQ_seconds": slq_seconds}) + "\n")
            handle.flush()
        del action

    deepest = {tower_id: max(int(level) for level in config["sector"]["towers"][tower_id]["levels"]) for tower_id in TOWERS}
    deepest_cdfs = [outputs[(tower_id, level)]["method_average_cdf"] for tower_id, level in deepest.items()]
    cross_reference = np.mean(np.asarray(deepest_cdfs), axis=0)
    derived_rows = []
    regularity = {}
    density_rows = []
    for tower_id, level in level_list:
        output = outputs[(tower_id, level)]
        kappa = cdf_distance(output["method_average_cdf"], cross_reference)
        eta = max(float(config["broadening"]["floor_for_zero_numerical_kappa"]), math.sqrt(max(kappa, 0.0)))
        coherence_density = lorentzian_density(output["slq_nodes"], output["slq_weights"], grid, eta)
        is_holdout = level == deepest[tower_id]
        finite_cover_error = cdf_distance(output["method_average_cdf"], outputs[(tower_id, deepest[tower_id])]["method_average_cdf"])
        reference_error = max(cdf_distance(cdf, cross_reference) for cdf in deepest_cdfs)
        derived_rows.append(
            {
                "tower_id": tower_id,
                "p": int(output["p"]),
                "level": level,
                "dimension": int(output["dimension"]),
                "split": "holdout" if is_holdout else "pilot",
                "kappa_N": kappa,
                "eta_N": eta,
                "kappa_over_eta": kappa / eta,
                "sampling_error": 0.0,
                "KPM_truncation_resolution": math.pi / order,
                "SLQ_quadrature_resolution": 2.0 / depth,
                "finite_cover_error": finite_cover_error,
                "reference_error": reference_error,
                "KPM_SLQ_disagreement": float(output["method_disagreement"]),
                "broadening_bias_unresolved": True,
            }
        )
        for index, energy in enumerate(grid):
            density_rows.append(
                {
                    "tower_id": tower_id,
                    "level": level,
                    "split": "holdout" if is_holdout else "pilot",
                    "energy": float(energy),
                    "eta_N": eta,
                    "coherence_weighted_density": float(coherence_density[index]),
                }
            )
        if is_holdout:
            regularity[tower_id] = regular_interval_audit(
                grid,
                coherence_density,
                output["slq_nodes"],
                max(math.pi / order, 2.0 / depth, sigma),
            )
    audit_frame = pd.DataFrame(audit_rows)
    derived_frame = pd.DataFrame(derived_rows)
    method_frame = pd.DataFrame(method_rows)
    density_frame = pd.DataFrame(density_rows)
    audit_frame.to_parquet(run_dir / "raw" / "r10_action_method_audit.parquet", index=False)
    method_frame.to_parquet(run_dir / "derived" / "r10_method_spectral_measures.parquet", index=False)
    derived_frame.to_parquet(run_dir / "derived" / "r10_error_budget.parquet", index=False)
    density_frame.to_parquet(run_dir / "derived" / "r10_coherence_density.parquet", index=False)
    method_frame.to_parquet(run_dir / "figure_data" / "figure_10_KPM_SLQ_CDF.parquet", index=False)
    derived_frame.to_parquet(run_dir / "figure_data" / "figure_10_vanishing_schedule.parquet", index=False)
    density_frame[density_frame.split == "holdout"].to_parquet(run_dir / "figure_data" / "figure_10_holdout_coherence_density.parquet", index=False)
    with (run_dir / "raw" / "r10_cross_reference.npz").open("xb") as handle:
        np.savez_compressed(handle, energy=grid, cross_reference_CDF=cross_reference)

    method_threshold = float(config["acceptance"]["KPM_SLQ_CDF_disagreement_max"])
    method_pass = bool((audit_frame.KPM_SLQ_CDF_disagreement <= method_threshold).all())
    holdout = derived_frame[derived_frame.split == "holdout"]
    weak_pass_towers = holdout[
        (holdout.kappa_N <= method_threshold)
        & (holdout.KPM_SLQ_disagreement <= method_threshold)
    ]
    weak_cdf_pass = int(len(weak_pass_towers)) >= int(config["acceptance"]["at_least_two_towers_must_pass"])
    schedule_towers = holdout[
        (holdout.kappa_over_eta <= float(config["acceptance"]["holdout_kappa_over_eta_max"]))
        & (holdout.eta_N <= float(config["acceptance"]["holdout_eta_max"]))
    ]
    schedule_pass = len(schedule_towers) >= 2
    analytic_regularity_pass = all(bool(value["analytic_uniform_modulus_certified"]) for value in regularity.values())
    if weak_cdf_pass and schedule_pass and analytic_regularity_pass:
        classification = "PASS_PIECEWISE_DOS"
    else:
        classification = "INCONCLUSIVE"
    certificate = {
        "run_id": run_id,
        "preregistration_sha256": sha256_file(config_path),
        "resource_amendment_sha256": sha256_file(resource_path),
        "anchor_checks": identity["anchor_checks"],
        "sector_scope": "transitive projective-line retained sector; not the complete regular representation",
        "KPM": {"order": order, "kernel": "Jackson", "vector": "delta_basepoint"},
        "SLQ": {"depth": depth, "reorthogonalization": "two-pass full", "smoothing_sigma": sigma, "vector": "delta_basepoint"},
        "method_pass": method_pass,
        "weak_CDF_pass": weak_cdf_pass,
        "weak_CDF_passing_towers": weak_pass_towers.tower_id.tolist(),
        "vanishing_schedule_finite_holdout_pass": schedule_pass,
        "regularity_audit": regularity,
        "analytic_uniform_regularity_certified": analytic_regularity_pass,
        "classification": classification,
        "scientific_guard": "The eta rule is frozen and evaluated, but without a tower-uniform limiting regularity theorem no local or unsmoothed DOS claim is certified.",
    }
    write_json(run_dir / "certificates" / "r10_dos_certificate.json", certificate)
    tasks = {
        "R10-A": "PASS_CERTIFIED",
        "R10-B": "PASS_CONVERGED" if weak_cdf_pass else "INCONCLUSIVE",
        "R10-C": "INCONCLUSIVE" if not analytic_regularity_pass else "PASS_CERTIFIED",
        "R10-D": "PASS_CONVERGED" if schedule_pass else "INCONCLUSIVE",
        "R10-E": "INCONCLUSIVE" if classification == "INCONCLUSIVE" else "PASS_CONVERGED",
        "R10-F": classification,
    }
    freeze = finalize_run(run_dir, tasks, classification)
    print(json.dumps({"run_id": run_id, "classification": classification, "method_pass": method_pass, "weak_CDF_pass": weak_cdf_pass, "schedule_pass": schedule_pass, "holdout": holdout[["tower_id", "kappa_N", "eta_N", "KPM_SLQ_disagreement"]].to_dict("records"), "freeze": freeze["tree_inventory_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

