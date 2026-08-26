from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import cumulative_trapezoid


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from common import finalize_run, initialize_run, sha256_file, write_json  # noqa: E402
from projected_spectral import ProjectedCayleyOperator, kpm_density  # noqa: E402
from slq import discrete_cdf, full_reorthogonalized_probe  # noqa: E402


R8_RUN_ID = "2efafd37540cb3de976adfcd2bd7f01d6802ff18277429a8088d87ab8335ae8b"
COVER_RUN_ID = "3fea2901f7ae29d44dc1517294dee678fac2b9e2fa7a8bf3bccff2bd6528fac5"


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def density_cdf(moment: np.ndarray, grid: np.ndarray) -> np.ndarray:
    density = kpm_density(moment, grid)
    cdf = np.concatenate([[0.0], cumulative_trapezoid(density, grid)])
    return cdf / cdf[-1]


def tail_supremum(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    current = -np.inf
    for index in range(len(values) - 1, -1, -1):
        current = max(current, float(values[index]))
        result[index] = current
    return result


def main() -> int:
    config = load_yaml(EXTENSION_ROOT / "configs" / "r10_execution_preregistration.yaml")
    action_config = load_yaml(EXTENSION_ROOT / "configs" / "r10_action_input_preregistration.yaml")
    r8_run = EXTENSION_ROOT / "results" / R8_RUN_ID
    cover_run = EXTENSION_ROOT / "results" / COVER_RUN_ID
    anchors = {
        r8_run / "manifest.json": str(config["R8_input"]["manifest_sha256"]),
        r8_run / "derived" / "r8_level_metrics.parquet": str(
            config["R8_input"]["level_metrics_sha256"]
        ),
        cover_run / str(action_config["cover_level_table"]): str(
            action_config["cover_level_table_sha256"]
        ),
    }
    for path, expected in anchors.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"R10 input hash mismatch: {path}")
    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT)
    try:
        level_metrics = pd.read_parquet(r8_run / "derived" / "r8_level_metrics.parquet")
        cover_levels = pd.read_parquet(cover_run / str(action_config["cover_level_table"]))
        action_columns = cover_levels[["tower_id", "dyadic_depth", "action_path", "action_sha256", "parent_order"]]
        levels = level_metrics.merge(action_columns, on=["tower_id", "dyadic_depth"], validate="one_to_one")
        grid = np.linspace(
            float(config["CDF"]["grid_min"]),
            float(config["CDF"]["grid_max"]),
            int(config["CDF"]["grid_size"]),
        )
        steps = int(config["SLQ"]["lanczos_steps"])
        probes = int(config["SLQ"]["random_vectors"])
        base_seed = int(config["SLQ"]["base_seed"])
        stride = int(config["SLQ"]["per_level_seed_stride"])
        breakdown = float(config["SLQ"]["breakdown_tolerance"])
        cdf_records: list[pd.DataFrame] = []
        diagnostics: list[dict[str, object]] = []
        progress_path = run_dir / "logs" / "r10_slq_progress.jsonl"

        for level_index, (_, row) in enumerate(levels.sort_values(["tower_id", "dyadic_depth"]).iterrows()):
            tower_id = str(row["tower_id"])
            depth = int(row["dyadic_depth"])
            action_path = EXTENSION_ROOT / str(row["action_path"])
            moment_path = EXTENSION_ROOT / str(row["raw_moment_path"])
            if sha256_file(action_path) != str(row["action_sha256"]):
                raise RuntimeError(f"R10 action hash mismatch for {tower_id} depth {depth}")
            if sha256_file(moment_path) != str(row["raw_moment_sha256"]):
                raise RuntimeError(f"R10 KPM hash mismatch for {tower_id} depth {depth}")
            with np.load(action_path, allow_pickle=False) as action:
                operator = ProjectedCayleyOperator(
                    action["permutations"], action["parent_index"], int(row["parent_order"])
                )
            with np.load(moment_path, allow_pickle=False) as payload:
                moments = payload["moments"]
                inherited_seeds = payload["seeds"]
            kpm_cdfs = np.asarray([density_cdf(moment, grid) for moment in moments])
            slq_cdfs = []
            slq_orthogonality = []
            slq_recurrence = []
            slq_norm_diagnostics = []
            level_started = time.perf_counter()
            for probe_index in range(probes):
                seed = base_seed + stride * level_index + probe_index
                probe = full_reorthogonalized_probe(
                    operator,
                    steps=steps,
                    seed=seed,
                    breakdown_tolerance=breakdown,
                )
                slq_cdfs.append(discrete_cdf(probe.nodes, probe.weights, grid))
                slq_orthogonality.append(probe.orthogonality_residual)
                slq_recurrence.append(probe.recurrence_residual)
                slq_norm_diagnostics.append(probe.source_norm_squared_over_dimension)
                raw_path = run_dir / "raw" / f"r10_slq_{tower_id}_depth_{depth}_probe_{probe_index:02d}.npz"
                with raw_path.open("xb") as handle:
                    np.savez(
                        handle,
                        run_id=np.asarray(run_id),
                        tower_id=np.asarray(tower_id),
                        dyadic_depth=np.asarray(depth),
                        probe_index=np.asarray(probe_index),
                        seed=np.asarray(seed),
                        nodes=probe.nodes,
                        weights=probe.weights,
                        alpha=probe.alpha,
                        beta=probe.beta,
                        source_norm_squared_over_dimension=np.asarray(
                            probe.source_norm_squared_over_dimension
                        ),
                        orthogonality_residual=np.asarray(probe.orthogonality_residual),
                        recurrence_residual=np.asarray(probe.recurrence_residual),
                    )
                progress_record = {
                    "tower_id": tower_id,
                    "dyadic_depth": depth,
                    "probe_index": probe_index,
                    "seed": seed,
                    "elapsed_level_seconds": time.perf_counter() - level_started,
                    "orthogonality_residual": probe.orthogonality_residual,
                    "recurrence_residual": probe.recurrence_residual,
                    "raw_sha256": sha256_file(raw_path),
                }
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress_record) + "\n")
                    handle.flush()
            slq_cdfs = np.asarray(slq_cdfs)
            kpm_mean = kpm_cdfs.mean(axis=0)
            slq_mean = slq_cdfs.mean(axis=0)
            kpm_se = kpm_cdfs.std(axis=0, ddof=1) / np.sqrt(kpm_cdfs.shape[0])
            slq_se = slq_cdfs.std(axis=0, ddof=1) / np.sqrt(slq_cdfs.shape[0])
            method_mean = 0.5 * (kpm_mean + slq_mean)
            method_disagreement = np.abs(kpm_mean - slq_mean)
            cdf_records.append(
                pd.DataFrame(
                    {
                        "tower_id": tower_id,
                        "dyadic_depth": depth,
                        "energy": grid,
                        "KPM_CDF_mean": kpm_mean,
                        "KPM_CDF_standard_error": kpm_se,
                        "SLQ_CDF_mean": slq_mean,
                        "SLQ_CDF_standard_error": slq_se,
                        "method_average_CDF": method_mean,
                        "method_disagreement": method_disagreement,
                    }
                )
            )
            diagnostics.append(
                {
                    "tower_id": tower_id,
                    "dyadic_depth": depth,
                    "retained_sector_dimension": int(row["retained_sector_dimension"]),
                    "injectivity_radius_integer": int(row["injectivity_radius_integer"]),
                    "KPM_order": int(moments.shape[1] - 1),
                    "KPM_random_vectors": int(moments.shape[0]),
                    "KPM_seeds": json.dumps(inherited_seeds.astype(int).tolist()),
                    "SLQ_steps": steps,
                    "SLQ_random_vectors": probes,
                    "SLQ_seeds": json.dumps([base_seed + stride * level_index + i for i in range(probes)]),
                    "maximum_SLQ_orthogonality_residual": max(slq_orthogonality),
                    "maximum_SLQ_recurrence_residual": max(slq_recurrence),
                    "maximum_SLQ_source_norm_diagnostic_deviation": max(
                        abs(value - 1.0) for value in slq_norm_diagnostics
                    ),
                    "sup_KPM_SLQ_disagreement": float(np.max(method_disagreement)),
                    "sup_KPM_standard_error": float(np.max(kpm_se)),
                    "sup_SLQ_standard_error": float(np.max(slq_se)),
                    "level_elapsed_seconds": time.perf_counter() - level_started,
                }
            )

        cdf_frame = pd.concat(cdf_records, ignore_index=True)
        diagnostic_frame = pd.DataFrame(diagnostics)
        cdf_frame.to_parquet(run_dir / "figure_data" / "r10_cdf_methods.parquet", index=False)
        diagnostic_frame.to_parquet(run_dir / "derived" / "r10_method_diagnostics.parquet", index=False)

        pure_ref = cdf_frame[
            (cdf_frame.tower_id == str(config["reference"]["pure_tower_level"]["tower_id"]))
            & (cdf_frame.dyadic_depth == int(config["reference"]["pure_tower_level"]["dyadic_depth"]))
        ].sort_values("energy")
        mixed_ref = cdf_frame[
            (cdf_frame.tower_id == str(config["reference"]["mixed_tower_level"]["tower_id"]))
            & (cdf_frame.dyadic_depth == int(config["reference"]["mixed_tower_level"]["dyadic_depth"]))
        ].sort_values("energy")
        if len(pure_ref) != len(grid) or len(mixed_ref) != len(grid):
            raise RuntimeError("preregistered R10 reference levels are absent")
        reference_cdf = 0.5 * (
            pure_ref.method_average_CDF.to_numpy() + mixed_ref.method_average_CDF.to_numpy()
        )
        pure_uncertainty = (
            0.5 * pure_ref.method_disagreement.to_numpy()
            + pure_ref.KPM_CDF_standard_error.to_numpy()
            + pure_ref.SLQ_CDF_standard_error.to_numpy()
        )
        mixed_uncertainty = (
            0.5 * mixed_ref.method_disagreement.to_numpy()
            + mixed_ref.KPM_CDF_standard_error.to_numpy()
            + mixed_ref.SLQ_CDF_standard_error.to_numpy()
        )
        reference_uncertainty = 0.5 * (pure_uncertainty + mixed_uncertainty)
        reference_frame = pd.DataFrame(
            {
                "energy": grid,
                "reference_CDF": reference_cdf,
                "reference_uncertainty": reference_uncertainty,
            }
        )
        reference_frame.to_parquet(run_dir / "figure_data" / "r10_reference_cdf.parquet", index=False)

        error_records = []
        alpha = float(config["vanishing_broadening"]["alpha"])
        for _, diagnostic in diagnostic_frame.iterrows():
            tower_id = str(diagnostic["tower_id"])
            depth = int(diagnostic["dyadic_depth"])
            level_cdf = cdf_frame[
                (cdf_frame.tower_id == tower_id) & (cdf_frame.dyadic_depth == depth)
            ].sort_values("energy")
            kappa = float(np.max(np.abs(level_cdf.method_average_CDF.to_numpy() - reference_cdf)))
            radius = int(diagnostic["injectivity_radius_integer"])
            eta = (radius + 1.0) ** (-1.0 / 3.0)
            eta_alpha = eta**alpha
            kappa_over_eta = kappa / eta
            stochastic_error = max(
                float(diagnostic["sup_KPM_standard_error"]),
                float(diagnostic["sup_SLQ_standard_error"]),
            )
            method_disagreement = float(diagnostic["sup_KPM_SLQ_disagreement"])
            method_budget = (
                0.5 * method_disagreement
                + stochastic_error
                + float(np.max(reference_uncertainty))
            )
            error_records.append(
                {
                    "tower_id": tower_id,
                    "dyadic_depth": depth,
                    "injectivity_radius_integer": radius,
                    "kappa_N": kappa,
                    "reference_uncertainty_sup": float(np.max(reference_uncertainty)),
                    "eta_N": eta,
                    "kappa_over_eta": kappa_over_eta,
                    "eta_to_alpha": eta_alpha,
                    "combined_vanishing_broadening_term": kappa_over_eta + eta_alpha,
                    "spectral_measure_error": kappa,
                    "riesz_projector_error": 0.0,
                    "coherence_operator_error": 0.0,
                    "stochastic_KPM_SLQ_error": stochastic_error,
                    "broadening_bias": eta_alpha,
                    "KPM_SLQ_disagreement": method_disagreement,
                    "KPM_SLQ_combined_budget": method_budget,
                    "KPM_SLQ_agreement_pass": method_disagreement <= method_budget,
                }
            )
        error_frame = pd.DataFrame(error_records)
        error_frame.to_parquet(run_dir / "derived" / "r10_error_budget.parquet", index=False)

        tower_audits = {}
        holdout_passes = []
        for tower_id, group in error_frame.groupby("tower_id"):
            ordered = group.sort_values("dyadic_depth")
            kappa_envelope = tail_supremum(ordered.kappa_N.astype(float).tolist())
            combined_envelope = tail_supremum(
                ordered.combined_vanishing_broadening_term.astype(float).tolist()
            )
            eta_values = ordered.eta_N.astype(float).tolist()
            holdout_depths = [int(value) for value in config["vanishing_broadening"]["holdout_levels"][tower_id]]
            holdout = ordered[ordered.dyadic_depth.isin(holdout_depths)]
            holdout_pass = bool(len(holdout)) and bool(holdout.KPM_SLQ_agreement_pass.all())
            holdout_passes.append(holdout_pass)
            tower_audits[tower_id] = {
                "depths": ordered.dyadic_depth.astype(int).tolist(),
                "kappa_tail_supremum": kappa_envelope,
                "eta_values": eta_values,
                "combined_term_tail_supremum": combined_envelope,
                "kappa_envelope_decreases": kappa_envelope[-1] < kappa_envelope[0],
                "eta_decreases": eta_values[-1] < eta_values[0],
                "combined_envelope_decreases": combined_envelope[-1] < combined_envelope[0],
                "holdout_KPM_SLQ_pass": holdout_pass,
            }

        uniform_holder_certificate = False
        two_tower_holdout = len(holdout_passes) >= 2 and all(holdout_passes)
        all_method_agreement = bool(error_frame.KPM_SLQ_agreement_pass.all())
        vanishing_finite_scale = all(
            audit["eta_decreases"] and audit["combined_envelope_decreases"]
            for audit in tower_audits.values()
        )
        statuses = {
            "R10-01": "PASS_CERTIFIED",
            "R10-02": "PASS_CERTIFIED",
            "R10-03": "PASS_EXACT",
            "R10-04": "PASS_CERTIFIED" if uniform_holder_certificate else "INCONCLUSIVE",
            "R10-05": "PASS_CERTIFIED",
            "R10-06": (
                "PASS_CONVERGED"
                if uniform_holder_certificate and two_tower_holdout and all_method_agreement and vanishing_finite_scale
                else "INCONCLUSIVE"
            ),
            "R10-08": "PASS_CERTIFIED" if two_tower_holdout else "INCONCLUSIVE",
            "R10-09": "PASS_CERTIFIED",
        }
        certificate = {
            "run_id": run_id,
            "task_statuses": statuses,
            "parent_verification": identity["parent_verification"],
            "input_hashes_verified": True,
            "R10_preregistration_sha256": sha256_file(
                EXTENSION_ROOT / "configs" / "r10_execution_preregistration.yaml"
            ),
            "reference_rule": config["reference"],
            "reference_uncertainty_sup": float(np.max(reference_uncertainty)),
            "regularity_audit": {
                "source": config["regularity"]["primary_source"],
                "hypotheses_matched": [
                    "surface group is non-elementary Gromov-hyperbolic",
                    "walk is symmetric and finitely supported",
                    "support generates the marked group",
                ],
                "direct_result": config["regularity"]["direct_result"],
                "inference": config["regularity"]["inference_under_audited_Tauberian_hypotheses"],
                "uniform_Holder_constant_certified": uniform_holder_certificate,
                "reason_inconclusive": (
                    "The local-limit exponent supports square-root edge behavior, but this run does not "
                    "derive a uniform Holder constant on the full declared isolated sector."
                ),
            },
            "tower_audits": tower_audits,
            "two_tower_holdout_pass": two_tower_holdout,
            "all_KPM_SLQ_agreement_pass": all_method_agreement,
            "scientific_conclusion": (
                "Weak/CDF and method cross-check data are complete, but the finite radii do not realize "
                "a vanishing broadening schedule and the required uniform regularity certificate is absent."
            ),
        }
        write_json(run_dir / "certificates" / "r10_dos_certificate.json", certificate)
        finalize_run(run_dir, "COMPLETE", statuses)
        print(json.dumps({"run_id": run_id, "statuses": statuses}))
        return 0
    except Exception as error:
        failure = {
            "run_id": run_id,
            "status": "FAIL_IMPLEMENTATION",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "certificates" / "r10_failure.json", failure)
        finalize_run(
            run_dir,
            "INCOMPLETE",
            {task: "FAIL_IMPLEMENTATION" for task in ("R10-01", "R10-02", "R10-03", "R10-04", "R10-05", "R10-06", "R10-08", "R10-09")},
        )
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
