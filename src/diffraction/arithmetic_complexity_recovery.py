from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def _theta_inverse(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=float)
    return np.sqrt((values * values + 3.0) / 3.0)


def _linear_extrapolation(blocks: pd.DataFrame, response: str) -> dict[str, float]:
    x = np.asarray(blocks["inverse_log_theta_at_block_start"], dtype=float)
    y = np.asarray(blocks[response], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residuals = y - fitted
    return {
        "slope": float(slope),
        "intercept_at_inverse_log_zero": float(intercept),
        "maximum_absolute_fit_residual": float(np.max(np.abs(residuals))),
        "rms_fit_residual": float(np.sqrt(np.mean(residuals * residuals))),
    }


def _criteria(config: dict, pair_blocks: pd.DataFrame, envelopes: pd.DataFrame, minimum_separation: float, g11_value: float):
    target = float(config["exact_target_exponent"])
    limits = config["acceptance"]
    pair_fit_starts = [int(value) for value in config["extrapolation"]["pair_fit_blocks"]]
    envelope_fit_starts = [int(value) for value in config["extrapolation"]["envelope_fit_blocks"]]
    pair_fit = pair_blocks[pair_blocks.block_start.isin(pair_fit_starts)].sort_values("block_start")
    envelope_fit = envelopes[envelopes.block_start.isin(envelope_fit_starts)].sort_values("block_start")
    primary_fit = _linear_extrapolation(pair_fit, "median_beta")
    upper_fit = _linear_extrapolation(envelope_fit, "upper_pointwise_exponent")
    lower_fit = _linear_extrapolation(envelope_fit, "lower_pointwise_exponent")
    tail_pairs = pair_blocks[pair_blocks.block_start.isin(config["tail_pair_blocks"])].sort_values("block_start")
    if len(tail_pairs) != 2:
        raise RuntimeError("preregistered tail pair blocks are incomplete")
    tail_center = float(tail_pairs.median_beta.mean())
    tail_drift = float(abs(tail_pairs.median_beta.iloc[-1] - tail_pairs.median_beta.iloc[-2]))
    primary = primary_fit["intercept_at_inverse_log_zero"]
    upper = upper_fit["intercept_at_inverse_log_zero"]
    lower = lower_fit["intercept_at_inverse_log_zero"]
    envelope_gap = float(abs(upper - lower))
    checks = {
        "minimum_log_scale_separation": minimum_separation >= float(config["minimum_log_scale_separation"]),
        "primary_extrapolation": abs(primary - target) <= float(limits["primary_extrapolation_absolute_error_max"]),
        "primary_tail_center": abs(tail_center - target) <= float(limits["primary_tail_center_absolute_error_max"]),
        "primary_tail_block_drift": tail_drift <= float(limits["primary_tail_block_drift_max"]),
        "g11_comparison": abs(primary - g11_value) <= float(limits["g11_comparison_absolute_difference_max"]),
        "upper_envelope_extrapolation": abs(upper - target) <= float(limits["upper_envelope_extrapolation_absolute_error_max"]),
        "lower_envelope_extrapolation": abs(lower - target) <= float(limits["lower_envelope_extrapolation_absolute_error_max"]),
        "envelope_extrapolation_gap": envelope_gap <= float(limits["envelope_extrapolation_gap_max"]),
    }
    uncertainty = {
        "tail_block_median_drift": tail_drift,
        "tail_block_q10_q90_half_widths": [
            float((row.q90_beta - row.q10_beta) / 2.0) for row in tail_pairs.itertuples(index=False)
        ],
        "tail_block_median_absolute_deviations": [float(value) for value in tail_pairs.mad_beta],
        "primary_tail_to_extrapolation_difference": float(abs(tail_center - primary)),
        "upper_lower_envelope_extrapolation_gap": envelope_gap,
        "primary_fit_rms_residual": primary_fit["rms_fit_residual"],
        "upper_fit_rms_residual": upper_fit["rms_fit_residual"],
        "lower_fit_rms_residual": lower_fit["rms_fit_residual"],
    }
    convergence_control = (
        tail_drift <= float(config["contradiction"]["require_primary_tail_drift_max"])
        and envelope_gap <= float(config["contradiction"]["require_envelope_extrapolation_gap_max"])
    )
    margin = float(config["contradiction"]["same_side_exclusion_margin"])
    estimates = [primary, tail_center, upper, lower]
    same_side_contradiction = all(value > target + margin for value in estimates) or all(
        value < target - margin for value in estimates
    )
    return {
        "checks": checks,
        "primary_fit": primary_fit,
        "upper_fit": upper_fit,
        "lower_fit": lower_fit,
        "primary_tail_center": tail_center,
        "uncertainty": uncertainty,
        "convergence_control_for_contradiction": bool(convergence_control),
        "same_side_contradiction": bool(same_side_contradiction),
    }


def run(config, recovery_config, run_dir: Path, run_id: str, root: Path, context):
    spec = recovery_config["d08_preregistered_estimator"]
    if not spec.get("frozen_before_outcome") or not spec.get("no_posthoc_changes"):
        raise RuntimeError("D-08 estimator is not preregistered")
    if int(spec["fixed_ratio_c"]) != 2 or not spec.get("adjacent_point_regression_forbidden"):
        raise RuntimeError("D-08 recovery must use the preregistered c=2 scale separation")
    raw = run_dir / "raw" / "d08_arithmetic_complexity_recovery"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d08_arithmetic_complexity_recovery"
    derived.mkdir(parents=True, exist_ok=False)
    certificate = run_dir / "certificates" / "d08_arithmetic_complexity_recovery.json"
    growth = pd.read_parquet(root / str(config["arithmetic_sources"]["growth"])).copy()
    required = {"j", str(spec["q_column"])}
    if not required.issubset(growth.columns):
        raise RuntimeError(f"missing D-08 input columns: {sorted(required - set(growth.columns))}")
    growth = growth.sort_values("j").reset_index(drop=True)
    q_by_j = {int(row.j): int(getattr(row, spec["q_column"])) for row in growth.itertuples(index=False)}
    pair_start = int(spec["pair_j_min_inclusive"])
    pair_stop = int(spec["pair_j_max_exclusive"])
    ratio = int(spec["fixed_ratio_c"])
    pair_rows = []
    for j in range(pair_start, pair_stop):
        if j not in q_by_j or ratio * j not in q_by_j:
            raise RuntimeError(f"missing preregistered scale-separated pair {j},{ratio*j}")
        inv_j, inv_2j = _theta_inverse(np.asarray([j, ratio * j]))
        x_j, x_2j = math.log(float(inv_j)), math.log(float(inv_2j))
        denominator = x_2j - x_j
        beta = (math.log(q_by_j[ratio * j]) - math.log(q_by_j[j])) / denominator
        block_start = 1 << (j.bit_length() - 1)
        pair_rows.append({
            "j": j, "ratio": ratio, "j_scaled": ratio * j, "block_start": block_start,
            "q_j": q_by_j[j], "q_scaled": q_by_j[ratio * j],
            "theta_inverse_j": float(inv_j), "theta_inverse_scaled": float(inv_2j),
            "log_theta_inverse_j": x_j, "log_theta_inverse_scaled": x_2j,
            "delta_log_theta_inverse": denominator, "beta_effective": beta,
        })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_parquet(raw / "scale_separated_pairs.parquet", index=False)
    pairs.to_parquet(derived / "scale_separated_effective_exponents.parquet", index=False)
    quantile_low, quantile_high = [float(value) for value in spec["uncertainty_quantiles"]]
    block_rows = []
    for block_start in [int(value) for value in spec["pair_block_starts"]]:
        subset = pairs[pairs.block_start == block_start]
        if len(subset) != block_start:
            raise RuntimeError(f"incomplete preregistered pair block {block_start}")
        values = np.asarray(subset.beta_effective, dtype=float)
        median = float(np.median(values))
        block_rows.append({
            "block_start": block_start, "pair_count": len(subset),
            "inverse_log_theta_at_block_start": 1.0 / math.log(float(_theta_inverse(np.asarray([block_start]))[0])),
            "median_beta": median, "mean_beta": float(np.mean(values)),
            "q10_beta": float(np.quantile(values, quantile_low)),
            "q90_beta": float(np.quantile(values, quantile_high)),
            "mad_beta": float(np.median(np.abs(values - median))),
            "minimum_beta": float(np.min(values)), "maximum_beta": float(np.max(values)),
        })
    pair_blocks = pd.DataFrame(block_rows)
    pair_blocks.to_parquet(derived / "pair_block_summaries.parquet", index=False)
    envelope_rows = []
    for block_start in [int(value) for value in spec["envelope_block_starts"]]:
        subset = growth[(growth.j >= block_start) & (growth.j < 2 * block_start)].copy()
        if len(subset) != block_start:
            raise RuntimeError(f"incomplete preregistered envelope block {block_start}")
        indices = np.asarray(subset.j, dtype=int)
        q_values = np.asarray(subset[spec["q_column"]], dtype=np.int64)
        pointwise = np.log(q_values.astype(float)) / np.log(_theta_inverse(indices))
        upper_index = int(np.argmax(pointwise))
        lower_index = int(np.argmin(pointwise))
        envelope_rows.append({
            "block_start": block_start, "block_stop_exclusive": 2 * block_start,
            "point_count": len(subset),
            "inverse_log_theta_at_block_start": 1.0 / math.log(float(_theta_inverse(np.asarray([block_start]))[0])),
            "q_upper_envelope": int(np.max(q_values)), "q_lower_envelope": int(np.min(q_values)),
            "upper_pointwise_exponent": float(pointwise[upper_index]),
            "upper_exponent_index": int(indices[upper_index]),
            "lower_pointwise_exponent": float(pointwise[lower_index]),
            "lower_exponent_index": int(indices[lower_index]),
            "median_pointwise_exponent": float(np.median(pointwise)),
        })
    envelopes = pd.DataFrame(envelope_rows)
    envelopes.to_parquet(raw / "dyadic_arithmetic_envelopes.parquet", index=False)
    envelopes.to_parquet(derived / "dyadic_arithmetic_envelopes.parquet", index=False)
    g10 = json.loads((root / str(config["arithmetic_sources"]["g10_certificate"])).read_text(encoding="utf-8"))
    g11 = json.loads((root / str(config["arithmetic_sources"]["g11_certificate"])).read_text(encoding="utf-8"))
    g11_expected = float(recovery_config["g11_frozen_reference"]["extrapolation"])
    if abs(float(g11["numerical_extrapolate_to_inverse_log_zero"]) - g11_expected) > 1.0e-12:
        raise RuntimeError("frozen G-11 extrapolation changed")
    minimum_separation = float(pairs.delta_log_theta_inverse.min())
    diagnostics = _criteria(spec, pair_blocks, envelopes, minimum_separation, g11_expected)
    sources_valid = (
        g10.get("status") == "PASS_EXACT"
        and g11.get("status") == "PASS_CONVERGED"
        and bool(g11.get("compatible_with_4"))
        and bool(g11.get("incompatible_with_1"))
    )
    passed = sources_valid and all(diagnostics["checks"].values())
    contradicted = (
        sources_valid
        and diagnostics["convergence_control_for_contradiction"]
        and diagnostics["same_side_contradiction"]
    )
    if passed:
        status = "PASS_CERTIFIED"
    elif contradicted:
        status = "FAIL_THEORY"
    else:
        status = str(spec["nonpassing_noncontradictory_status"])
    summary = {
        "task_id": "D-08", "run_id": run_id, "status": status,
        "estimator_version": spec["estimator_version"], "fixed_ratio_c": ratio,
        "pair_index_range": [pair_start, pair_stop], "pair_count": len(pairs),
        "pair_block_starts": spec["pair_block_starts"],
        "envelope_block_starts": spec["envelope_block_starts"],
        "minimum_log_scale_separation": minimum_separation,
        "primary_extrapolated_exponent": diagnostics["primary_fit"]["intercept_at_inverse_log_zero"],
        "primary_tail_center": diagnostics["primary_tail_center"],
        "upper_envelope_extrapolated_exponent": diagnostics["upper_fit"]["intercept_at_inverse_log_zero"],
        "lower_envelope_extrapolated_exponent": diagnostics["lower_fit"]["intercept_at_inverse_log_zero"],
        "uncertainty_and_finite_scale": diagnostics["uncertainty"],
        "acceptance_checks": diagnostics["checks"],
        "convergence_control_for_contradiction": diagnostics["convergence_control_for_contradiction"],
        "same_side_contradiction": diagnostics["same_side_contradiction"],
        "exact_target_exponent": spec["exact_target_exponent"],
        "g11_frozen_extrapolation": g11_expected,
        "g10_status": g10.get("status"), "g11_status": g11.get("status"),
        "g10_formula": g10.get("formula"),
        "adjacent_point_regression_used": False,
        "posthoc_window_selection_used": False,
        "nonpassing_rule": "FAIL_THEORY only for converged same-side exclusion; otherwise INCONCLUSIVE",
    }
    write_json(raw / "estimator_definition.json", spec)
    write_json(derived / "estimator_summary.json", summary)
    write_json(certificate, summary)
    context["d08_recovery"] = summary
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
