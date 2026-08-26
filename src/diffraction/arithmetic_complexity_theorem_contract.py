from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json
from audit.run_manifest import sha256_file


def _theta(indices: np.ndarray) -> np.ndarray:
    return 2.0 * np.arctan(np.sqrt(3.0) / np.asarray(indices, dtype=float))


def _fit(frame: pd.DataFrame, response: str) -> dict[str, float | int | str]:
    predictor_columns = {
        "median_normalized_residual": "inverse_log_theta_at_median_index",
        "upper_normalized_residual": "inverse_log_theta_at_upper_index",
        "lower_normalized_residual": "inverse_log_theta_at_lower_index",
    }
    x = frame[predictor_columns[response]].to_numpy(dtype=float)
    y = frame[response].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return {
        "response": response,
        "slope": float(slope),
        "intercept_at_inverse_log_zero": float(intercept),
        "rms_fit_residual": float(np.sqrt(np.mean(residual * residual))),
        "point_count": int(len(frame)),
    }


def _leave_one_out(frame: pd.DataFrame, response: str) -> list[dict[str, float | int | str]]:
    records = []
    for omitted in frame["block_start"].tolist():
        fit = _fit(frame[frame.block_start != omitted], response)
        records.append({
            "response": response,
            "omitted_block_start": int(omitted),
            "intercept_at_inverse_log_zero": fit["intercept_at_inverse_log_zero"],
        })
    return records


def _validate_contract(recovery: dict) -> dict:
    contract = recovery["theorem_contract"]
    spec = recovery["d08_preregistered_theorem_matched_test"]
    if not contract.get("frozen_before_new_outcome") or contract.get("branch") != "A_LOG_ASYMPTOTIC_EXPONENT":
        raise RuntimeError("D-08 theorem contract is not frozen on branch A")
    if contract.get("stronger_regular_variation_claim_registered") is not False:
        raise RuntimeError("strong regular variation must not be inferred from theorem A")
    if not spec.get("frozen_before_new_outcome") or not spec.get("no_posthoc_changes"):
        raise RuntimeError("D-08 theorem-matched test was not preregistered")
    if spec.get("fixed_ratio_diagnostic", {}).get("acceptance_role_for_theorem_A") != "NONE":
        raise RuntimeError("fixed-ratio diagnostic cannot be a theorem-A acceptance condition")
    if not spec.get("fixed_ratio_result_recalculation_forbidden") or not spec.get("adjacent_point_regression_forbidden"):
        raise RuntimeError("forbidden estimator guard is missing")
    return spec


def run(config, recovery_config, run_dir: Path, run_id: str, root: Path, context):
    spec = _validate_contract(recovery_config)
    raw = run_dir / "raw" / "d08_theorem_matched"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d08_theorem_matched"
    derived.mkdir(parents=True, exist_ok=False)
    certificate = run_dir / "certificates" / "d08_theorem_matched.json"

    growth_path = root / str(config["arithmetic_sources"]["growth"])
    growth = pd.read_parquet(growth_path).sort_values("j").reset_index(drop=True)
    q_column = str(spec["q_column"])
    if not {"j", q_column}.issubset(growth.columns):
        raise RuntimeError("theorem-matched input is missing j or q")
    j_min = int(spec["j_min_inclusive"])
    j_stop = int(spec["j_max_exclusive"])
    selected = growth[(growth.j >= j_min) & (growth.j < j_stop)][["j", q_column]].copy()
    if selected.j.tolist() != list(range(j_min, j_stop)):
        raise RuntimeError("the preregistered theorem-matched index interval is incomplete")
    selected.rename(columns={q_column: "q_j"}).to_parquet(raw / "theorem_matched_inputs.parquet", index=False)

    indices = selected.j.to_numpy(dtype=np.int64)
    q_values = selected[q_column].to_numpy(dtype=np.int64)
    theta = _theta(indices)
    log_theta_inverse = np.log(1.0 / np.abs(theta))
    log_q = np.log(q_values.astype(float))
    target = float(spec["target_exponent"])
    residual = log_q - target * log_theta_inverse
    normalized = residual / log_theta_inverse
    pointwise = log_q / log_theta_inverse
    identity_error = float(np.max(np.abs(normalized - (pointwise - target))))
    rows = pd.DataFrame({
        "j": indices,
        "theta_j": theta,
        "log_theta_inverse": log_theta_inverse,
        "log_q_j": log_q,
        "R_j": residual,
        "normalized_residual": normalized,
        "pointwise_log_exponent": pointwise,
    })
    rows.to_parquet(derived / "theorem_matched_residuals.parquet", index=False)

    quantile_low, quantile_high = [float(value) for value in spec["uncertainty_quantiles"]]
    block_records = []
    for block_start in [int(value) for value in spec["dyadic_block_starts"]]:
        subset = rows[(rows.j >= block_start) & (rows.j < 2 * block_start)]
        if len(subset) != block_start:
            raise RuntimeError(f"incomplete preregistered theorem block {block_start}")
        values = subset.normalized_residual.to_numpy(dtype=float)
        exponents = subset.pointwise_log_exponent.to_numpy(dtype=float)
        local_indices = subset.j.to_numpy(dtype=np.int64)
        median_value = float(np.median(values))
        median_position = int(np.argmin(np.abs(values - median_value)))
        upper_position = int(np.argmax(values))
        lower_position = int(np.argmin(values))
        theta_start = abs(float(_theta(np.asarray([block_start]))[0]))
        block_records.append({
            "block_start": block_start,
            "block_stop_exclusive": 2 * block_start,
            "point_count": int(len(subset)),
            "inverse_log_theta_at_block_start": float(1.0 / math.log(1.0 / theta_start)),
            "median_normalized_residual": median_value,
            "median_realizing_index": int(local_indices[median_position]),
            "upper_realizing_index": int(local_indices[upper_position]),
            "lower_realizing_index": int(local_indices[lower_position]),
            "inverse_log_theta_at_median_index": float(1.0 / subset.log_theta_inverse.iloc[median_position]),
            "inverse_log_theta_at_upper_index": float(1.0 / subset.log_theta_inverse.iloc[upper_position]),
            "inverse_log_theta_at_lower_index": float(1.0 / subset.log_theta_inverse.iloc[lower_position]),
            "mean_normalized_residual": float(np.mean(values)),
            "q10_normalized_residual": float(np.quantile(values, quantile_low)),
            "q90_normalized_residual": float(np.quantile(values, quantile_high)),
            "upper_normalized_residual": float(np.max(values)),
            "lower_normalized_residual": float(np.min(values)),
            "maximum_absolute_normalized_residual": float(np.max(np.abs(values))),
            "median_absolute_normalized_residual": float(np.median(np.abs(values))),
            "upper_pointwise_exponent": float(np.max(exponents)),
            "lower_pointwise_exponent": float(np.min(exponents)),
            "median_pointwise_exponent": float(np.median(exponents)),
        })
    blocks = pd.DataFrame(block_records)
    blocks.to_parquet(derived / "theorem_residual_block_summaries.parquet", index=False)
    blocks.to_parquet(derived / "dyadic_arithmetic_envelopes.parquet", index=False)
    plot_blocks = blocks[["block_start", "median_pointwise_exponent"]].rename(
        columns={"median_pointwise_exponent": "median_beta"}
    )
    plot_blocks.to_parquet(derived / "pair_block_summaries.parquet", index=False)

    fit_starts = [int(value) for value in spec["fit_blocks"]]
    fit_frame = blocks[blocks.block_start.isin(fit_starts)].sort_values("block_start")
    if fit_frame.block_start.tolist() != fit_starts:
        raise RuntimeError("the preregistered fit blocks are incomplete")
    responses = ["median_normalized_residual", "upper_normalized_residual", "lower_normalized_residual"]
    fits = {response: _fit(fit_frame, response) for response in responses}
    loo = [record for response in responses for record in _leave_one_out(fit_frame, response)]
    maximum_loo = float(max(abs(float(record["intercept_at_inverse_log_zero"])) for record in loo))
    upper = float(fits["upper_normalized_residual"]["intercept_at_inverse_log_zero"])
    lower = float(fits["lower_normalized_residual"]["intercept_at_inverse_log_zero"])
    median = float(fits["median_normalized_residual"]["intercept_at_inverse_log_zero"])
    envelope_gap = float(abs(upper - lower))
    first, last = blocks.iloc[0], blocks.iloc[-1]
    tiny = np.finfo(float).tiny
    median_ratio = float(abs(last.median_normalized_residual) / max(abs(first.median_normalized_residual), tiny))
    maximum_ratio = float(last.maximum_absolute_normalized_residual / max(first.maximum_absolute_normalized_residual, tiny))

    contract = recovery_config["theorem_contract"]
    g10_path = root / str(contract["g10"]["path"])
    g11_path = root / str(contract["g11"]["path"])
    frozen_path = root / str(recovery_config["frozen_predecessor"]["d08_certificate"])
    for path, expected, label in (
        (g10_path, contract["g10"]["sha256"], "G-10"),
        (g11_path, contract["g11"]["sha256"], "G-11"),
        (frozen_path, recovery_config["frozen_predecessor"]["d08_certificate_sha256"], "frozen D-08"),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label} certificate hash changed")
    g10 = json.loads(g10_path.read_text(encoding="utf-8"))
    g11 = json.loads(g11_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    metrics = recovery_config["frozen_predecessor"]["frozen_metrics"]
    for key, source_key in (
        ("fixed_ratio_extrapolation", "primary_extrapolated_exponent"),
        ("fixed_ratio_tail_center", "primary_tail_center"),
        ("dyadic_upper_envelope_extrapolation", "upper_envelope_extrapolated_exponent"),
        ("dyadic_lower_envelope_extrapolation", "lower_envelope_extrapolated_exponent"),
    ):
        if abs(float(metrics[key]) - float(frozen[source_key])) > 1.0e-12:
            raise RuntimeError(f"frozen D-08 metric changed: {key}")

    old_checks = frozen["acceptance_checks"]
    limits = spec["acceptance"]
    checks = {
        "residual_identity": identity_error <= float(limits["residual_identity_absolute_tolerance"]),
        "upper_residual_extrapolation": abs(upper) <= float(limits["upper_residual_extrapolation_absolute_max"]),
        "lower_residual_extrapolation": abs(lower) <= float(limits["lower_residual_extrapolation_absolute_max"]),
        "residual_envelope_extrapolation_gap": envelope_gap <= float(limits["residual_envelope_extrapolation_gap_max"]),
        "leave_one_out_stability": maximum_loo <= float(limits["leave_one_out_intercept_absolute_max"]),
        "median_finite_scale_contraction": median_ratio <= float(limits["last_to_first_median_absolute_ratio_max"]),
        "maximum_finite_scale_contraction": maximum_ratio <= float(limits["last_to_first_maximum_absolute_ratio_max"]),
        "frozen_upper_envelope": bool(old_checks.get("upper_envelope_extrapolation")),
        "frozen_lower_envelope": bool(old_checks.get("lower_envelope_extrapolation")),
        "frozen_envelope_gap": bool(old_checks.get("envelope_extrapolation_gap")),
        "g10_exact_source": g10.get("status") == "PASS_EXACT" and bool(g10.get("all_local_products_equal_global_degree")),
        "g11_converged_source": g11.get("status") == "PASS_CONVERGED" and bool(g11.get("compatible_with_4")),
    }
    convergence_control = checks["residual_envelope_extrapolation_gap"] and checks["leave_one_out_stability"]
    margin = float(spec["contradiction"]["same_side_residual_exclusion_margin"])
    same_side = (upper > margin and lower > margin) or (upper < -margin and lower < -margin)
    loo_intercepts = [float(record["intercept_at_inverse_log_zero"]) for record in loo]
    loo_same_side = all(value > margin for value in loo_intercepts) or all(value < -margin for value in loo_intercepts)
    contradicted = bool(convergence_control and same_side and loo_same_side and checks["g10_exact_source"])
    status = "PASS_CERTIFIED" if all(checks.values()) else ("FAIL_THEORY" if contradicted else str(spec["nonpassing_noncontradictory_status"]))

    diagnostics = pd.DataFrame([
        {"diagnostic": "theorem_matched_log_exponent", "estimate": target + median, "acceptance_role": "PRIMARY_THEOREM_A", "interpretation": "log-asymptotic exponent"},
        {"diagnostic": "fixed_ratio_c2_exponent", "estimate": float(metrics["fixed_ratio_extrapolation"]), "acceptance_role": "NONE_FOR_THEOREM_A", "interpretation": "doubling regular-variation diagnostic"},
        {"diagnostic": "g11_frozen_extrapolation", "estimate": float(metrics["g11_extrapolation"]), "acceptance_role": "INDEPENDENT_EVIDENCE", "interpretation": "frozen G-11 log exponent"},
        {"diagnostic": "frozen_dyadic_upper_envelope", "estimate": float(metrics["dyadic_upper_envelope_extrapolation"]), "acceptance_role": "INDEPENDENT_ENVELOPE", "interpretation": "frozen upper envelope"},
        {"diagnostic": "frozen_dyadic_lower_envelope", "estimate": float(metrics["dyadic_lower_envelope_extrapolation"]), "acceptance_role": "INDEPENDENT_ENVELOPE", "interpretation": "frozen lower envelope"},
    ])
    diagnostics.to_parquet(derived / "diagnostic_summary.parquet", index=False)
    source_inventory = {
        "growth": {"path": growth_path.relative_to(root).as_posix(), "sha256": sha256_file(growth_path)},
        "g10": {"path": g10_path.relative_to(root).as_posix(), "sha256": sha256_file(g10_path)},
        "g11": {"path": g11_path.relative_to(root).as_posix(), "sha256": sha256_file(g11_path)},
        "frozen_d08": {"path": frozen_path.relative_to(root).as_posix(), "sha256": sha256_file(frozen_path)},
    }
    write_json(raw / "source_hash_inventory.json", source_inventory)
    write_json(raw / "theorem_matched_test_definition.json", spec)
    write_json(derived / "fit_diagnostics.json", {"fits": fits, "leave_one_block_out": loo})
    summary = {
        "task_id": "D-08", "run_id": run_id, "status": status,
        "theorem_contract_branch": contract["branch"], "registered_claim": contract["registered_claim"],
        "strong_regular_variation_claim_registered": False,
        "estimator_version": spec["estimator_version"], "index_range": [j_min, j_stop],
        "dyadic_block_starts": spec["dyadic_block_starts"],
        "normalized_residual_extrapolation": median,
        "upper_normalized_residual_extrapolation": upper,
        "lower_normalized_residual_extrapolation": lower,
        "residual_envelope_extrapolation_gap": envelope_gap,
        "maximum_leave_one_out_intercept_absolute": maximum_loo,
        "last_to_first_median_absolute_ratio": median_ratio,
        "last_to_first_maximum_absolute_ratio": maximum_ratio,
        "residual_identity_maximum_absolute_error": identity_error,
        "acceptance_checks": checks,
        "uncertainty_and_finite_scale": {"fits": fits, "leave_one_block_out": loo, "first_block": first.to_dict(), "last_block": last.to_dict()},
        "g10_status": g10.get("status"), "g11_status": g11.get("status"),
        "g11_frozen_extrapolation": float(metrics["g11_extrapolation"]),
        "frozen_dyadic_upper_envelope": float(metrics["dyadic_upper_envelope_extrapolation"]),
        "frozen_dyadic_lower_envelope": float(metrics["dyadic_lower_envelope_extrapolation"]),
        "fixed_ratio_c2_extrapolation": float(metrics["fixed_ratio_extrapolation"]),
        "fixed_ratio_c2_tail_center": float(metrics["fixed_ratio_tail_center"]),
        "fixed_ratio_acceptance_role_for_theorem_A": "NONE",
        "data_support_log_asymptotic_exponent_four": status == "PASS_CERTIFIED",
        "data_support_strong_regular_variation_under_doubling": False,
        "interpretation": "The data may support exponent four while the separate doubling diagnostic fails smooth regular variation.",
        "new_outcome_inspected_before_rule_freeze": False,
        "posthoc_window_selection_used": False, "adjacent_point_regression_used": False,
        "fixed_ratio_result_recalculated": False, "source_hash_inventory": source_inventory,
        "contradiction_controlled": contradicted,
        "nonpassing_rule": "FAIL_THEORY only for a convergence-controlled same-side exclusion of zero; otherwise INCONCLUSIVE",
    }
    write_json(derived / "theorem_matched_summary.json", summary)
    write_json(certificate, summary)
    context["d08_theorem_contract"] = summary
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
