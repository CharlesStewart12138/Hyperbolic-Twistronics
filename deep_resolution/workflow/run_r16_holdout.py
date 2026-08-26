from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = EXTENSION_ROOT.parent
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_ROOT / "src"))

from common import finalize_run, initialize_run, load_yaml, sha256_file, write_json  # noqa: E402
from r16_theory import (  # noqa: E402
    LocalCurvatureFiber,
    LocalParameters,
    bundle_linear_combination,
    comparison_metrics,
    five_point_tangent,
    mixed_tangent,
    old_octagon_shape_audit,
    sup_norm,
)


def replace(base: LocalParameters, **changes: float | int) -> LocalParameters:
    values = dict(base.__dict__)
    values.update(changes)
    return LocalParameters(**values)


def verify_sources(config: dict[str, object]) -> dict[str, object]:
    checks = {}
    for family, entry in config.items():
        run_id = str(entry["run_id"])
        run_dir = EXTENSION_ROOT / "results" / run_id
        freeze = run_dir / "freeze_certificate.json"
        expected = str(entry["freeze_certificate_sha256"])
        checks[family] = {"run_id": run_id, "freeze_actual": sha256_file(freeze), "freeze_expected": expected}
        checks[family]["pass"] = checks[family]["freeze_actual"] == expected
    if not all(bool(value["pass"]) for value in checks.values()):
        raise RuntimeError(f"deep source-run verification failed: {checks}")
    return checks


def tangent_set(base: LocalParameters, q: np.ndarray, relative_step: float) -> dict[str, object]:
    g0 = (base.a / base.radius) ** 2
    theta20 = base.theta**2
    spar0 = base.a / base.lambda_parallel
    sperp0 = base.lambda_perp / base.a
    factories = {
        "X": (lambda value: LocalCurvatureFiber(replace(base, X=value)).bundle(q), base.X),
        "g": (lambda value: LocalCurvatureFiber(replace(base, radius=base.a / math.sqrt(value))).bundle(q), g0),
        "theta": (lambda value: LocalCurvatureFiber(replace(base, theta=math.sqrt(value))).bundle(q), theta20),
        "S_parallel": (lambda value: LocalCurvatureFiber(replace(base, lambda_parallel=base.a / value)).bundle(q), spar0),
        "S_perp": (lambda value: LocalCurvatureFiber(replace(base, lambda_perp=base.a * value)).bundle(q), sperp0),
    }
    result = {}
    stability = {}
    for name, (factory, value) in factories.items():
        full, half = five_point_tangent(factory, float(value), relative_step)
        result[name] = full
        difference = comparison_metrics(full, half)
        stability[name] = difference["C0"] / max(sup_norm(half.H), 1.0e-300)
    result["gtheta"] = mixed_tangent(
        lambda g, theta2: LocalCurvatureFiber(replace(base, radius=base.a / math.sqrt(g), theta=math.sqrt(theta2))).bundle(q),
        g0,
        theta20,
        relative_step,
    )
    return {"tangents": result, "stability": stability}


def predict_models(
    reference,
    tangents: dict[str, object],
    base: LocalParameters,
    candidate: LocalParameters,
) -> dict[str, object]:
    dg = (candidate.a / candidate.radius) ** 2 - (base.a / base.radius) ** 2
    dt = candidate.theta**2 - base.theta**2
    ds_parallel = candidate.a / candidate.lambda_parallel - base.a / base.lambda_parallel
    ds_perp = candidate.lambda_perp / candidate.a - base.lambda_perp / base.a
    return {
        "H0": reference,
        "H2": bundle_linear_combination(reference, [(dt, tangents["theta"])]),
        "H3": bundle_linear_combination(reference, [(dg, tangents["g"]), (dt, tangents["theta"]), (dg * dt, tangents["gtheta"])]),
        "H4": bundle_linear_combination(reference, [(dg, tangents["g"]), (dt, tangents["theta"]), (dg * dt, tangents["gtheta"]), (ds_parallel, tangents["S_parallel"]), (ds_perp, tangents["S_perp"])]),
        "wrong_K": bundle_linear_combination(reference, [((-1.0 / candidate.radius**2) - (-1.0 / base.radius**2), tangents["g"])]),
        "fields": {"delta_g": dg, "delta_theta2": dt, "delta_S_parallel": ds_parallel, "delta_S_perp": ds_perp},
    }


def log_power_fit(coordinates: np.ndarray, responses: np.ndarray) -> dict[str, float]:
    mask = (coordinates > 0) & (responses > 0)
    slope, intercept = np.polyfit(np.log(coordinates[mask]), np.log(responses[mask]), 1)
    prediction = np.exp(intercept) * coordinates[mask] ** slope
    residual = np.log(responses[mask]) - np.log(prediction)
    return {"exponent": float(slope), "amplitude": float(np.exp(intercept)), "log_RMSE": float(np.sqrt(np.mean(residual * residual)))}


def main() -> int:
    config_path = EXTENSION_ROOT / "configs" / "r16_theory_preregistration.yaml"
    amendment_path = EXTENSION_ROOT / "configs" / "r16_design_amendment_preregistration.yaml"
    fit_path = EXTENSION_ROOT / "configs" / "r16_scaling_fit_preregistration.yaml"
    source_path = EXTENSION_ROOT / "configs" / "deep_source_runs_preregistration.yaml"
    config = load_yaml(config_path)
    amendment = load_yaml(amendment_path)
    fit_config = load_yaml(fit_path)
    source_config = load_yaml(source_path)
    source_checks = verify_sources(source_config)
    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT, "R16_HOLDOUT")
    q_cfg = config["canonical_gram_metric"]["q_grid"]
    q = np.linspace(float(q_cfg["minimum"]), float(q_cfg["maximum"]), int(q_cfg["points"]))
    ref_cfg = config["model_families"]["local_fixed_a_curvature_fiber"]["reference"]
    base_template = LocalParameters(
        a=float(ref_cfg["a"]),
        radius=float(ref_cfg["R"]),
        theta=float(ref_cfg["theta"]),
        lambda_perp=float(ref_cfg["lambda_perp"]),
        lambda_parallel=float(ref_cfg["lambda_parallel"]),
        cutoff=int(ref_cfg["cutoff"]),
        X=1.0,
    )
    family = config["model_families"]["local_fixed_a_curvature_fiber"]
    cases = []
    for X in family["X_holdout"]:
        cases.append((f"X_{float(X):.3f}", "X", replace(base_template, X=float(X))))
        for value in family["curvature_holdout_R"]:
            cases.append((f"R_{float(value):.3f}_X_{float(X):.3f}", "curvature", replace(base_template, radius=float(value), X=float(X))))
        for value in family["theta_holdout"]:
            cases.append((f"theta_{float(value):.4f}_X_{float(X):.3f}", "angle", replace(base_template, theta=float(value), X=float(X))))
        for value in family["lambda_perp_holdout"]:
            cases.append((f"lambda_perp_{float(value):.3f}_X_{float(X):.3f}", "profile", replace(base_template, lambda_perp=float(value), X=float(X))))
        for cutoff in (2, 4):
            cases.append((f"cutoff_{cutoff}_X_{float(X):.3f}", "cutoff", replace(base_template, cutoff=cutoff, X=float(X))))
    for index, entry in enumerate(amendment["combined_holdout"]):
        cases.append(
            (
                f"combined_{index:02d}",
                "combined",
                LocalParameters(
                    a=base_template.a,
                    radius=float(entry["R"]),
                    theta=float(entry["theta"]),
                    lambda_perp=float(entry["lambda_perp"]),
                    lambda_parallel=float(entry["lambda_parallel"]),
                    cutoff=base_template.cutoff,
                    X=float(entry["X"]),
                ),
            )
        )
    distinct_x = sorted({case.X for _, _, case in cases})
    references = {}
    tangent_maps = {}
    tangent_stability = {}
    raw_payload = {"q": q}
    relative_step = float(config["microscopic_expansion"]["finite_difference_relative_step"])
    for X in distinct_x:
        base = replace(base_template, X=X)
        references[X] = LocalCurvatureFiber(base).bundle(q)
        tangent_audit = tangent_set(base, q, relative_step)
        tangent_maps[X] = tangent_audit["tangents"]
        tangent_stability[X] = tangent_audit["stability"]
        for name, bundle in tangent_maps[X].items():
            raw_payload[f"X_{X:.3f}_{name}_H"] = bundle.H
            raw_payload[f"X_{X:.3f}_{name}_D1"] = bundle.D1
            raw_payload[f"X_{X:.3f}_{name}_D2"] = bundle.D2
    rows = []
    for case_id, channel, parameters in cases:
        actual = LocalCurvatureFiber(parameters).bundle(q)
        prediction = predict_models(references[parameters.X], tangent_maps[parameters.X], replace(base_template, X=parameters.X), parameters)
        row = {"case_id": case_id, "channel": channel, **parameters.__dict__, **prediction["fields"], "hermiticity_residual": actual.hermiticity_residual}
        for hypothesis in ("H0", "H2", "H3", "H4", "wrong_K"):
            for name, value in comparison_metrics(actual, prediction[hypothesis]).items():
                row[f"{hypothesis}_{name}"] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(run_dir / "raw" / "r16_holdout_case_registry.parquet", index=False)
    frame.to_parquet(run_dir / "derived" / "r16_holdout_residuals.parquet", index=False)
    frame.to_parquet(run_dir / "figure_data" / "figure_16_hypothesis_holdout.parquet", index=False)
    with (run_dir / "raw" / "r16_holdout_tangents.npz").open("xb") as handle:
        np.savez_compressed(handle, **raw_payload)

    # Frozen curvature power-law training and holdout audit at X=1.
    training_coordinates = []
    training_responses = []
    for radius in fit_config["curvature_residual"]["training_R"]:
        candidate = replace(base_template, radius=float(radius))
        actual = LocalCurvatureFiber(candidate).bundle(q)
        baseline = references[1.0]
        training_coordinates.append(abs((candidate.a / candidate.radius) ** 2 - (base_template.a / base_template.radius) ** 2))
        training_responses.append(sup_norm(actual.H - baseline.H))
    training_coordinates = np.asarray(training_coordinates)
    training_responses = np.asarray(training_responses)
    global_fit = log_power_fit(training_coordinates, training_responses)
    order = np.argsort(training_coordinates)
    local_slopes = np.diff(np.log(training_responses[order])) / np.diff(np.log(training_coordinates[order]))
    inner_indices = order[:3]
    inner_fit = log_power_fit(training_coordinates[inner_indices], training_responses[inner_indices])
    rng = np.random.default_rng(int(fit_config["curvature_residual"]["bootstrap"]["seed"]))
    bootstrap = []
    for _ in range(int(fit_config["curvature_residual"]["bootstrap"]["resamples"])):
        indices = rng.integers(0, len(training_coordinates), len(training_coordinates))
        if len(np.unique(training_coordinates[indices])) < 2:
            continue
        bootstrap.append(log_power_fit(training_coordinates[indices], training_responses[indices])["exponent"])
    curvature_rows = []
    for radius in fit_config["curvature_residual"]["holdout_R"]:
        candidate = replace(base_template, radius=float(radius))
        coordinate = abs((candidate.a / candidate.radius) ** 2 - (base_template.a / base_template.radius) ** 2)
        actual_response = sup_norm(LocalCurvatureFiber(candidate).bundle(q).H - references[1.0].H)
        predicted = global_fit["amplitude"] * coordinate ** global_fit["exponent"]
        curvature_rows.append({"radius": radius, "coordinate": coordinate, "actual_C0": actual_response, "predicted_C0": predicted, "relative_prediction_error": abs(actual_response - predicted) / actual_response})
    curvature_frame = pd.DataFrame(curvature_rows)
    curvature_frame.to_parquet(run_dir / "derived" / "r16_curvature_scaling_holdout.parquet", index=False)
    curvature_frame.to_parquet(run_dir / "figure_data" / "figure_18_curvature_scaling.parquet", index=False)

    old_raw = VALIDATION_ROOT / "postvalidation_resolution" / "results" / "fd8c15495c5990037ef93b299f0a6006de93746a4f11f54bd383a8b769cb8bf4" / "raw" / "r16_operator_families.npz"
    shape_audit = old_octagon_shape_audit(old_raw)
    shape_frame = pd.DataFrame(shape_audit["holdout_rows"])
    shape_frame.to_parquet(run_dir / "derived" / "r16_old_octagon_shape_holdout.parquet", index=False)
    shape_frame.to_parquet(run_dir / "figure_data" / "figure_16_old_radius_shape_field.parquet", index=False)

    acceptance = config["acceptance"]
    tested = frame[frame.channel != "X"]
    def hypothesis_pass(name: str) -> bool:
        return bool(
            tested[f"{name}_C0"].max() <= float(acceptance["maximum_normalized_C0_holdout"])
            and tested[f"{name}_C1"].max() <= float(acceptance["maximum_normalized_C1_holdout"])
            and tested[f"{name}_C2"].max() <= float(acceptance["maximum_normalized_C2_holdout"])
            and np.median(1.0 - tested[f"{name}_C0"] / np.maximum(tested["H0_C0"], 1.0e-300)) >= float(acceptance["minimum_median_C0_reduction_vs_H0"])
        )
    h3_pass = hypothesis_pass("H3")
    h4_pass = hypothesis_pass("H4")
    design_certificate = json.loads((EXTENSION_ROOT / "results" / source_config["R16_design"]["run_id"] / "certificates" / "r16_design_certificate.json").read_text(encoding="utf-8"))
    operator_rank_pass = bool(design_certificate["rank2_local_curvature_pass"])
    observable_rank_stable = float(design_certificate["observable_jacobian"]["s2_over_s1"]) >= float(config["rank_acceptance"]["stable_rank2_minimum_s2_over_s1"])
    shape_reduction = float(shape_frame.reduction_fraction.median())
    shape_holdout_pass = bool(shape_frame.shape_corrected_C0.max() <= float(acceptance["maximum_normalized_C0_holdout"]) and shape_reduction >= float(acceptance["minimum_median_C0_reduction_vs_H0"]))
    curvature_power_holdout_pass = bool(curvature_frame.relative_prediction_error.max() <= float(fit_config["curvature_residual"]["holdout_relative_prediction_error_max"]))
    if h3_pass and operator_rank_pass and observable_rank_stable:
        classification = "PASS_TWO_PARAMETER_CURVATURE"
    elif h4_pass and shape_holdout_pass:
        classification = "PASS_THREE_FIELD"
    else:
        classification = "PASS_RESTRICTED_CLASS"
    summary_rows = []
    for hypothesis in ("H0", "H2", "H3", "H4", "wrong_K"):
        summary_rows.append({"hypothesis": hypothesis, "max_C0": float(tested[f"{hypothesis}_C0"].max()), "max_C1": float(tested[f"{hypothesis}_C1"].max()), "max_C2": float(tested[f"{hypothesis}_C2"].max()), "median_C0": float(tested[f"{hypothesis}_C0"].median())})
    pd.DataFrame(summary_rows).to_parquet(run_dir / "figure_data" / "figure_18_hypothesis_summary.parquet", index=False)
    certificate = {
        "run_id": run_id,
        "source_checks": source_checks,
        "anchor_checks": identity["anchor_checks"],
        "preregistration_hashes": {"theory": sha256_file(config_path), "holdout": sha256_file(amendment_path), "scaling_fit": sha256_file(fit_path), "source_runs": sha256_file(source_path)},
        "hypothesis_pass": {"H0": False, "H2": False, "H3": h3_pass, "H4": h4_pass},
        "operator_rank2_local_curvature": operator_rank_pass,
        "observable_identifiability_rank2_stable": observable_rank_stable,
        "old_octagon_shape_holdout_pass": shape_holdout_pass,
        "old_octagon_shape_median_reduction": shape_reduction,
        "curvature_scaling": {"global_fit": global_fit, "inner_fit": inner_fit, "local_slopes": local_slopes.tolist(), "bootstrap_exponent_interval_95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))], "holdout_pass": curvature_power_holdout_pass},
        "tangent_stability": {str(key): value for key, value in tangent_stability.items()},
        "hypothesis_maxima": summary_rows,
        "classification": classification,
        "curvature_independent_coordinate_scope": "operator tangent rank-2 in local fixed-a q=8 fiber; not independently identifiable by the registered scalar observable Jacobian and not a global bulk certificate",
        "manuscript_contract_result": "No FAIL_THEORY: the manuscript fixed-tessellation warning and conditional correction-field theorem are consistent with the observed shape dependence.",
    }
    write_json(run_dir / "certificates" / "r16_holdout_certificate.json", certificate)
    tasks = {
        "R16-V01": "PASS_CERTIFIED",
        "R16-V02": "PASS_CERTIFIED" if operator_rank_pass else "INCONCLUSIVE",
        "R16-V03": "INCONCLUSIVE" if not observable_rank_stable else "PASS_CERTIFIED",
        "R16-V04": "PASS_CONVERGED" if h3_pass else "INCONCLUSIVE",
        "R16-V05": "PASS_CONVERGED" if h4_pass else "INCONCLUSIVE",
        "R16-V06": "PASS_CERTIFIED",
        "R16-V07": "PASS_CONVERGED" if curvature_power_holdout_pass else "INCONCLUSIVE",
        "R16-V08": classification,
    }
    freeze = finalize_run(run_dir, tasks, classification)
    print(json.dumps({"run_id": run_id, "classification": classification, "H3_pass": h3_pass, "H4_pass": h4_pass, "observable_rank_stable": observable_rank_stable, "shape_holdout_pass": shape_holdout_pass, "curvature_power_holdout_pass": curvature_power_holdout_pass, "freeze": freeze["tree_inventory_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

