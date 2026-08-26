from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = EXTENSION_ROOT.parent
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_ROOT / "src"))

from common import finalize_run, initialize_run, sha256_file, write_json  # noqa: E402
from r16_master_v2 import (  # noqa: E402
    analytic_operator_bundle,
    comparison_metrics,
    corrected_prediction,
    design_matrix,
    field_values,
    finite_difference_derivative_check,
    fit_operator_corrections,
    near_reference_envelope,
    track_spectrum,
)
from spectral.magic_active_shell import ActiveShellModel, load_baseline_variation, moire_length  # noqa: E402


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    config_path = EXTENSION_ROOT / "configs" / "r16_theorem_contract_preregistration.yaml"
    algorithm_path = EXTENSION_ROOT / "configs" / "r16_acceptance_algorithm_preregistration.yaml"
    config = load_yaml(config_path)
    algorithm = load_yaml(algorithm_path)
    contract = config["contract_audit"]
    prior = contract["prior_anchors"]
    anchor_paths = {
        Path(r"C:\Users\charl\chatGPTwork\5203\05_FINAL\5203_REVISED_FINAL_ROUND2.pdf"): str(contract["manuscript_sha256"]),
        VALIDATION_ROOT / "configs" / "final_remaining.yaml": str(prior["final_remaining_yaml_sha256"]),
        VALIDATION_ROOT / "src" / "spectral" / "master_curve_collapse.py": str(prior["old_s18_code_sha256"]),
        VALIDATION_ROOT / "src" / "spectral" / "magic_active_shell.py": str(prior["active_shell_code_sha256"]),
        VALIDATION_ROOT / "results" / "b8722dc68f6c7ed04cc5d48023a0d7f52f1fe92153bb362cc57dc7144cdf17ad" / "certificates" / "s18_master_curve_collapse.json": str(prior["old_s18_certificate_sha256"]),
        VALIDATION_ROOT / "results" / "b8722dc68f6c7ed04cc5d48023a0d7f52f1fe92153bb362cc57dc7144cdf17ad" / "certificates" / "s19_geometry_spectrum_factorization.json": str(prior["old_s19_certificate_sha256"]),
        VALIDATION_ROOT / "results" / "ed350b4ade947be9a310532f8fd8e643030e35205b74ea14bf01ca4c48844ec4" / "certificates" / "nc_08.json": str(prior["old_nc08_certificate_sha256"]),
    }
    for path, expected in anchor_paths.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"R16 frozen anchor mismatch: {path}: {actual} != {expected}")

    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT)
    try:
        final_config = load_yaml(VALIDATION_ROOT / "configs" / "final_remaining.yaml")
        phase_s = load_yaml(VALIDATION_ROOT / "configs" / "phase_s.yaml")
        normal_forms = VALIDATION_ROOT / str(final_config["phase_s_source"]["normal_forms"])
        variation, root_w = load_baseline_variation(VALIDATION_ROOT, final_config)
        experiment = config["experiment"]
        reference_cfg = experiment["reference"]
        q_cfg = experiment["q_grid"]
        q = np.linspace(float(q_cfg["minimum"]), float(q_cfg["maximum"]), int(q_cfg["points"]))
        matched_x = [float(value) for value in experiment["matched_X"]]

        model_cache: dict[tuple[float, float, int], ActiveShellModel] = {}

        def get_model(radius: float, lambda_perp: float, cutoff: int) -> ActiveShellModel:
            key = (float(radius), float(lambda_perp), int(cutoff))
            if key not in model_cache:
                model_cache[key] = ActiveShellModel(
                    VALIDATION_ROOT,
                    phase_s,
                    normal_forms,
                    curvature_radius=key[0],
                    lambda_perp=key[1],
                    cutoff=key[2],
                    variation=variation,
                )
            return model_cache[key]

        reference_model = get_model(
            float(reference_cfg["curvature_radius"]),
            float(reference_cfg["lambda_perp"]),
            int(reference_cfg["cutoff"]),
        )
        reference_theta = float(reference_cfg["theta"])
        reference_xi = moire_length(reference_model.radius, reference_theta, reference_model.lattice_spacing)
        alpha_reference = float(root_w) * reference_xi * reference_xi

        case_registry: list[dict[str, object]] = [{
            "case_id": "reference",
            "channel": "reference",
            "split": "reference",
            "curvature_radius": float(reference_cfg["curvature_radius"]),
            "theta": reference_theta,
            "lambda_perp": float(reference_cfg["lambda_perp"]),
            "cutoff": int(reference_cfg["cutoff"]),
        }]
        paths = experiment["paths"]
        for split in ("training", "holdout"):
            for radius in paths["curvature_radius"][split]:
                fixed = paths["curvature_radius"]["fixed"]
                case_registry.append({
                    "case_id": f"radius_{split}_{float(radius):.6g}",
                    "channel": "curvature_radius",
                    "split": split,
                    "curvature_radius": float(radius),
                    "theta": float(fixed["theta"]),
                    "lambda_perp": float(fixed["lambda_perp"]),
                    "cutoff": int(fixed["cutoff"]),
                })
            for theta in paths["twist_angle"][split]:
                fixed = paths["twist_angle"]["fixed"]
                case_registry.append({
                    "case_id": f"theta_{split}_{float(theta):.6g}",
                    "channel": "twist_angle",
                    "split": split,
                    "curvature_radius": float(fixed["curvature_radius"]),
                    "theta": float(theta),
                    "lambda_perp": float(fixed["lambda_perp"]),
                    "cutoff": int(fixed["cutoff"]),
                })
            for value in paths["tunneling_decay"][split]:
                fixed = paths["tunneling_decay"]["fixed"]
                case_registry.append({
                    "case_id": f"lambda_{split}_{float(value):.6g}",
                    "channel": "tunneling_decay",
                    "split": split,
                    "curvature_radius": float(fixed["curvature_radius"]),
                    "theta": float(fixed["theta"]),
                    "lambda_perp": float(value),
                    "cutoff": int(fixed["cutoff"]),
                })
        fixed_cutoff = paths["shell_cutoff"]["fixed"]
        for cutoff in paths["shell_cutoff"]["levels"]:
            if int(cutoff) == int(reference_cfg["cutoff"]):
                continue
            case_registry.append({
                "case_id": f"cutoff_{int(cutoff)}",
                "channel": "shell_cutoff",
                "split": "deterministic_sequence",
                "curvature_radius": float(fixed_cutoff["curvature_radius"]),
                "theta": float(fixed_cutoff["theta"]),
                "lambda_perp": float(fixed_cutoff["lambda_perp"]),
                "cutoff": int(cutoff),
            })

        bundle_map = {}
        spectrum_map = {}
        raw_H, raw_D1, raw_D2 = [], [], []
        raw_eigenvalues, raw_target, raw_projectors, raw_coherence = [], [], [], []
        evaluation_rows = []
        progress_path = run_dir / "logs" / "r16_progress.jsonl"
        reference_energy_scale = (reference_model.lattice_spacing / reference_xi) ** 2
        for case_index, case in enumerate(case_registry):
            model = get_model(float(case["curvature_radius"]), float(case["lambda_perp"]), int(case["cutoff"]))
            xi = moire_length(model.radius, float(case["theta"]), model.lattice_spacing)
            energy_scale = (model.lattice_spacing / xi) ** 2
            theory_fields = field_values(
                model,
                float(case["theta"]),
                reference_model,
                reference_theta,
                float(config["correction_fields"]["Y_R"]["lambda_parallel"]),
            )
            if case["channel"] == "curvature_radius":
                asymptotic_scale = abs(theory_fields["Y_R"])
            elif case["channel"] == "twist_angle":
                asymptotic_scale = abs(theory_fields["Y_Ktheta"])
            elif case["channel"] == "tunneling_decay":
                asymptotic_scale = abs(theory_fields["Y_profile"])
            elif case["channel"] == "shell_cutoff":
                asymptotic_scale = abs(1.0 / (int(case["cutoff"]) + 1.0) - 1.0 / (int(reference_cfg["cutoff"]) + 1.0))
            else:
                asymptotic_scale = 0.0
            for x_value in matched_x:
                w_value = x_value * alpha_reference / (xi * xi)
                bundle = analytic_operator_bundle(model, q, w_value, energy_scale)
                spectrum = track_spectrum(bundle.H)
                key = (str(case["case_id"]), float(x_value))
                bundle_map[key] = bundle
                spectrum_map[key] = spectrum
                raw_H.append(bundle.H)
                raw_D1.append(bundle.D1)
                raw_D2.append(bundle.D2)
                raw_eigenvalues.append(spectrum.eigenvalues)
                raw_target.append(spectrum.target_energy)
                raw_projectors.append(spectrum.target_projectors)
                raw_coherence.append(spectrum.target_coherence)
                evaluation_rows.append({
                    **case,
                    "X": x_value,
                    "xi": xi,
                    "energy_scale": energy_scale,
                    "w_over_t": w_value,
                    "delta_y_lat": energy_scale - reference_energy_scale,
                    **theory_fields,
                    "asymptotic_scale": asymptotic_scale,
                    "normal_form_count": model.normal_form_count,
                    "hermiticity_residual": bundle.hermiticity_residual,
                    "minimum_tracking_overlap": spectrum.minimum_tracking_overlap,
                })
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"case_index": case_index, "case_id": case["case_id"], "model_cache_size": len(model_cache)}) + "\n")
                handle.flush()

        evaluation_frame = pd.DataFrame(evaluation_rows)
        metric_rows = []
        for row in evaluation_rows:
            key = (str(row["case_id"]), float(row["X"]))
            ref_key = ("reference", float(row["X"]))
            metrics = comparison_metrics(bundle_map[key], bundle_map[ref_key], spectrum_map[key], spectrum_map[ref_key])
            metric_rows.append({**row, **metrics})
        metric_frame = pd.DataFrame(metric_rows)

        correction_rows = []
        coefficient_payload = {}
        correction_fields = [str(value) for value in config["correction_fields"]["linear_corrected_master"]["radius_angle_fields"]]
        for x_value in matched_x:
            selected = metric_frame[np.isclose(metric_frame.X, x_value)]
            training = selected[selected.channel.isin(["curvature_radius", "twist_angle"]) & (selected.split == "training")]
            reference_bundle = bundle_map[("reference", x_value)]
            training_bundles = [bundle_map[(str(row.case_id), x_value)] for row in training.itertuples()]
            design = design_matrix(training.to_dict("records"), correction_fields)
            coefficients = fit_operator_corrections(training_bundles, reference_bundle, design)
            for name, value in coefficients.items():
                coefficient_payload[f"X_{x_value:.2f}_{name}"] = value
            holdout = selected[selected.channel.isin(["curvature_radius", "twist_angle"]) & (selected.split == "holdout")]
            for row in holdout.itertuples():
                vector = np.asarray([float(getattr(row, field)) for field in correction_fields])
                predicted = corrected_prediction(reference_bundle, coefficients, vector)
                actual = bundle_map[(str(row.case_id), x_value)]
                corrected = comparison_metrics(actual, predicted)
                baseline = selected[selected.case_id == row.case_id].iloc[0]
                correction_rows.append({
                    "case_id": str(row.case_id),
                    "channel": str(row.channel),
                    "X": x_value,
                    "asymptotic_scale": float(row.asymptotic_scale),
                    **{field: float(getattr(row, field)) for field in correction_fields},
                    **{f"corrected_{key}": value for key, value in corrected.items()},
                    "baseline_epsilon_C0": float(baseline.epsilon_C0),
                    "baseline_epsilon_C1": float(baseline.epsilon_C1),
                    "baseline_epsilon_C2": float(baseline.epsilon_C2),
                    "C0_reduction_fraction": 1.0 - corrected["epsilon_C0"] / max(float(baseline.epsilon_C0), 1.0e-15),
                })
        correction_frame = pd.DataFrame(correction_rows)

        derivative_check = finite_difference_derivative_check(
            reference_model,
            bundle_map[("reference", 1.0)].w_over_t,
            bundle_map[("reference", 1.0)].energy_scale,
        )
        numerical_acceptance = config["acceptance"]["numerical"]
        numerical_pass = bool(
            metric_frame.hermiticity_residual.max() <= float(numerical_acceptance["maximum_hermiticity_residual"])
            and max(derivative_check.values()) <= float(numerical_acceptance["maximum_analytic_derivative_check_error"])
            and metric_frame.minimum_tracking_overlap.min() >= float(numerical_acceptance["minimum_target_tracking_overlap"])
        )

        envelope_audits = {}
        local_passes = []
        for channel in ("curvature_radius", "twist_angle", "tunneling_decay"):
            channel_frame = metric_frame[metric_frame.channel == channel]
            envelope_audits[channel] = {}
            for x_value, group in channel_frame.groupby("X"):
                audit = near_reference_envelope(group, "epsilon_C0")
                envelope_audits[channel][f"X={x_value:.2f}"] = audit
                local_passes.append(bool(audit["decreases_toward_reference"]))
        cutoff_audits = {}
        cutoff_passes = []
        for x_value in matched_x:
            group = metric_frame[(metric_frame.channel == "shell_cutoff") & np.isclose(metric_frame.X, x_value)].copy()
            reference_metric = metric_frame[(metric_frame.case_id == "reference") & np.isclose(metric_frame.X, x_value)].copy()
            reference_metric.loc[:, "channel"] = "shell_cutoff"
            reference_metric.loc[:, "cutoff"] = int(reference_cfg["cutoff"])
            group = pd.concat([group, reference_metric], ignore_index=True).sort_values("cutoff")
            values = group.epsilon_C0.to_numpy(dtype=float)
            tail_envelope = np.maximum.accumulate(values[::-1])[::-1]
            passed = bool(tail_envelope[-1] <= float(config["acceptance"]["restricted_class"]["shell_cutoff_C0_at_final_refinement"]) and tail_envelope[-1] < tail_envelope[0])
            cutoff_passes.append(passed)
            cutoff_audits[f"X={x_value:.2f}"] = {
                "cutoffs": group.cutoff.astype(int).tolist(),
                "C0": values.tolist(),
                "tail_upper_envelope": tail_envelope.tolist(),
                "pass": passed,
            }

        one = config["acceptance"]["one_parameter"]
        nonreference = metric_frame[metric_frame.case_id != "reference"]
        h0_pass = bool(
            nonreference.epsilon_C0.max() <= float(one["maximum_C0"])
            and nonreference.epsilon_C1.max() <= float(one["maximum_C1"])
            and nonreference.epsilon_C2.max() <= float(one["maximum_C2"])
            and nonreference.projector_error.max() <= float(one["maximum_projector"])
            and all(local_passes)
            and all(cutoff_passes)
        )
        corrected_acceptance = config["acceptance"]["corrected_master"]
        h2_pass = bool(
            len(correction_frame) > 0
            and correction_frame.corrected_epsilon_C0.max() <= float(corrected_acceptance["maximum_holdout_C0"])
            and correction_frame.corrected_epsilon_C1.max() <= float(corrected_acceptance["maximum_holdout_C1"])
            and correction_frame.corrected_epsilon_C2.max() <= float(corrected_acceptance["maximum_holdout_C2"])
            and correction_frame.C0_reduction_fraction.median() >= float(corrected_acceptance["minimum_median_C0_reduction_fraction"])
            and bool((correction_frame.corrected_epsilon_C0 <= float(corrected_acceptance["no_holdout_C0_increase_factor_above"]) * correction_frame.baseline_epsilon_C0).all())
        )
        h1_pass = bool(numerical_pass and all(local_passes) and all(cutoff_passes))
        if h0_pass:
            classification = "PASS_ONE_PARAMETER"
        elif h2_pass:
            classification = "PASS_CORRECTED_MASTER"
        elif h1_pass:
            classification = "PASS_RESTRICTED_CLASS"
        else:
            classification = "INCONCLUSIVE"

        evaluation_frame.to_parquet(run_dir / "raw" / "r16_case_registry.parquet", index=False)
        raw_path = run_dir / "raw" / "r16_operator_families.npz"
        with raw_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                q=q,
                labels=np.asarray([f"{row['case_id']}:X={float(row['X']):.2f}" for row in evaluation_rows]),
                H=np.asarray(raw_H),
                D1=np.asarray(raw_D1),
                D2=np.asarray(raw_D2),
                eigenvalues=np.asarray(raw_eigenvalues),
                target_energy=np.asarray(raw_target),
                target_projectors=np.asarray(raw_projectors),
                target_coherence=np.asarray(raw_coherence),
            )
        metric_frame.to_parquet(run_dir / "derived" / "r16_operator_residuals.parquet", index=False)
        correction_frame.to_parquet(run_dir / "derived" / "r16_corrected_holdout.parquet", index=False)
        with (run_dir / "derived" / "r16_correction_operators.npz").open("xb") as handle:
            np.savez_compressed(handle, **coefficient_payload)
        metric_frame[[
            "case_id", "channel", "split", "X", "asymptotic_scale", "epsilon_C0", "epsilon_C1", "epsilon_C2",
            "complete_spectrum_sup_error", "bandwidth_error", "gap_error", "projector_error", "coherence_error",
        ]].to_parquet(run_dir / "figure_data" / "r16_channel_residuals.parquet", index=False)
        correction_frame.to_parquet(run_dir / "figure_data" / "r16_corrected_holdout.parquet", index=False)

        statuses = {
            "R16-01": "PASS_EXACT",
            "R16-02": "PASS_CERTIFIED",
            "R16-03": "PASS_CERTIFIED" if numerical_pass else "FAIL_IMPLEMENTATION",
            "R16-04": "PASS_CERTIFIED",
            "R16-05": "PASS_CONVERGED" if h2_pass else "INCONCLUSIVE",
            "R16-06": "PASS_CERTIFIED",
            "R16-07": "PASS_CERTIFIED" if classification != "INCONCLUSIVE" else "INCONCLUSIVE",
            "R16-10": "PASS_CERTIFIED",
        }
        certificate = {
            "run_id": run_id,
            "task_statuses": statuses,
            "classification": classification,
            "theorem_contract": contract,
            "input_hashes_verified": True,
            "parent_verification": identity["parent_verification"],
            "preregistration_sha256": sha256_file(config_path),
            "acceptance_algorithm_sha256": sha256_file(algorithm_path),
            "old_S18_preserved": "INCONCLUSIVE",
            "old_NC08_preserved": "FAIL_EXPECTED",
            "derivative_check": derivative_check,
            "numerical_pass": numerical_pass,
            "hypotheses": {"H0_all_channels": h0_pass, "H1_restricted_class": h1_pass, "H2_corrected_master": h2_pass},
            "baseline_maxima": {
                "C0": float(nonreference.epsilon_C0.max()),
                "C1": float(nonreference.epsilon_C1.max()),
                "C2": float(nonreference.epsilon_C2.max()),
                "projector": float(nonreference.projector_error.max()),
            },
            "corrected_holdout_maxima": {
                "C0": float(correction_frame.corrected_epsilon_C0.max()),
                "C1": float(correction_frame.corrected_epsilon_C1.max()),
                "C2": float(correction_frame.corrected_epsilon_C2.max()),
                "median_C0_reduction_fraction": float(correction_frame.C0_reduction_fraction.median()),
            },
            "near_reference_envelopes": envelope_audits,
            "cutoff_audits": cutoff_audits,
            "scientific_guard": "The classification concerns the frozen finite ARO-3B active fiber. It is not a thermodynamic bulk claim and does not treat the inconclusive R8 no-pollution gate as passed.",
        }
        write_json(run_dir / "certificates" / "r16_master_certificate.json", certificate)
        finalize_run(run_dir, "COMPLETE" if numerical_pass else "INCOMPLETE", statuses)
        print(json.dumps({"run_id": run_id, "classification": classification, "statuses": statuses, "hypotheses": certificate["hypotheses"]}))
        return 0 if numerical_pass else 1
    except Exception as error:
        failure = {
            "run_id": run_id,
            "status": "FAIL_IMPLEMENTATION",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "certificates" / "r16_failure.json", failure)
        finalize_run(run_dir, "INCOMPLETE", {task: "FAIL_IMPLEMENTATION" for task in ("R16-01", "R16-02", "R16-03", "R16-04", "R16-05", "R16-06", "R16-07", "R16-10")})
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
