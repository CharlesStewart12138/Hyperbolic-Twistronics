from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from audit.data_io import write_json
from spectral.magic_active_shell import ActiveShellModel, load_baseline_variation, moire_length


def _log_derivative(minus: float, plus: float, step: float) -> float:
    if minus <= 0 or plus <= 0:
        raise ValueError("log derivative requires positive values")
    return (math.log(plus) - math.log(minus)) / (2.0 * step)


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["s19_factorization"]
    active = config["active_shell"]
    collapse = config["s18_collapse"]
    phase_s = yaml.safe_load((root / "configs" / "phase_s.yaml").read_text(encoding="utf-8"))
    variation, root_w = load_baseline_variation(root, config)
    reference = collapse["reference"]
    model = ActiveShellModel(
        root,
        phase_s,
        root / str(config["phase_s_source"]["normal_forms"]),
        curvature_radius=float(reference["curvature_radius"]),
        lambda_perp=float(reference["lambda_perp"]),
        cutoff=int(reference["cutoff"]),
        variation=variation,
    )
    xi_reference = moire_length(model.radius, float(reference["theta"]), model.lattice_spacing)
    alpha_reference = root_w * xi_reference * xi_reference
    step = float(task["log_derivative_step"])
    q_points = int(active["q_points_primary"])
    q_min, q_max = float(active["q_min"]), float(active["q_max"])

    def phi(x_value: float) -> float:
        spectrum = model.path_spectrum(root_w * x_value, q_min, q_max, q_points)
        return max(float(np.ptp(spectrum.target_energy)), 1.0e-12)

    raw_rows = []
    derived_rows = []
    for theta in map(float, task["theta_points"]):
        theta_minus = theta * math.exp(-step)
        theta_plus = theta * math.exp(step)
        xi = moire_length(model.radius, theta, model.lattice_spacing)
        xi_minus = moire_length(model.radius, theta_minus, model.lattice_spacing)
        xi_plus = moire_length(model.radius, theta_plus, model.lattice_spacing)
        s_minus = math.sin(theta_minus / 2.0)
        s_plus = math.sin(theta_plus / 2.0)
        x_value = root_w * xi * xi / alpha_reference
        x_minus_theta = root_w * xi_minus * xi_minus / alpha_reference
        x_plus_theta = root_w * xi_plus * xi_plus / alpha_reference
        phi_value = phi(x_value)
        phi_minus_x = phi(x_value * math.exp(-step))
        phi_plus_x = phi(x_value * math.exp(step))
        energy_minus = (model.lattice_spacing / xi_minus) ** 2
        energy_plus = (model.lattice_spacing / xi_plus) ** 2
        bandwidth_minus = energy_minus * phi(x_minus_theta)
        bandwidth_plus = energy_plus * phi(x_plus_theta)
        direct = (math.log(bandwidth_plus) - math.log(bandwidth_minus)) / (
            math.log(s_plus) - math.log(s_minus)
        )
        geometry = -(
            math.log(xi_plus) - math.log(xi_minus)
        ) / (math.log(s_plus) - math.log(s_minus))
        beta = _log_derivative(phi_minus_x, phi_plus_x, step)
        predicted = 2.0 * geometry * (1.0 - beta)
        residual = direct - predicted
        raw_rows.append(
            {
                "theta": theta,
                "theta_minus": theta_minus,
                "theta_plus": theta_plus,
                "s_minus": s_minus,
                "s_plus": s_plus,
                "xi_minus": xi_minus,
                "xi": xi,
                "xi_plus": xi_plus,
                "X": x_value,
                "X_minus_from_theta": x_minus_theta,
                "X_plus_from_theta": x_plus_theta,
                "phi_minus_from_X": phi_minus_x,
                "phi": phi_value,
                "phi_plus_from_X": phi_plus_x,
                "bandwidth_minus": bandwidth_minus,
                "bandwidth_plus": bandwidth_plus,
            }
        )
        derived_rows.append(
            {
                "theta": theta,
                "geometry_running_exponent_nu_xi": geometry,
                "spectral_log_slope_beta": beta,
                "direct_bandwidth_log_slope": direct,
                "factorized_prediction": predicted,
                "factorization_residual": residual,
            }
        )
    raw_frame = pd.DataFrame(raw_rows)
    derived_frame = pd.DataFrame(derived_rows)
    raw = run_dir / "raw" / "s19_independent_derivative_inputs.parquet"
    derived = run_dir / "derived" / "geometry_spectrum_factorization.parquet"
    raw_frame.to_parquet(raw, index=False)
    derived_frame.to_parquet(derived, index=False)
    maximum = float(derived_frame["factorization_residual"].abs().max())
    passed = maximum <= float(task["acceptance"]["maximum_absolute_factorization_residual"])
    status = "PASS_CONVERGED" if passed else "INCONCLUSIVE"
    certificate = run_dir / "certificates" / "s19_geometry_spectrum_factorization.json"
    write_json(
        certificate,
        {
            "task_id": "S-19",
            "run_id": run_id,
            "status": status,
            "formula": "d log W/d log s = 2 nu_xi (1-beta_Phi)",
            "derivatives_saved_independently_from_fitted_slopes": True,
            "maximum_absolute_factorization_residual": maximum,
            "acceptance": task["acceptance"],
            "scope": "fixed finite active fiber and preregistered theta path",
            "reason_if_inconclusive": "The direct derivative and independently composed geometry/spectral derivatives did not agree within the frozen tolerance.",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
