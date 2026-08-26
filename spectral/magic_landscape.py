from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from audit.data_io import write_json, write_zarr
from spectral.magic_active_shell import ActiveShellModel, load_baseline_variation, moire_length


def _registry_length(curvature: float, theta: float, lattice_spacing: float) -> float:
    if abs(curvature) < 1.0e-18:
        return lattice_spacing / (2.0 * math.sin(theta / 2.0))
    if curvature > 0:
        raise ValueError("only K <= 0 is preregistered")
    return moire_length(1.0 / math.sqrt(-curvature), theta, lattice_spacing)


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["s20_landscape"]
    active = config["active_shell"]
    reference = config["s18_collapse"]["reference"]
    phase_s = yaml.safe_load((root / "configs" / "phase_s.yaml").read_text(encoding="utf-8"))
    variation, root_w = load_baseline_variation(root, config)
    model = ActiveShellModel(
        root,
        phase_s,
        root / str(config["phase_s_source"]["normal_forms"]),
        curvature_radius=float(reference["curvature_radius"]),
        lambda_perp=float(reference["lambda_perp"]),
        cutoff=int(reference["cutoff"]),
        variation=variation,
    )
    lattice_spacing = model.lattice_spacing
    reference_curvature = -1.0 / float(reference["curvature_radius"]) ** 2
    xi_reference = _registry_length(reference_curvature, float(reference["theta"]), lattice_spacing)
    alpha_reference = root_w * xi_reference * xi_reference
    k_values = np.asarray(task["K_values"], dtype=float)
    theta_values = np.asarray(task["theta_values"], dtype=float)
    w_values = np.asarray(task["w_over_t_values"], dtype=float)
    shape = (len(k_values), len(theta_values), len(w_values))
    arrays = {
        name: np.empty(shape, dtype=float)
        for name in ("score_M", "W", "Delta", "Omega_max", "rho_coh", "C_coh", "X", "xi", "energy_scale", "tracking_overlap")
    }
    cache: dict[float, dict[str, float]] = {}
    rows = []
    eta = float(active["lorentzian_eta"])
    for ik, curvature in enumerate(k_values):
        for it, theta in enumerate(theta_values):
            xi = _registry_length(float(curvature), float(theta), lattice_spacing)
            energy_scale = (lattice_spacing / xi) ** 2
            for iw, w_value in enumerate(w_values):
                x_value = float(w_value * xi * xi / alpha_reference)
                cache_key = round(x_value, 12)
                if cache_key not in cache:
                    spectrum = model.path_spectrum(
                        root_w * x_value,
                        float(active["q_min"]),
                        float(active["q_max"]),
                        int(active["q_points_primary"]),
                    )
                    cache[cache_key] = model.metrics(spectrum, eta)
                scaled = cache[cache_key]
                bandwidth = energy_scale * scaled["bandwidth_W"]
                gap = energy_scale * scaled["gap_Delta"]
                omega = energy_scale * scaled["Omega_max"]
                rho = scaled["rho_coh_max"] / max(energy_scale, 1.0e-15)
                coherence = scaled["C_coh"]
                isolation = gap / (gap + bandwidth + 1.0e-15)
                dispersion = gap / (gap + omega + 1.0e-15)
                dos_factor = rho / (1.0 + rho)
                score = isolation * dispersion * dos_factor * coherence
                values = {
                    "score_M": score,
                    "W": bandwidth,
                    "Delta": gap,
                    "Omega_max": omega,
                    "rho_coh": rho,
                    "C_coh": coherence,
                    "X": x_value,
                    "xi": xi,
                    "energy_scale": energy_scale,
                    "tracking_overlap": scaled["minimum_tracking_overlap"],
                }
                for name, value in values.items():
                    arrays[name][ik, it, iw] = float(value)
                rows.append({"K": curvature, "theta": theta, "w_over_t": w_value, **values})
    raw = run_dir / "raw" / "magic_landscape.zarr"
    write_zarr(
        raw,
        {"K": k_values, "theta": theta_values, "w_over_t": w_values, **arrays},
        {
            "task_id": "S-20",
            "run_id": run_id,
            "score_definition": task["score_definition"],
            "scientific_results_computed_outside_plotting": True,
            "bulk_claim_permitted": False,
        },
    )
    derived = run_dir / "derived" / "magic_landscape_flat.parquet"
    frame = pd.DataFrame(rows)
    frame.to_parquet(derived, index=False)
    score_min = float(arrays["score_M"].min())
    score_max = float(arrays["score_M"].max())
    finite = bool(all(np.isfinite(values).all() for values in arrays.values()))
    expected_size = int(np.prod(shape))
    low, high = map(float, task["acceptance"]["score_range"])
    passed = len(frame) == expected_size and finite and score_min >= low - 1.0e-12 and score_max <= high + 1.0e-12
    status = "PASS_CONVERGED" if passed else "FAIL_IMPLEMENTATION"
    certificate = run_dir / "certificates" / "s20_magic_landscape.json"
    write_json(
        certificate,
        {
            "task_id": "S-20",
            "run_id": run_id,
            "status": status,
            "cube_shape": list(shape),
            "expected_points": expected_size,
            "stored_points": len(frame),
            "finite_values": finite,
            "score_range_observed": [score_min, score_max],
            "score_factors": ["Delta/(Delta+W)", "Delta/(Delta+Omega_max)", "rho_coh/(1+rho_coh)", "C_coh"],
            "same_target_rule_as_S17": True,
            "scope": "fixed finite active-fiber landscape over the frozen K, theta, w/t box",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
