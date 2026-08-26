from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dos.common import finish, gaussian_density, retained_group, task_paths


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw, derived, certificate = task_paths(run_dir, "d05_coherence_weighted_dos")
    blocks = context["blocks"]
    tower_id = "congruence_p7_r2"
    level = int(blocks[blocks.tower_id == tower_id].level.max())
    subset = retained_group(blocks, tower_id, level)
    adjacency = np.asarray(subset.adjacency_eigenvalue, dtype=float) / 8.0
    multiplicity = np.asarray(subset.regular_multiplicity, dtype=float)
    minus = adjacency - 1.0
    plus = adjacency + 1.0
    atoms = pd.DataFrame({
        "tower_id": tower_id, "level": level,
        "adjacency_energy": adjacency,
        "bilayer_minus_energy": minus,
        "bilayer_plus_energy": plus,
        "regular_multiplicity": multiplicity.astype(int),
        "layer_even_coherence_minus": np.zeros_like(adjacency),
        "layer_even_coherence_plus": np.ones_like(adjacency),
    })
    atoms.to_parquet(raw / "coherence_weighted_atoms.parquet", index=False)
    eta = float(config["coherence_dos"]["fixed_broadening"])
    grid = np.linspace(float(np.min(minus)) - 0.25, float(np.max(plus)) + 0.25, int(config["coherence_dos"]["grid_size"]))
    density = gaussian_density(plus, multiplicity, grid, eta)
    pd.DataFrame({
        "energy": grid, "layer_even_coherence_weighted_density": density,
        "broadening": eta, "tower_id": tower_id, "level": level,
    }).to_parquet(derived, index=False)
    b11 = pd.read_parquet(context["phase_b_dir"] / "derived" / "b11_full_shell_balance.parquet")
    b12 = pd.read_parquet(context["phase_b_dir"] / "derived" / "b12_full_shell_spectral_inheritance.parquet")
    row11 = b11[(b11.tower_id == tower_id) & (b11.level == level)].iloc[0]
    row12 = b12[(b12.tower_id == tower_id) & (b12.level == level)].iloc[0]
    d03 = context["d03_terms"]
    row03 = d03[(d03.tower_id == tower_id) & (d03.level == level)].iloc[0]
    errors = {
        "spectral_error": float(row11.balanced_error_sum),
        "projector_error": float(row12.riesz_projection_norm_error_upper),
        "coherence_weight_error": float(np.max(subset.characteristic_residual)),
        "smoothing_local_law_error": float(row03.combined_term),
    }
    error_table = pd.DataFrame([{"error_component": key, "upper_bound": value} for key, value in errors.items()])
    error_table.to_parquet(raw / "coherence_error_components.parquet", index=False)
    status = "PASS_CONVERGED" if context.get("d03_status") == "PASS_CONVERGED" else "INCONCLUSIVE"
    finish(certificate, {
        "probe": "layer-even bilayer coherence",
        "coherence_rule": "symmetric branch weight one; antisymmetric branch weight zero",
        "error_components": errors, "errors_tracked_independently": True,
        "fixed_broadening": eta,
        "reason_if_inconclusive": None if status == "PASS_CONVERGED" else "D-03 asymptotic vanishing-broadening gate is inconclusive; finite fixed-broadening measure is retained",
    }, status, run_id, "D-05")
    context["d05_errors"] = errors
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
