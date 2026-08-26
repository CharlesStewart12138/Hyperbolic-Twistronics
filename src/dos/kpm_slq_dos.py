from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.run_manifest import sha256_file
from bulk.finite_cover_model import load_action
from dos.common import finish, task_paths


def jackson_coefficients(count: int) -> np.ndarray:
    index = np.arange(count + 1, dtype=float)
    angle = math.pi / (count + 2.0)
    return (
        (count - index + 2.0) * np.cos(index * angle)
        + np.sin(index * angle) / math.tan(angle)
    ) / (count + 2.0)


def chebyshev_moments(permutations: np.ndarray, count: int, seeds: list[int]) -> np.ndarray:
    order = permutations.shape[1]

    def apply(vector):
        return np.sum(vector[permutations], axis=0) / permutations.shape[0]

    rows = []
    for seed in seeds:
        vector = np.random.default_rng(seed).choice(np.asarray([-1.0, 1.0]), size=order)
        previous = vector.copy()
        current = apply(vector)
        moments = [float(vector @ previous / order), float(vector @ current / order)]
        for _ in range(2, count + 1):
            following = 2.0 * apply(current) - previous
            moments.append(float(vector @ following / order))
            previous, current = current, following
        rows.append(moments)
    return np.asarray(rows)


def exact_moments(blocks: pd.DataFrame, tower_id: str, level: int, count: int) -> np.ndarray:
    subset = blocks[(blocks.tower_id == tower_id) & (blocks.level == level)]
    values = np.asarray(subset.adjacency_eigenvalue, dtype=float) / 8.0
    weights = np.asarray(subset.regular_multiplicity, dtype=float)
    weights /= np.sum(weights)
    previous = np.ones_like(values)
    current = values.copy()
    moments = [float(weights @ previous), float(weights @ current)]
    for _ in range(2, count + 1):
        following = 2.0 * values * current - previous
        moments.append(float(weights @ following))
        previous, current = current, following
    return np.asarray(moments)


def reconstruct(moment: np.ndarray, coefficients: np.ndarray, grid: np.ndarray) -> np.ndarray:
    theta = np.arccos(grid)
    orders = np.arange(len(moment), dtype=float)
    series = coefficients[0] * moment[0] + 2.0 * np.sum(
        (coefficients[1:] * moment[1:])[None, :] * np.cos(theta[:, None] * orders[1:]),
        axis=1,
    )
    return series / (math.pi * np.sqrt(1.0 - grid * grid))


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw, derived, certificate = task_paths(run_dir, "d01_kpm_slq_dos")
    count = int(config["kpm"]["moments"])
    vector_count = int(config["kpm"]["random_vectors"])
    base_seed = int(config["kpm"]["random_seed"])
    grid = np.linspace(-0.999, 0.999, int(config["kpm"]["reconstruction_grid"]))
    jackson = jackson_coefficients(count)
    summaries = []
    density_rows = []
    for action_index, action in enumerate(context["actions"]):
        permutations, metadata = load_action(action)
        tower_id, level = str(metadata["tower_id"]), int(metadata["level"])
        seeds = [base_seed + action_index * 1000 + i for i in range(vector_count)]
        samples = chebyshev_moments(permutations, count, seeds)
        mean = np.mean(samples, axis=0)
        exact = exact_moments(context["blocks"], tower_id, level, count)
        standard_error = np.std(samples, axis=0, ddof=1) / math.sqrt(vector_count)
        density = reconstruct(mean, jackson, grid)
        exact_density = reconstruct(exact, jackson, grid)
        normalization_error = abs(float(np.trapezoid(density, grid)) - 1.0)
        rms_error = float(np.sqrt(np.mean((mean - exact) ** 2)))
        np.savez_compressed(
            raw / f"{action.stem}_kpm.npz",
            run_id=np.asarray(run_id),
            action_sha256=np.asarray(sha256_file(action)),
            tower_id=np.asarray(tower_id),
            level=np.asarray(level),
            order=np.asarray(metadata["order"]),
            seeds=np.asarray(seeds, dtype=np.int64),
            moments_per_random_vector=samples,
            mean_moments=mean,
            exact_regular_moments=exact,
            standard_error=standard_error,
            jackson_coefficients=jackson,
            reconstruction_grid=grid,
            jackson_density=density,
            exact_jackson_density=exact_density,
        )
        summaries.append(
            {
                "tower_id": tower_id, "level": level, "order": int(metadata["order"]),
                "moment_count": count, "random_vector_count": vector_count,
                "moment_rms_error": rms_error,
                "maximum_moment_standard_error": float(np.max(standard_error)),
                "density_normalization_error": normalization_error,
                "full_regular_bulk_admissible": False,
                "reason": "full regular KPM retains certified pollution and is method-validation data only",
            }
        )
        for energy, estimate, reference in zip(grid, density, exact_density):
            density_rows.append({
                "tower_id": tower_id, "level": level, "energy": float(energy),
                "kpm_density": float(estimate), "exact_jackson_density": float(reference),
            })
    pd.DataFrame(density_rows).to_parquet(derived, index=False)
    rms_limit = float(config["kpm"]["moment_rms_error_limit"])
    norm_limit = float(config["kpm"]["normalization_error_limit"])
    status = "PASS_CONVERGED" if all(
        row["moment_rms_error"] <= rms_limit and row["density_normalization_error"] <= norm_limit
        for row in summaries
    ) else "FAIL_IMPLEMENTATION"
    finish(certificate, {
        "method": "stochastic KPM with Jackson reconstruction and exact Wedderburn moment crosscheck",
        "records": summaries, "raw_seeds_saved": True, "broadening_coefficients_saved": True,
        "reconstruction_coefficients_saved": True, "full_regular_bulk_claim": False,
        "bulk_guard": "D-02 onward uses only B-03-certified retained irreducible sectors",
        "moment_rms_error_limit": rms_limit, "normalization_error_limit": norm_limit,
    }, status, run_id, "D-01")
    context["d01_summary"] = pd.DataFrame(summaries)
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
