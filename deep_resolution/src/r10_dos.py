from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from projective_action import ProjectiveAction


def jackson_factors(order: int) -> np.ndarray:
    if order < 2:
        raise ValueError("Jackson order must be at least two")
    k = np.arange(order + 1, dtype=float)
    denominator = order + 2.0
    return (
        (order - k + 2.0) * np.cos(math.pi * k / denominator)
        + np.sin(math.pi * k / denominator) / math.tan(math.pi / denominator)
    ) / denominator


def kpm_local_moments(action: ProjectiveAction, order: int) -> tuple[np.ndarray, dict[str, float]]:
    seed = np.zeros(action.dimension, dtype=float)
    seed[0] = 1.0
    previous = seed.copy()
    current = action.apply(seed)
    moments = np.empty(order + 1, dtype=float)
    moments[0] = 1.0
    moments[1] = current[0]
    recurrence_residual = 0.0
    maximum_norm = max(np.linalg.norm(previous), np.linalg.norm(current))
    for degree in range(2, order + 1):
        following = 2.0 * action.apply(current) - previous
        moments[degree] = following[0]
        recurrence_residual = max(
            recurrence_residual,
            float(np.linalg.norm(following - (2.0 * action.apply(current) - previous))),
        )
        maximum_norm = max(maximum_norm, float(np.linalg.norm(following)))
        previous, current = current, following
    return moments, {
        "mu0_error": abs(moments[0] - 1.0),
        "maximum_chebyshev_vector_norm": maximum_norm,
        "recurrence_residual": recurrence_residual,
    }


def kpm_density_cdf(moments: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moments = np.asarray(moments, dtype=float)
    grid = np.asarray(grid, dtype=float)
    factors = jackson_factors(len(moments) - 1)
    theta = np.arccos(grid)
    degrees = np.arange(len(moments), dtype=float)
    values = factors[0] * moments[0] + 2.0 * np.sum(
        (factors[1:] * moments[1:])[:, None] * np.cos(degrees[1:, None] * theta[None, :]),
        axis=0,
    )
    density = values / (math.pi * np.sqrt(1.0 - grid * grid))
    density = np.maximum(density, 0.0)
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    normalization = cdf[-1]
    if normalization <= 0:
        raise RuntimeError("KPM density normalization is nonpositive")
    density /= normalization
    cdf /= normalization
    return density, cdf


def slq_local(
    action: ProjectiveAction,
    depth: int,
    *,
    temp_parent: Path | None = None,
    breakdown_tolerance: float = 1.0e-13,
) -> dict[str, np.ndarray | float | int]:
    if depth < 2:
        raise ValueError("Lanczos depth must be at least two")
    with tempfile.TemporaryDirectory(prefix="deep_slq_", dir=temp_parent) as directory:
        basis_path = Path(directory) / "basis.dat"
        basis = np.memmap(basis_path, dtype="float32", mode="w+", shape=(depth + 1, action.dimension))
        current = np.zeros(action.dimension, dtype=float)
        current[0] = 1.0
        previous = np.zeros_like(current)
        basis[0] = current.astype(np.float32)
        alpha: list[float] = []
        beta: list[float] = []
        maximum_orthogonality = 0.0
        for step in range(depth):
            image = action.apply(current)
            diagonal = float(np.dot(current, image))
            residual = image - diagonal * current
            if step > 0:
                residual -= beta[-1] * previous
            for _ in range(2):
                stored = np.asarray(basis[: step + 1], dtype=np.float32)
                coefficients = stored @ residual.astype(np.float32)
                residual -= np.asarray(coefficients, dtype=float) @ stored.astype(float)
            alpha.append(diagonal)
            off_diagonal = float(np.linalg.norm(residual))
            if off_diagonal <= breakdown_tolerance or step == depth - 1:
                break
            beta.append(off_diagonal)
            previous, current = current, residual / off_diagonal
            basis[step + 1] = current.astype(np.float32)
            overlaps = np.asarray(basis[: step + 1], dtype=np.float32) @ current.astype(np.float32)
            maximum_orthogonality = max(maximum_orthogonality, float(np.max(np.abs(overlaps))))
        actual_depth = len(alpha)
        tridiagonal = np.diag(np.asarray(alpha, dtype=float))
        if beta:
            off = np.asarray(beta[: actual_depth - 1], dtype=float)
            tridiagonal += np.diag(off, 1) + np.diag(off, -1)
        nodes, vectors = np.linalg.eigh(tridiagonal)
        weights = np.abs(vectors[0]) ** 2
        return {
            "alpha": np.asarray(alpha, dtype=float),
            "beta": np.asarray(beta[: actual_depth - 1], dtype=float),
            "nodes": nodes,
            "weights": weights,
            "actual_depth": actual_depth,
            "weight_sum_error": float(abs(np.sum(weights) - 1.0)),
            "maximum_orthogonality_residual": maximum_orthogonality,
        }


def gaussian_slq_density_cdf(nodes: np.ndarray, weights: np.ndarray, grid: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    grid = np.asarray(grid, dtype=float)
    scaled = (grid[:, None] - nodes[None, :]) / sigma
    density = np.sum(weights[None, :] * np.exp(-0.5 * scaled * scaled) / (math.sqrt(2.0 * math.pi) * sigma), axis=1)
    cdf = np.sum(weights[None, :] * ndtr(scaled), axis=1)
    cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1.0e-300)
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    density /= max(float(np.sum(increments)), 1.0e-300)
    return density, cdf


def cdf_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def lorentzian_density(nodes: np.ndarray, weights: np.ndarray, grid: np.ndarray, eta: float) -> np.ndarray:
    difference = grid[:, None] - np.asarray(nodes, dtype=float)[None, :]
    density = np.sum(
        np.asarray(weights, dtype=float)[None, :] * eta / math.pi / (difference * difference + eta * eta),
        axis=1,
    )
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    return density / max(float(np.sum(increments)), 1.0e-300)


def regular_interval_audit(
    grid: np.ndarray,
    density: np.ndarray,
    nodes: np.ndarray,
    resolution: float,
) -> dict[str, object]:
    """Finite-level diagnostics only; this is not a limiting regularity proof."""
    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    critical = [-1.0, 1.0]
    # Persistent high-weight SLQ atoms are registered as possible singularities.
    nodes = np.asarray(nodes, dtype=float)
    for value in nodes:
        if all(abs(value - existing) > 2.0 * resolution for existing in critical):
            critical.append(float(value))
    regular = np.ones(len(grid), dtype=bool)
    for value in critical:
        regular &= np.abs(grid - value) > 2.0 * resolution
    derivative = np.gradient(density, grid, edge_order=2)
    return {
        "critical_energies": sorted(critical),
        "exclusion_radius": 2.0 * resolution,
        "regular_grid_fraction": float(np.mean(regular)),
        "finite_level_density_sup_on_regular_grid": float(np.max(density[regular])) if np.any(regular) else math.inf,
        "finite_level_lipschitz_diagnostic": float(np.max(np.abs(derivative[regular]))) if np.any(regular) else math.inf,
        "analytic_uniform_modulus_certified": False,
        "reason": "finite projective-sector data do not establish a tower-uniform limiting coarea/Morse hypothesis",
    }

