from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from projected_spectral import ProjectedCayleyOperator


@dataclass(frozen=True)
class SLQProbe:
    nodes: np.ndarray
    weights: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    source_norm_squared_over_dimension: float
    orthogonality_residual: float
    recurrence_residual: float


def full_reorthogonalized_probe(
    operator: ProjectedCayleyOperator,
    *,
    steps: int,
    seed: int,
    breakdown_tolerance: float,
) -> SLQProbe:
    rng = np.random.default_rng(seed)
    source = operator.project(rng.choice(np.asarray([-1.0, 1.0]), size=operator.shape[0]))
    source_norm = np.linalg.norm(source)
    if source_norm <= breakdown_tolerance:
        raise ArithmeticError("projected SLQ source vanished")
    basis = np.empty((operator.shape[0], steps), dtype=float, order="F")
    basis[:, 0] = source / source_norm
    alpha = np.empty(steps, dtype=float)
    beta = np.empty(steps - 1, dtype=float)
    actual_steps = steps
    recurrence_residual = 0.0
    for index in range(steps):
        vector = basis[:, index]
        transformed = operator @ vector
        alpha[index] = np.dot(vector, transformed)
        residual = transformed - alpha[index] * vector
        if index:
            residual -= beta[index - 1] * basis[:, index - 1]
        coefficients = basis[:, : index + 1].T @ residual
        residual -= basis[:, : index + 1] @ coefficients
        correction = basis[:, : index + 1].T @ residual
        residual -= basis[:, : index + 1] @ correction
        recurrence_residual = max(recurrence_residual, float(np.linalg.norm(correction)))
        if index == steps - 1:
            break
        beta[index] = np.linalg.norm(residual)
        if beta[index] <= breakdown_tolerance:
            actual_steps = index + 1
            break
        basis[:, index + 1] = residual / beta[index]
    alpha = alpha[:actual_steps]
    beta = beta[: max(0, actual_steps - 1)]
    nodes, eigenvectors = eigh_tridiagonal(alpha, beta)
    weights = np.square(eigenvectors[0])
    weights /= weights.sum()
    gram = basis[:, :actual_steps].T @ basis[:, :actual_steps]
    orthogonality_residual = float(np.linalg.norm(gram - np.eye(actual_steps), ord=2))
    return SLQProbe(
        nodes=nodes,
        weights=weights,
        alpha=alpha,
        beta=beta,
        source_norm_squared_over_dimension=float(
            source_norm * source_norm / operator.retained_dimension
        ),
        orthogonality_residual=orthogonality_residual,
        recurrence_residual=recurrence_residual,
    )


def discrete_cdf(nodes: np.ndarray, weights: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(nodes)
    sorted_nodes = nodes[order]
    cumulative = np.cumsum(weights[order])
    indices = np.searchsorted(sorted_nodes, grid, side="right") - 1
    result = np.zeros_like(grid, dtype=float)
    mask = indices >= 0
    result[mask] = cumulative[indices[mask]]
    return result
