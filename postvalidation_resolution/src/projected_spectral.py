from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import iv


@dataclass(frozen=True)
class EdgeResult:
    lower_values: np.ndarray
    upper_values: np.ndarray
    lower_vectors: np.ndarray
    upper_vectors: np.ndarray
    residuals: np.ndarray


class ProjectedCayleyOperator(LinearOperator):
    def __init__(self, permutations: np.ndarray, parent_index: np.ndarray, parent_order: int):
        self.permutations = np.asarray(permutations, dtype=np.int32)
        self.parent_index = np.asarray(parent_index, dtype=np.int32)
        self.parent_order = int(parent_order)
        if self.permutations.shape[0] != 8:
            raise ValueError("eight signed generator permutations required")
        order = int(self.permutations.shape[1])
        if self.parent_index.shape != (order,):
            raise ValueError("parent-index size mismatch")
        self.fiber_counts = np.bincount(self.parent_index, minlength=self.parent_order).astype(float)
        if np.any(self.fiber_counts <= 0) or np.any(self.fiber_counts != self.fiber_counts[0]):
            raise ValueError("conditional-expectation fibers are not positive and uniform")
        self.retained_dimension = order - self.parent_order
        super().__init__(dtype=np.dtype(np.float64), shape=(order, order))

    def project(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        sums = np.bincount(self.parent_index, weights=values, minlength=self.parent_order)
        return values - (sums / self.fiber_counts)[self.parent_index]

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        projected = self.project(vector)
        result = np.zeros(self.shape[0], dtype=float)
        for permutation in self.permutations:
            result += projected[permutation]
        return self.project(result / 8.0)

    def symmetry_residual(self, seed: int) -> float:
        rng = np.random.default_rng(seed)
        left = self.project(rng.standard_normal(self.shape[0]))
        right = self.project(rng.standard_normal(self.shape[0]))
        scale = max(np.linalg.norm(left) * np.linalg.norm(right), np.finfo(float).tiny)
        return abs(np.dot(left, self @ right) - np.dot(self @ left, right)) / scale


def compute_edges(operator: ProjectedCayleyOperator, *, k: int, tolerance: float, maximum_iterations: int, seed: int) -> EdgeResult:
    if operator.retained_dimension <= k:
        raise ValueError("retained dimension is too small for the preregistered eigensolver count")
    rng = np.random.default_rng(seed)
    v0 = operator.project(rng.standard_normal(operator.shape[0]))
    v0 /= np.linalg.norm(v0)
    lower_values, lower_vectors = eigsh(
        operator,
        k=k,
        which="SA",
        tol=tolerance,
        maxiter=maximum_iterations,
        v0=v0,
    )
    upper_values, upper_vectors = eigsh(
        operator,
        k=k,
        which="LA",
        tol=tolerance,
        maxiter=maximum_iterations,
        v0=v0,
    )
    lower_order = np.argsort(lower_values)
    upper_order = np.argsort(upper_values)
    lower_values, lower_vectors = lower_values[lower_order], lower_vectors[:, lower_order]
    upper_values, upper_vectors = upper_values[upper_order], upper_vectors[:, upper_order]
    values = np.concatenate([lower_values, upper_values])
    vectors = np.column_stack([lower_vectors, upper_vectors])
    residuals = np.asarray(
        [np.linalg.norm(operator @ vectors[:, index] - values[index] * vectors[:, index]) for index in range(values.size)]
    )
    return EdgeResult(lower_values, upper_values, lower_vectors, upper_vectors, residuals)


def jackson_coefficients(order: int) -> np.ndarray:
    indices = np.arange(order + 1, dtype=float)
    angle = np.pi / (order + 2)
    return (
        (order - indices + 2) * np.cos(indices * angle)
        + np.sin(indices * angle) / np.tan(angle)
    ) / (order + 2)


def stochastic_chebyshev_moments(
    operator: ProjectedCayleyOperator,
    *,
    order: int,
    random_vectors: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    moments = np.empty((random_vectors, order + 1), dtype=float)
    dimension = float(operator.retained_dimension)
    for probe in range(random_vectors):
        raw = rng.choice(np.asarray([-1.0, 1.0]), size=operator.shape[0])
        source = operator.project(raw)
        previous = source.copy()
        moments[probe, 0] = np.dot(source, previous) / dimension
        current = operator @ source
        moments[probe, 1] = np.dot(source, current) / dimension
        for degree in range(2, order + 1):
            following = 2.0 * (operator @ current) - previous
            moments[probe, degree] = np.dot(source, following) / dimension
            previous, current = current, following
    return moments


def kpm_density(moment_vector: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = len(moment_vector) - 1
    jackson = jackson_coefficients(order)
    theta = np.arccos(grid)
    degrees = np.arange(order + 1)[:, None]
    chebyshev = np.cos(degrees * theta[None, :])
    coefficients = jackson * moment_vector
    values = coefficients[0] + 2.0 * np.sum(coefficients[1:, None] * chebyshev[1:], axis=0)
    density = values / (np.pi * np.sqrt(1.0 - grid * grid))
    density = np.maximum(density, 0.0)
    normalization = np.trapezoid(density, grid)
    if normalization <= 0:
        raise ArithmeticError("KPM density normalization is nonpositive")
    return density / normalization


def gaussian_smooth_density(density: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    delta = grid[:, None] - grid[None, :]
    kernel = np.exp(-0.5 * (delta / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    smoothed = np.trapezoid(kernel * density[None, :], grid, axis=1)
    normalization = np.trapezoid(smoothed, grid)
    return smoothed / normalization


def heat_trace_from_moments(moment_vector: np.ndarray, times: list[float]) -> np.ndarray:
    degrees = np.arange(len(moment_vector))
    return np.asarray(
        [iv(0, time) * moment_vector[0] + 2.0 * np.sum(iv(degrees[1:], time) * moment_vector[1:]) for time in times]
    )


def interval_distance(value: float, lower: float, upper: float) -> float:
    return max(lower - value, 0.0, value - upper)
