from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator


def load_action(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path) as archive:
        permutations = np.asarray(archive["permutations"], dtype=np.int64)
        metadata = {
            "tower_id": str(archive["tower_id"]),
            "level": int(archive["level"]),
            "order": int(permutations.shape[1]),
            "modulus": int(archive["modulus"]),
        }
    return permutations, metadata


def adjacency_operator(permutations: np.ndarray) -> LinearOperator:
    order = int(permutations.shape[1])

    def matvec(vector: np.ndarray) -> np.ndarray:
        return np.sum(vector[permutations], axis=0)

    def matmat(matrix: np.ndarray) -> np.ndarray:
        return np.sum(matrix[permutations], axis=0)

    return LinearOperator((order, order), matvec=matvec, matmat=matmat, dtype=np.float64)


def bilayer_energies(adjacency_eigenvalues: np.ndarray, scale: float, coupling: float) -> np.ndarray:
    normalized = scale * np.asarray(adjacency_eigenvalues, dtype=float) / 8.0
    return np.concatenate([normalized - coupling, normalized + coupling])


def gaussian_dos(
    eigenvalues: np.ndarray, multiplicities: np.ndarray, grid: np.ndarray, broadening: float
) -> np.ndarray:
    values = np.asarray(eigenvalues, dtype=float)
    weights = np.asarray(multiplicities, dtype=float)
    weights /= np.sum(weights)
    shifted = (grid[:, None] - values[None, :]) / broadening
    return np.sum(weights[None, :] * np.exp(-0.5 * shifted**2), axis=1) / (
        math.sqrt(2.0 * math.pi) * broadening
    )


def directed_interval_loss(points: np.ndarray, lower: float, upper: float) -> float:
    ordered = np.unique(np.sort(np.asarray(points, dtype=float)))
    if len(ordered) == 0:
        return math.inf
    gaps = np.diff(ordered)
    return float(
        max(
            abs(ordered[0] - lower),
            abs(upper - ordered[-1]),
            0.5 * float(np.max(gaps)) if len(gaps) else 0.0,
        )
    )


def bilayer_reference(config: dict[str, object]) -> dict[str, float]:
    reference = config["reference_adjacency"]
    family = config["bilayer_family"]
    rho_lower = float(reference["markov_spectral_radius_lower"])
    rho_upper = float(reference["markov_spectral_radius_upper"])
    scale = float(family["normalized_adjacency_scale"])
    coupling = float(family["interlayer_coupling"])
    return {
        "rho_lower": rho_lower,
        "rho_upper": rho_upper,
        "lower_edge_lower": -coupling - scale * rho_upper,
        "lower_edge_upper": -coupling - scale * rho_lower,
        "upper_edge_lower": coupling + scale * rho_lower,
        "upper_edge_upper": coupling + scale * rho_upper,
        "bandwidth_lower": 2.0 * scale * rho_lower,
        "bandwidth_upper": 2.0 * scale * rho_upper,
        "gap_lower": 2.0 * (coupling - scale * rho_upper),
        "gap_upper": 2.0 * (coupling - scale * rho_lower),
    }

