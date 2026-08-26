from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from projected_spectral import (  # noqa: E402
    ProjectedCayleyOperator,
    gaussian_smooth_density,
    kpm_density,
    stochastic_chebyshev_moments,
)


def test_projection_annihilates_parent_lifts_and_operator_is_symmetric() -> None:
    order = 8
    forward = (np.arange(order) + 1) % order
    backward = (np.arange(order) - 1) % order
    permutations = np.vstack([forward, forward, forward, forward, backward, backward, backward, backward])
    parent = np.arange(order) % 4
    operator = ProjectedCayleyOperator(permutations, parent, 4)
    lifted = np.asarray([1.0, 2.0, 3.0, 4.0])[parent]
    assert np.linalg.norm(operator.project(lifted)) < 1.0e-14
    assert operator.symmetry_residual(1234) < 1.0e-14


def test_kpm_density_is_positive_and_normalized() -> None:
    order = 16
    forward = (np.arange(order) + 1) % order
    backward = (np.arange(order) - 1) % order
    permutations = np.vstack([forward, forward, forward, forward, backward, backward, backward, backward])
    parent = np.arange(order) % 8
    operator = ProjectedCayleyOperator(permutations, parent, 8)
    moments = stochastic_chebyshev_moments(operator, order=24, random_vectors=3, seed=99)
    grid = np.linspace(-0.995, 0.995, 301)
    density = kpm_density(moments.mean(axis=0), grid)
    smoothed = gaussian_smooth_density(density, grid, 0.08)
    assert np.min(density) >= 0
    assert abs(np.trapezoid(density, grid) - 1.0) < 1.0e-12
    assert abs(np.trapezoid(smoothed, grid) - 1.0) < 1.0e-12
