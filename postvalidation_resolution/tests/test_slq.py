from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from projected_spectral import ProjectedCayleyOperator  # noqa: E402
from slq import discrete_cdf, full_reorthogonalized_probe  # noqa: E402


def test_full_reorthogonalized_slq_probability_measure() -> None:
    order = 32
    forward = (np.arange(order) + 1) % order
    backward = (np.arange(order) - 1) % order
    permutations = np.vstack([forward, forward, forward, forward, backward, backward, backward, backward])
    parent = np.arange(order) % 16
    operator = ProjectedCayleyOperator(permutations, parent, 16)
    probe = full_reorthogonalized_probe(operator, steps=12, seed=42, breakdown_tolerance=1.0e-13)
    assert abs(probe.weights.sum() - 1.0) < 1.0e-14
    assert probe.orthogonality_residual < 1.0e-10
    grid = np.linspace(-1.0, 1.0, 101)
    cdf = discrete_cdf(probe.nodes, probe.weights, grid)
    assert np.all(np.diff(cdf) >= -1.0e-15)
    assert cdf[-1] > 0.999999
