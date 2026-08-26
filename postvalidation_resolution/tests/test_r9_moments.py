from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "workflow"))

from run_r9_balanced import c1_tail, c2_tail, two_by_two_jacobi_from_chebyshev  # noqa: E402


def test_two_point_symmetric_measure_jacobi_matrix() -> None:
    # Equal mass at x=+-1/2 has Chebyshev moments T0=1,T1=0,T2=-1/2,T3=0.
    jacobi = two_by_two_jacobi_from_chebyshev(np.asarray([1.0, 0.0, -0.5, 0.0]))
    assert np.allclose(jacobi, np.asarray([[0.0, 0.5], [0.5, 0.0]]))


def test_closed_form_derivative_tails() -> None:
    q = 0.2
    cutoff = 3
    direct1 = sum(index * q**index for index in range(cutoff + 1, 200))
    direct2 = sum(index * index * q**index for index in range(cutoff + 1, 200))
    assert abs(c1_tail(q, cutoff) - direct1) < 1.0e-14
    assert abs(c2_tail(q, cutoff) - direct2) < 1.0e-14
