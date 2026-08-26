from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(VALIDATION / "src"))

from projective_action import build_projective_action, symmetry_residual  # noqa: E402
from r16_theory import LocalCurvatureFiber, LocalParameters  # noqa: E402
from tower_height import exact_group_audit, hensel_root, polynomial  # noqa: E402


def test_hensel_and_relator() -> None:
    root = hensel_root(7, 2, 5)
    assert polynomial(root) % (7**5) == 0
    audit = exact_group_audit(7, 2, 5)
    assert audit["relator_pass"]
    assert audit["nonabelian_witness_pass"]


def test_projective_action_is_symmetric() -> None:
    action = build_projective_action(7, 2, 2)
    assert action.dimension == 56
    assert action.inverse_pair_pass
    assert symmetry_residual(action) < 1.0e-10
    constant = np.ones(action.dimension)
    assert np.allclose(action.apply(constant), constant)


def test_local_curvature_fiber_is_hermitian() -> None:
    parameters = LocalParameters(1.0, 8.0, 0.08, 0.20, 0.90, 5, 1.0)
    q = np.linspace(-0.2, 0.2, 17)
    bundle = LocalCurvatureFiber(parameters).bundle(q)
    assert bundle.hermiticity_residual < 1.0e-10
    assert all(np.linalg.norm(matrix - matrix.conj().T) < 1.0e-10 for matrix in bundle.H)
    assert math.isfinite(bundle.energy_scale) and bundle.energy_scale > 0

