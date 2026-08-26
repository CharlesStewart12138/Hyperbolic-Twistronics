from __future__ import annotations

import numpy as np

from hodge.commutant_test import exact_commutant
from spectral.natural_surface_model import I0,J0,quaternion_group,symmetry_average
from spectral.square_fivestate_exact import exact_data


def test_square_exact_data()->None:
    data=exact_data();assert data["checks"]["alpha_zero_target"];assert data["checks"]["gap_at_least_one"]

def test_quaternion_relations()->None:
    identity=np.eye(4);assert np.allclose(I0@I0,-identity);assert np.allclose(J0@J0,-identity);assert np.allclose(I0@J0,-J0@I0);assert len(quaternion_group())==8

def test_symmetric_commutant_exact()->None:
    assert exact_commutant()["scalar_commutant"]

def test_symmetry_average_is_scalar()->None:
    matrix=np.array([[2,.2,.1,0],[.2,1,0,.3],[.1,0,3,.1],[0,.3,.1,4.]])
    averaged=symmetry_average(matrix);assert np.linalg.norm(averaged-np.trace(averaged)/4*np.eye(4))<1e-12

