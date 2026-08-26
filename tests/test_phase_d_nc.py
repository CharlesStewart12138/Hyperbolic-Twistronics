from __future__ import annotations

from pathlib import Path

import numpy as np

from dos.common import weighted_cdf_distance
from dos.kpm_slq_dos import jackson_coefficients
from external.reproduce_hyperbloch_dos import parse_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_weighted_cdf_distance_is_symmetric_and_zero_on_self():
    values = np.asarray([-1.0, 0.0, 2.0])
    weights = np.asarray([1.0, 2.0, 1.0])
    other = np.asarray([-1.0, 1.0, 2.0])
    other_weights = np.asarray([1.0, 1.0, 2.0])
    assert weighted_cdf_distance(values, weights, values, weights) == 0.0
    assert weighted_cdf_distance(values, weights, other, other_weights) == weighted_cdf_distance(other, other_weights, values, weights)


def test_jackson_coefficients_are_normalized_and_nonnegative():
    coefficients = jackson_coefficients(64)
    assert coefficients.shape == (65,)
    assert abs(coefficients[0] - 1.0) < 1.0e-14
    assert np.all(coefficients >= -1.0e-14)


def test_public_hyperbloch_graph_parser():
    path = (
        PROJECT_ROOT / "public_data" / "HyperBloch"
        / "b13cc279bea13dda81abdfca880abad05da2565d" / "repo" / "Paclet"
        / "Resources" / "ExampleData" / "{8,3}-tess_T2.1_3.hcm"
    )
    adjacency, edges = parse_graph(path)
    assert adjacency.shape == (16, 16)
    assert len(edges) == 24
    assert np.array_equal(adjacency, adjacency.T)
    assert np.all(np.sum(adjacency, axis=1) == 3)


def test_phase_d_config_preserves_d15_dependency_order():
    text = (PROJECT_ROOT / "configs" / "analysis_plan.yaml").read_text(encoding="utf-8")
    assert "D-01:D-14" in text
    assert text.index("D-01:D-14") < text.index("G-13:G-15") < text.index("S-17:S-24") < text.index("D-15")
