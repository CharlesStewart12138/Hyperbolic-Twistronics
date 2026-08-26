from __future__ import annotations

import math

import sympy as sp

from exact.coincidence_index_height import local_factor_record
from exact.commensurator_sequence import exact_record
from exact.euclidean_angle_catalog import enumerate_catalog
from exact.euclidean_square_csl import csl_record, exact_certificate
from geometry.crossover_core import beta_m, crossover_f, displacement, effective_area, flow_rhs, moire_length
from geometry.dimensional_extension import beta_dimension


def test_distance_and_registry_closed_forms() -> None:
    radius, theta, threshold = 1.7, 0.13, 0.2
    xi = moire_length(radius, theta, threshold)
    assert abs(displacement(radius, theta, xi) - threshold) < 1.0e-13
    assert effective_area(radius, theta, threshold) > 0


def test_crossover_exact_relations() -> None:
    chi = 2.3
    f = crossover_f(chi)
    assert abs(beta_m(chi) - 2.0 / (2.0 - f)) < 1.0e-14
    h = 1.0e-6
    derivative = (crossover_f(chi * math.exp(h)) - crossover_f(chi * math.exp(-h))) / (2 * h)
    assert abs(derivative - flow_rhs(f)) < 1.0e-9


def test_dimensional_endpoints() -> None:
    for dimension in (2, 4, 6):
        assert abs(beta_dimension(dimension, 0.01) - dimension) < 1.0e-3
        assert abs(beta_dimension(dimension, 12.0) - (dimension - 1)) < 1.0e-3


def test_square_csl_and_complete_bound() -> None:
    assert exact_certificate()["status"] == "PASS_EXACT"
    assert csl_record(2, 1)["bilayer_atoms"] == 10
    assert csl_record(4, 1)["bilayer_atoms"] == 34
    assert all(int(row["bilayer_atoms"]) < 100 for row in enumerate_catalog(100))


def test_commensurator_and_local_products() -> None:
    assert all(exact_record(7)["checks"].values())
    for j in range(1, 50):
        record = local_factor_record(j)
        assert record["product_identity"]
        assert sp.Integer(record["coincidence_degree_maximal_order"]) > 0

