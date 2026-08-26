import numpy as np

from bulk.finite_cover_model import bilayer_energies, bilayer_reference, directed_interval_loss
from representation.wedderburn_exact import cyclotomic_complex


def test_gap_cyclotomic_parser():
    assert abs(cyclotomic_complex("E(4)") - 1j) < 1.0e-12
    assert abs(cyclotomic_complex("-E(3)-E(3)^2") - 1.0) < 1.0e-12


def test_directed_interval_loss_exact_for_grid():
    assert directed_interval_loss(np.array([-1.0, 0.0, 1.0]), -1.0, 1.0) == 0.5


def test_bilayer_family_has_certified_gap():
    config = {"reference_adjacency": {"markov_spectral_radius_lower": 0.662477, "markov_spectral_radius_upper": 0.6628153757}, "bilayer_family": {"normalized_adjacency_scale": 1.0, "interlayer_coupling": 1.0}}
    reference = bilayer_reference(config)
    assert reference["gap_lower"] > 0
    values = bilayer_energies(np.array([-8.0, 8.0]), 1.0, 1.0)
    assert set(values) == {-2.0, 0.0, 2.0}
