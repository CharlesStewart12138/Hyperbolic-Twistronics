from __future__ import annotations

import numpy as np

from r16_master import *  # noqa: F401,F403
from r16_master import analytic_operator_bundle, sup_operator_norm


def finite_difference_derivative_check(model, w: float, energy_scale: float) -> dict[str, float]:
    """Independent five-point check of the exact derivative implementation."""
    coordinates = (-0.21, -0.07, 0.13, 0.29)
    h = 1.0e-3
    first_errors = []
    second_errors = []
    for coordinate in coordinates:
        analytic = analytic_operator_bundle(model, np.asarray([coordinate]), w, energy_scale)
        plus2 = model.hamiltonian(coordinate + 2.0 * h, w) / energy_scale
        plus = model.hamiltonian(coordinate + h, w) / energy_scale
        center = model.hamiltonian(coordinate, w) / energy_scale
        minus = model.hamiltonian(coordinate - h, w) / energy_scale
        minus2 = model.hamiltonian(coordinate - 2.0 * h, w) / energy_scale
        first_fd = (-plus2 + 8.0 * plus - 8.0 * minus + minus2) / (12.0 * h)
        second_fd = (-plus2 + 16.0 * plus - 30.0 * center + 16.0 * minus - minus2) / (12.0 * h * h)
        first_errors.append(sup_operator_norm(np.asarray([first_fd - analytic.D1[0]])))
        second_errors.append(sup_operator_norm(np.asarray([second_fd - analytic.D2[0]])))
    return {
        "maximum_first_derivative_check_error": float(max(first_errors)),
        "maximum_second_derivative_check_error": float(max(second_errors)),
    }
