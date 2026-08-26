from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r16_master import (  # noqa: E402
    OperatorBundle,
    corrected_prediction,
    fit_operator_corrections,
    track_spectrum,
)


def bundle(matrix: np.ndarray) -> OperatorBundle:
    q = np.linspace(-1.0, 1.0, len(matrix))
    return OperatorBundle(q, matrix, 2.0 * matrix, 3.0 * matrix, 0.0, 1.0, 1.0, 0.0)


def test_linear_operator_correction_recovers_holdout() -> None:
    q = np.linspace(-1.0, 1.0, 7)
    reference_matrices = np.asarray([np.diag([x, -x, 0.5]) for x in q], dtype=complex)
    psi1 = np.asarray([np.diag([1.0 + x, 0.0, -1.0]) for x in q], dtype=complex)
    psi2 = np.asarray([np.diag([0.0, x * x, 0.25]) for x in q], dtype=complex)
    reference = bundle(reference_matrices)
    design = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]])
    training = [bundle(reference_matrices + row[0] * psi1 + row[1] * psi2) for row in design]
    coefficients = fit_operator_corrections(training, reference, design)
    predicted = corrected_prediction(reference, coefficients, np.asarray([0.25, -0.75]))
    assert np.max(np.abs(predicted.H - (reference_matrices + 0.25 * psi1 - 0.75 * psi2))) < 1.0e-12


def test_tracking_returns_rank_one_projectors() -> None:
    q = np.linspace(-0.3, 0.3, 9)
    matrices = np.asarray([np.diag([x - 1.0, 0.2, 2.0 - x]) for x in q], dtype=complex)
    spectrum = track_spectrum(matrices)
    traces = np.trace(spectrum.target_projectors, axis1=1, axis2=2)
    assert np.allclose(traces, 1.0)
    assert spectrum.minimum_tracking_overlap == 1.0
