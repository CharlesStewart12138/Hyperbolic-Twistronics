from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OperatorBundle:
    q: np.ndarray
    H: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    energy_origin: float
    energy_scale: float
    w_over_t: float
    hermiticity_residual: float


@dataclass(frozen=True)
class SpectralBundle:
    eigenvalues: np.ndarray
    target_energy: np.ndarray
    target_projectors: np.ndarray
    target_coherence: np.ndarray
    external_gap: np.ndarray
    minimum_tracking_overlap: float


def analytic_operator_bundle(model, q: np.ndarray, w: float, energy_scale: float) -> OperatorBundle:
    """Evaluate H, dH/dq, and d2H/dq2 in the frozen tracking coordinate.

    The active model uses momentum=q*(1/2,1/2,1/2,1/2).  Differentiating
    the phase and even-kernel expressions therefore gives exact finite sums;
    no scientific derivative is taken inside a plotting routine.
    """
    q = np.asarray(q, dtype=float)
    if energy_scale <= 0:
        raise ValueError("energy scale must be positive")
    frequency = 0.5 * np.sum(np.asarray(model.abelian, dtype=float), axis=1)
    weights = np.asarray(model.weights, dtype=float)
    H = []
    D1 = []
    D2 = []
    maximum_hermiticity = 0.0
    for coordinate in q:
        value = np.diag(model.onsite).astype(complex)
        derivative = np.zeros_like(value)
        second = np.zeros_like(value)
        phase = np.exp(0.5j * float(coordinate))
        for block in model.blocks:
            value += phase * block + phase.conjugate() * block.T
            derivative += 0.5j * phase * block - 0.5j * phase.conjugate() * block.T
            second += -0.25 * phase * block - 0.25 * phase.conjugate() * block.T
        angles = float(coordinate) * frequency
        tau = float(np.sum(weights * np.cos(angles)))
        tau_d1 = float(-np.sum(weights * frequency * np.sin(angles)))
        tau_d2 = float(-np.sum(weights * frequency * frequency * np.cos(angles)))
        value += float(w) * tau * model.interlayer
        derivative += float(w) * tau_d1 * model.interlayer
        second += float(w) * tau_d2 * model.interlayer
        maximum_hermiticity = max(
            maximum_hermiticity,
            float(np.linalg.norm(value - value.conj().T, ord=2)),
            float(np.linalg.norm(derivative - derivative.conj().T, ord=2)),
            float(np.linalg.norm(second - second.conj().T, ord=2)),
        )
        H.append(0.5 * (value + value.conj().T))
        D1.append(0.5 * (derivative + derivative.conj().T))
        D2.append(0.5 * (second + second.conj().T))
    H_array = np.asarray(H)
    center = int(np.argmin(np.abs(q)))
    center_values, center_vectors = np.linalg.eigh(H_array[center])
    target = int(np.argmax(np.abs(center_vectors[0, :]) ** 2))
    energy_origin = float(center_values[target])
    identity = np.eye(H_array.shape[-1], dtype=complex)
    return OperatorBundle(
        q=q,
        H=(H_array - energy_origin * identity[None, :, :]) / float(energy_scale),
        D1=np.asarray(D1) / float(energy_scale),
        D2=np.asarray(D2) / float(energy_scale),
        energy_origin=energy_origin,
        energy_scale=float(energy_scale),
        w_over_t=float(w),
        hermiticity_residual=maximum_hermiticity,
    )


def track_spectrum(operators: np.ndarray) -> SpectralBundle:
    operators = np.asarray(operators, dtype=complex)
    values = []
    vectors = []
    for matrix in operators:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        values.append(eigenvalues)
        vectors.append(eigenvectors)
    eigenvalues = np.asarray(values, dtype=float)
    eigenvectors = np.asarray(vectors, dtype=complex)
    center = len(operators) // 2
    target_index = np.zeros(len(operators), dtype=int)
    target_index[center] = int(np.argmax(np.abs(eigenvectors[center][0, :]) ** 2))
    overlaps = np.ones(len(operators), dtype=float)
    for indices in (range(center + 1, len(operators)), range(center - 1, -1, -1)):
        previous = center
        for index in indices:
            prior = eigenvectors[previous][:, target_index[previous]]
            candidates = np.abs(prior.conj() @ eigenvectors[index]) ** 2
            target_index[index] = int(np.argmax(candidates))
            overlaps[index] = float(candidates[target_index[index]])
            previous = index
    target_vectors = eigenvectors[np.arange(len(operators)), :, target_index]
    projectors = np.einsum("qi,qj->qij", target_vectors, target_vectors.conj())
    target_energy = eigenvalues[np.arange(len(operators)), target_index]
    coherence = np.abs(target_vectors[:, 0]) ** 2
    external_gap = np.asarray(
        [
            np.min(np.abs(np.delete(eigenvalues[index], band) - target_energy[index]))
            for index, band in enumerate(target_index)
        ],
        dtype=float,
    )
    return SpectralBundle(
        eigenvalues=eigenvalues,
        target_energy=target_energy,
        target_projectors=projectors,
        target_coherence=coherence,
        external_gap=external_gap,
        minimum_tracking_overlap=float(np.min(overlaps)),
    )


def sup_operator_norm(values: np.ndarray) -> float:
    return float(max(np.linalg.norm(matrix, ord=2) for matrix in np.asarray(values)))


def comparison_metrics(
    candidate: OperatorBundle,
    reference: OperatorBundle,
    candidate_spectrum: SpectralBundle | None = None,
    reference_spectrum: SpectralBundle | None = None,
) -> dict[str, float]:
    if candidate_spectrum is None:
        candidate_spectrum = track_spectrum(candidate.H)
    if reference_spectrum is None:
        reference_spectrum = track_spectrum(reference.H)
    return {
        "epsilon_C0": sup_operator_norm(candidate.H - reference.H),
        "epsilon_C1": sup_operator_norm(candidate.D1 - reference.D1),
        "epsilon_C2": sup_operator_norm(candidate.D2 - reference.D2),
        "complete_spectrum_sup_error": float(
            np.max(np.abs(candidate_spectrum.eigenvalues - reference_spectrum.eigenvalues))
        ),
        "bandwidth_error": float(
            abs(np.ptp(candidate_spectrum.target_energy) - np.ptp(reference_spectrum.target_energy))
        ),
        "gap_error": float(
            abs(np.min(candidate_spectrum.external_gap) - np.min(reference_spectrum.external_gap))
        ),
        "projector_error": sup_operator_norm(
            candidate_spectrum.target_projectors - reference_spectrum.target_projectors
        ),
        "coherence_error": float(
            np.max(np.abs(candidate_spectrum.target_coherence - reference_spectrum.target_coherence))
        ),
        "minimum_tracking_overlap": candidate_spectrum.minimum_tracking_overlap,
    }


def design_matrix(records: Iterable[dict[str, float]], fields: list[str]) -> np.ndarray:
    return np.asarray([[float(record[field]) for field in fields] for record in records], dtype=float)


def fit_operator_corrections(
    training_bundles: list[OperatorBundle],
    reference: OperatorBundle,
    design: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(training_bundles) != len(design):
        raise ValueError("training/design length mismatch")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("correction-field design is rank deficient")
    result = {}
    for name in ("H", "D1", "D2"):
        baseline = getattr(reference, name)
        response = np.asarray([getattr(bundle, name) - baseline for bundle in training_bundles])
        coefficients, _, _, _ = np.linalg.lstsq(design, response.reshape(len(design), -1), rcond=None)
        result[name] = coefficients.reshape((design.shape[1],) + baseline.shape)
    return result


def corrected_prediction(
    reference: OperatorBundle,
    coefficients: dict[str, np.ndarray],
    fields: np.ndarray,
) -> OperatorBundle:
    fields = np.asarray(fields, dtype=float)
    predicted = {}
    for name in ("H", "D1", "D2"):
        predicted[name] = getattr(reference, name) + np.tensordot(fields, coefficients[name], axes=(0, 0))
    return OperatorBundle(
        q=reference.q.copy(),
        H=predicted["H"],
        D1=predicted["D1"],
        D2=predicted["D2"],
        energy_origin=reference.energy_origin,
        energy_scale=reference.energy_scale,
        w_over_t=reference.w_over_t,
        hermiticity_residual=max(
            sup_operator_norm(predicted[name] - predicted[name].conj().transpose(0, 2, 1))
            for name in ("H", "D1", "D2")
        ),
    )


def field_values(model, theta: float, reference_model, theta_reference: float, lambda_parallel: float) -> dict[str, float]:
    a = float(model.lattice_spacing)
    radius = float(model.radius)
    a_ref = float(reference_model.lattice_spacing)
    radius_ref = float(reference_model.radius)
    g_k = (a / radius) ** 2 / float(theta) ** 2
    g_k_ref = (a_ref / radius_ref) ** 2 / float(theta_reference) ** 2
    y_r = a / float(lambda_parallel) - a_ref / float(lambda_parallel)
    lambda_value = 1.0 / float(model.parameters["mu"])
    lambda_ref = 1.0 / float(reference_model.parameters["mu"])
    y_profile = lambda_value / a - lambda_ref / a_ref
    return {"Y_R": y_r, "Y_Ktheta": g_k - g_k_ref, "Y_profile": y_profile}


def finite_difference_derivative_check(model, w: float, energy_scale: float) -> dict[str, float]:
    coordinates = (-0.21, -0.07, 0.13, 0.29)
    h = 1.0e-4
    first_errors = []
    second_errors = []
    for coordinate in coordinates:
        analytic = analytic_operator_bundle(model, np.asarray([coordinate]), w, energy_scale)
        plus = model.hamiltonian(coordinate + h, w) / energy_scale
        center = model.hamiltonian(coordinate, w) / energy_scale
        minus = model.hamiltonian(coordinate - h, w) / energy_scale
        first_errors.append(sup_operator_norm(np.asarray([(plus - minus) / (2.0 * h) - analytic.D1[0]])))
        second_errors.append(
            sup_operator_norm(np.asarray([(plus - 2.0 * center + minus) / (h * h) - analytic.D2[0]]))
        )
    return {
        "maximum_first_derivative_check_error": float(max(first_errors)),
        "maximum_second_derivative_check_error": float(max(second_errors)),
    }


def near_reference_envelope(frame, value_column: str) -> dict[str, object]:
    ordered = frame.sort_values("asymptotic_scale")
    scales = ordered["asymptotic_scale"].to_numpy(dtype=float)
    values = ordered[value_column].to_numpy(dtype=float)
    upper = np.maximum.accumulate(values)
    return {
        "scales_ascending": scales.tolist(),
        "upper_envelope": upper.tolist(),
        "innermost_to_outermost_ratio": float(upper[0] / max(upper[-1], 1.0e-15)),
        "decreases_toward_reference": bool(upper[0] <= 0.75 * upper[-1]),
    }
