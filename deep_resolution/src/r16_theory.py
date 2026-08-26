from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from geometry.build_orbit_and_frames import frame_at
from model.build_aro3b_hamiltonian import slater_koster_block


@dataclass(frozen=True)
class LocalParameters:
    a: float
    radius: float
    theta: float
    lambda_perp: float
    lambda_parallel: float
    cutoff: int
    X: float


@dataclass(frozen=True)
class OperatorBundle:
    q: np.ndarray
    H: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    energy_origin: float
    energy_scale: float
    hermiticity_residual: float
    parameters: LocalParameters


def moire_length(radius: float, theta: float, spacing: float) -> float:
    return radius * math.asinh(math.sinh(spacing / (2.0 * radius)) / math.sin(theta / 2.0))


def sup_norm(matrices: np.ndarray) -> float:
    return float(max(np.linalg.norm(matrix, ord=2) for matrix in np.asarray(matrices)))


def _endpoint(radius: float, distance: float, angle: float) -> np.ndarray:
    rapidity = distance / radius
    return np.asarray(
        [
            radius * math.cosh(rapidity),
            radius * math.sinh(rapidity) * math.cos(angle),
            radius * math.sinh(rapidity) * math.sin(angle),
        ],
        dtype=float,
    )


class LocalCurvatureFiber:
    """Fixed-spacing q=8 local geodesic-star fiber.

    This model is intentionally a local, symmetry-reduced diagnostic.  It
    varies a/R at fixed physical spacing a and fixed coordination without
    claiming an exact global Bloch fiber for a changed regular tessellation.
    """

    def __init__(self, parameters: LocalParameters) -> None:
        if parameters.a <= 0 or parameters.radius <= 0:
            raise ValueError("positive a and R are required")
        if not 0 < parameters.theta < math.pi:
            raise ValueError("theta is outside (0,pi)")
        if parameters.lambda_perp <= 0 or parameters.lambda_parallel <= 0:
            raise ValueError("positive decay lengths are required")
        self.parameters = parameters
        self.onsite = np.diag(np.asarray([0.0, 0.25, 0.25], dtype=float))
        self.interlayer = np.diag(np.asarray([1.0, 0.70, 0.70], dtype=float))
        self.values = {
            "V_ss_sigma": -1.0,
            "V_sp_sigma": 0.35,
            "V_pp_sigma": 0.55,
            "V_pp_pi": -0.15,
        }
        origin = np.asarray([parameters.radius, 0.0, 0.0], dtype=float)
        origin_frame = frame_at(origin, parameters.radius)
        self.intralayer_terms: list[tuple[float, np.ndarray]] = []
        for shell in (1, 2):
            envelope = 1.0 if shell == 1 else math.exp(-parameters.a / parameters.lambda_parallel)
            for direction in range(4):
                angle = math.pi * direction / 4.0
                point = _endpoint(parameters.radius, shell * parameters.a, angle)
                block = envelope * slater_koster_block(
                    origin,
                    point,
                    origin_frame,
                    frame_at(point, parameters.radius),
                    parameters.radius,
                    self.values,
                )
                frequency = shell * math.cos(angle)
                self.intralayer_terms.append((frequency, block))
        reference_q1 = 25003.0 / 300000.0
        reference_lambda = 0.20
        ell1 = -math.log(reference_q1) * reference_lambda
        self.layer_height = (parameters.a * parameters.a - ell1 * ell1) / (2.0 * ell1)
        if self.layer_height <= 0:
            raise ValueError("registered local-fiber layer height is nonpositive")
        interlayer_terms = [(0.0, 1.0)]
        for shell in range(1, parameters.cutoff + 1):
            radial = math.sqrt(self.layer_height**2 + (shell * parameters.a) ** 2) - self.layer_height
            weight = math.exp(-radial / parameters.lambda_perp)
            for direction in range(8):
                angle = math.pi * direction / 4.0
                interlayer_terms.append((shell * math.cos(angle), weight))
        normalizer = sum(weight for _, weight in interlayer_terms)
        self.interlayer_terms = [(frequency, weight / normalizer) for frequency, weight in interlayer_terms]

    def bundle(self, q: np.ndarray) -> OperatorBundle:
        p = self.parameters
        q = np.asarray(q, dtype=float)
        xi = moire_length(p.radius, p.theta, p.a)
        energy_scale = (p.a / xi) ** 2
        raw_H: list[np.ndarray] = []
        raw_D1: list[np.ndarray] = []
        raw_D2: list[np.ndarray] = []
        hermiticity = 0.0
        for coordinate in q:
            value = self.onsite.astype(complex).copy()
            first = np.zeros((3, 3), dtype=complex)
            second = np.zeros((3, 3), dtype=complex)
            for frequency, block in self.intralayer_terms:
                phase = np.exp(1j * coordinate * frequency)
                value += phase * block + phase.conjugate() * block.T
                first += 1j * frequency * phase * block - 1j * frequency * phase.conjugate() * block.T
                second += -(frequency**2) * phase * block - (frequency**2) * phase.conjugate() * block.T
            tau = 0.0
            tau_first = 0.0
            tau_second = 0.0
            for frequency, weight in self.interlayer_terms:
                angle = coordinate * frequency
                tau += weight * math.cos(angle)
                tau_first -= weight * frequency * math.sin(angle)
                tau_second -= weight * frequency * frequency * math.cos(angle)
            value += p.X * energy_scale * tau * self.interlayer
            first += p.X * energy_scale * tau_first * self.interlayer
            second += p.X * energy_scale * tau_second * self.interlayer
            hermiticity = max(
                hermiticity,
                float(np.linalg.norm(value - value.conj().T, ord=2)),
                float(np.linalg.norm(first - first.conj().T, ord=2)),
                float(np.linalg.norm(second - second.conj().T, ord=2)),
            )
            raw_H.append(0.5 * (value + value.conj().T))
            raw_D1.append(0.5 * (first + first.conj().T))
            raw_D2.append(0.5 * (second + second.conj().T))
        H = np.asarray(raw_H)
        center = int(np.argmin(np.abs(q)))
        center_values, center_vectors = np.linalg.eigh(H[center])
        target = int(np.argmax(np.abs(center_vectors[0, :]) ** 2))
        energy_origin = float(center_values[target])
        identity = np.eye(3, dtype=complex)
        return OperatorBundle(
            q=q,
            H=(H - energy_origin * identity[None, :, :]) / energy_scale,
            D1=np.asarray(raw_D1) / energy_scale,
            D2=np.asarray(raw_D2) / energy_scale,
            energy_origin=energy_origin,
            energy_scale=energy_scale,
            hermiticity_residual=hermiticity,
            parameters=p,
        )


def bundle_linear_combination(reference: OperatorBundle, terms: list[tuple[float, OperatorBundle]]) -> OperatorBundle:
    fields = {}
    for name in ("H", "D1", "D2"):
        value = getattr(reference, name).copy()
        for coefficient, term in terms:
            value += coefficient * getattr(term, name)
        fields[name] = value
    return OperatorBundle(
        q=reference.q.copy(),
        H=fields["H"],
        D1=fields["D1"],
        D2=fields["D2"],
        energy_origin=reference.energy_origin,
        energy_scale=reference.energy_scale,
        hermiticity_residual=max(
            sup_norm(fields[name] - fields[name].conj().transpose(0, 2, 1)) for name in fields
        ),
        parameters=reference.parameters,
    )


def bundle_difference(left: OperatorBundle, right: OperatorBundle, scale: float = 1.0) -> OperatorBundle:
    zero = LocalParameters(1.0, 1.0, 0.1, 1.0, 1.0, 1, 1.0)
    return OperatorBundle(
        q=left.q.copy(),
        H=scale * (left.H - right.H),
        D1=scale * (left.D1 - right.D1),
        D2=scale * (left.D2 - right.D2),
        energy_origin=0.0,
        energy_scale=1.0,
        hermiticity_residual=0.0,
        parameters=zero,
    )


def five_point_tangent(factory: Callable[[float], OperatorBundle], value: float, relative_step: float) -> tuple[OperatorBundle, OperatorBundle]:
    step = relative_step * max(abs(value), 1.0e-3)

    def estimate(h: float) -> OperatorBundle:
        plus2 = factory(value + 2.0 * h)
        plus = factory(value + h)
        minus = factory(value - h)
        minus2 = factory(value - 2.0 * h)
        reference = factory(value)
        terms = [
            (-1.0 / (12.0 * h), plus2),
            (8.0 / (12.0 * h), plus),
            (-8.0 / (12.0 * h), minus),
            (1.0 / (12.0 * h), minus2),
        ]
        zeroed = bundle_linear_combination(reference, terms)
        return bundle_difference(zeroed, reference)

    return estimate(step), estimate(0.5 * step)


def mixed_tangent(
    factory: Callable[[float, float], OperatorBundle],
    left: float,
    right: float,
    relative_step: float,
) -> OperatorBundle:
    h = relative_step * max(abs(left), 1.0e-3)
    k = relative_step * max(abs(right), 1.0e-3)
    pp = factory(left + h, right + k)
    pm = factory(left + h, right - k)
    mp = factory(left - h, right + k)
    mm = factory(left - h, right - k)
    reference = factory(left, right)
    combination = bundle_linear_combination(
        reference,
        [(1.0 / (4.0 * h * k), pp), (-1.0 / (4.0 * h * k), pm), (-1.0 / (4.0 * h * k), mp), (1.0 / (4.0 * h * k), mm)],
    )
    return bundle_difference(combination, reference)


def hodge_inner(left: OperatorBundle, right: OperatorBundle) -> float:
    result = 0.0
    for name in ("H", "D1", "D2"):
        a = getattr(left, name)
        b = getattr(right, name)
        result += float(np.mean(np.real(np.einsum("qij,qij->q", a.conj(), b)))) / a.shape[-1]
    return result


def tangent_gram(tangents: dict[str, OperatorBundle]) -> dict[str, object]:
    names = list(tangents)
    norms = {name: math.sqrt(max(hodge_inner(tangents[name], tangents[name]), 0.0)) for name in names}
    gram = np.asarray(
        [
            [hodge_inner(tangents[left], tangents[right]) / max(norms[left] * norms[right], 1.0e-300) for right in names]
            for left in names
        ],
        dtype=float,
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))[::-1]
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    return {
        "names": names,
        "norms": norms,
        "gram": gram,
        "eigenvalues": eigenvalues,
        "singular_values": singular_values,
        "rank_tolerance_1e_8": int(np.sum(singular_values > 1.0e-8)),
        "s2_over_s1": float(singular_values[1] / singular_values[0]) if len(singular_values) > 1 else 0.0,
    }


def tangent_step_stability(full: dict[str, OperatorBundle], half: dict[str, OperatorBundle]) -> dict[str, float]:
    result = {}
    for name in full:
        numerator = math.sqrt(max(hodge_inner(bundle_difference(full[name], half[name]), bundle_difference(full[name], half[name])), 0.0))
        denominator = math.sqrt(max(hodge_inner(half[name], half[name]), 1.0e-300))
        result[name] = numerator / denominator
    return result


def comparison_metrics(actual: OperatorBundle, predicted: OperatorBundle) -> dict[str, float]:
    errors = {
        "C0": sup_norm(actual.H - predicted.H),
        "C1": sup_norm(actual.D1 - predicted.D1),
        "C2": sup_norm(actual.D2 - predicted.D2),
    }
    errors.update(
        {
            "relative_C0": errors["C0"] / max(sup_norm(actual.H), 1.0),
            "relative_C1": errors["C1"] / max(sup_norm(actual.D1), 1.0),
            "relative_C2": errors["C2"] / max(sup_norm(actual.D2), 1.0),
        }
    )
    return errors


def tracked_spectral_observables(bundle: OperatorBundle) -> dict[str, float]:
    values = []
    vectors = []
    for matrix in bundle.H:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        values.append(eigenvalues)
        vectors.append(eigenvectors)
    values = np.asarray(values)
    vectors = np.asarray(vectors)
    center = len(bundle.q) // 2
    indices = np.zeros(len(bundle.q), dtype=int)
    indices[center] = int(np.argmax(np.abs(vectors[center][0]) ** 2))
    for direction in (range(center + 1, len(bundle.q)), range(center - 1, -1, -1)):
        prior = center
        for index in direction:
            overlaps = np.abs(vectors[prior][:, indices[prior]].conj() @ vectors[index]) ** 2
            indices[index] = int(np.argmax(overlaps))
            prior = index
    target = values[np.arange(len(values)), indices]
    target_vectors = vectors[np.arange(len(values)), :, indices]
    gaps = [np.min(np.abs(np.delete(values[index], band) - target[index])) for index, band in enumerate(indices)]
    hodge = np.mean([np.real(np.trace(matrix.conj().T @ matrix)) for matrix in bundle.D1])
    hessian = np.mean([np.real(np.trace(matrix.conj().T @ matrix)) for matrix in bundle.D2])
    return {
        "bandwidth": float(np.ptp(target)),
        "gap": float(np.min(gaps)),
        "hodge_D1": float(hodge),
        "hodge_D2": float(hessian),
        "coherence": float(np.mean(np.abs(target_vectors[:, 0]) ** 2)),
        "moment_1": float(np.mean(np.trace(bundle.H, axis1=1, axis2=2).real) / 3.0),
        "moment_2": float(np.mean([np.trace(matrix @ matrix).real for matrix in bundle.H]) / 3.0),
        "moment_3": float(np.mean([np.trace(matrix @ matrix @ matrix).real for matrix in bundle.H]) / 3.0),
    }


def observable_jacobian(
    reference: LocalParameters,
    q: np.ndarray,
    relative_step: float,
) -> dict[str, object]:
    fields = ("X", "g", "theta2")
    base_values = {"X": reference.X, "g": (reference.a / reference.radius) ** 2, "theta2": reference.theta**2}

    def parameters(field: str, value: float) -> LocalParameters:
        values = dict(reference.__dict__)
        if field == "X":
            values["X"] = value
        elif field == "g":
            values["radius"] = reference.a / math.sqrt(value)
        elif field == "theta2":
            values["theta"] = math.sqrt(value)
        return LocalParameters(**values)

    names = list(tracked_spectral_observables(LocalCurvatureFiber(reference).bundle(q)))
    columns = []
    raw_columns = {}
    for field in fields:
        value = base_values[field]
        h = relative_step * max(abs(value), 1.0e-3)
        plus = tracked_spectral_observables(LocalCurvatureFiber(parameters(field, value + h)).bundle(q))
        minus = tracked_spectral_observables(LocalCurvatureFiber(parameters(field, value - h)).bundle(q))
        column = np.asarray([(plus[name] - minus[name]) / (2.0 * h) for name in names], dtype=float)
        raw_columns[field] = column
        columns.append(column / max(np.linalg.norm(column), 1.0e-300))
    jacobian = np.column_stack(columns)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return {
        "observable_names": names,
        "fields": list(fields),
        "raw_columns": raw_columns,
        "normalized_jacobian": jacobian,
        "singular_values": singular,
        "rank_tolerance_1e_8": int(np.sum(singular > 1.0e-8)),
        "s2_over_s1": float(singular[1] / singular[0]),
    }


def old_octagon_shape_audit(raw_path: Path, theta: float = 0.08, lambda_perp: float = 0.20) -> dict[str, object]:
    with np.load(raw_path, allow_pickle=False) as payload:
        labels = payload["labels"].astype(str)
        H = payload["H"]
    selected = {}
    for index, label in enumerate(labels):
        if not label.endswith("X=1.00"):
            continue
        if label.startswith("reference") or label.startswith("radius_"):
            selected[label] = H[index]
    reference = selected["reference:X=1.00"]
    factor = 2.0 * math.acosh(1.0 + math.sqrt(2.0))
    reference_radius = 8.0
    reference_shape = lambda_perp / (factor * reference_radius)
    training_radii = [5.5, 7.0, 9.0, 12.0]
    holdout_radii = [6.0, 7.5, 8.5, 10.0]
    train_design = []
    train_response = []
    for radius in training_radii:
        label = f"radius_training_{radius:g}:X=1.00"
        shape = lambda_perp / (factor * radius)
        delta = shape - reference_shape
        train_design.append([delta, delta * delta])
        train_response.append((selected[label] - reference).reshape(-1))
    design = np.asarray(train_design)
    response = np.asarray(train_response)
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    rows = []
    for radius in holdout_radii:
        label = f"radius_holdout_{radius:g}:X=1.00"
        shape = lambda_perp / (factor * radius)
        delta = shape - reference_shape
        prediction = reference + (np.asarray([delta, delta * delta]) @ coefficients).reshape(reference.shape)
        error = sup_norm(selected[label] - prediction)
        baseline = sup_norm(selected[label] - reference)
        rows.append(
            {
                "radius": radius,
                "S_perp": shape,
                "delta_S_perp": delta,
                "baseline_C0": baseline,
                "shape_corrected_C0": error,
                "reduction_fraction": 1.0 - error / max(baseline, 1.0e-300),
            }
        )
    return {
        "training_radii": training_radii,
        "holdout_rows": rows,
        "model": "microscopic_profile_coordinate delta(lambda_perp/a) plus registered quadratic Taylor term",
    }

