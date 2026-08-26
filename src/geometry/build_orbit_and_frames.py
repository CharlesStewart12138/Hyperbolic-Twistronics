from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import yaml


ETA = np.diag([-1.0, 1.0, 1.0])


def minkowski_dot(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ ETA @ y)


def hyperbolic_distance(x: np.ndarray, y: np.ndarray, radius: float = 1.0) -> float:
    argument = -minkowski_dot(x, y) / (radius * radius)
    return radius * math.acosh(max(1.0, argument))


def hyperboloid_to_disk(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    return x[1:] / (x[0] + radius)


def disk_to_hyperboloid(z: np.ndarray, radius: float = 1.0) -> np.ndarray:
    norm2 = float(z @ z)
    denominator = 1.0 - norm2
    return np.array(
        [radius * (1.0 + norm2) / denominator, *(2.0 * radius * z / denominator)]
    )


def poincare_distance(z: np.ndarray, w: np.ndarray, radius: float = 1.0) -> float:
    denominator = math.sqrt((1.0 - float(z @ z)) * (1.0 - float(w @ w)))
    return 2.0 * radius * math.asinh(float(np.linalg.norm(z - w)) / denominator)


def lorentz_boost(rapidity: float, angle: float) -> np.ndarray:
    direction = np.array([math.cos(angle), math.sin(angle)])
    c = math.cosh(rapidity)
    s = math.sinh(rapidity)
    matrix = np.eye(3)
    matrix[0, 0] = c
    matrix[0, 1:] = s * direction
    matrix[1:, 0] = s * direction
    matrix[1:, 1:] = np.eye(2) + (c - 1.0) * np.outer(direction, direction)
    return matrix


def octagon_generators() -> tuple[np.ndarray, ...]:
    inradius_over_radius = math.acosh(1.0 + math.sqrt(2.0))
    rapidity = 2.0 * inradius_over_radius
    return tuple(lorentz_boost(rapidity, k * math.pi / 4.0) for k in range(4))


def centered_rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def build_orbit(depth: int, radius: float = 1.0) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    generators = octagon_generators()
    actions = {
        **{index + 1: generator for index, generator in enumerate(generators)},
        **{-(index + 1): np.linalg.inv(generator) for index, generator in enumerate(generators)},
    }
    center = np.array([radius, 0.0, 0.0])
    queue: deque[tuple[np.ndarray, tuple[int, ...]]] = deque([(center, tuple())])
    points: list[np.ndarray] = []
    words: list[tuple[int, ...]] = []
    seen: set[tuple[float, float, float]] = set()
    while queue:
        point, word = queue.popleft()
        key = tuple(np.round(point / radius, 11))
        if key in seen:
            continue
        seen.add(key)
        points.append(point)
        words.append(word)
        if len(word) == depth:
            continue
        for letter, action in actions.items():
            if word and letter == -word[-1]:
                continue
            queue.append((action @ point, (*word, letter)))
    order = sorted(range(len(points)), key=lambda idx: (len(words[idx]), words[idx]))
    return np.stack([points[idx] for idx in order]), [words[idx] for idx in order]


def frame_at(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    spatial = x[1:]
    norm = float(np.linalg.norm(spatial))
    base = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    if norm < 1.0e-14:
        return base
    rapidity = math.asinh(norm / radius)
    angle = math.atan2(spatial[1], spatial[0])
    return lorentz_boost(rapidity, angle) @ base


def unit_tangent_toward(x: np.ndarray, y: np.ndarray, radius: float = 1.0) -> np.ndarray:
    distance = hyperbolic_distance(x, y, radius)
    if distance == 0:
        raise ValueError("direction is undefined for coincident points")
    c = math.cosh(distance / radius)
    return (y - c * x) / (radius * math.sinh(distance / radius))


def parallel_transport(x: np.ndarray, y: np.ndarray, vector: np.ndarray, radius: float = 1.0) -> np.ndarray:
    denominator = radius * radius - minkowski_dot(x, y)
    return vector + (minkowski_dot(y, vector) / denominator) * (x + y)


def frame_gram(frame: np.ndarray) -> np.ndarray:
    return frame.T @ ETA @ frame


def validation_checks(points: np.ndarray, frames: np.ndarray, radius: float) -> dict[str, float | bool]:
    disk = np.stack([hyperboloid_to_disk(point, radius) for point in points])
    constraints = [abs(minkowski_dot(point, point) + radius * radius) for point in points]
    coordinate_errors = []
    for i in range(min(len(points), 24)):
        for j in range(i + 1, min(len(points), 24)):
            coordinate_errors.append(
                abs(hyperbolic_distance(points[i], points[j], radius) - poincare_distance(disk[i], disk[j], radius))
            )
    gram_errors = [float(np.linalg.norm(frame_gram(frame) - np.eye(2), ord=2)) for frame in frames]
    transport_errors = []
    for index in range(1, min(len(points), 20)):
        transported = np.column_stack(
            [parallel_transport(points[0], points[index], frames[0][:, j], radius) for j in range(2)]
        )
        returned = np.column_stack(
            [parallel_transport(points[index], points[0], transported[:, j], radius) for j in range(2)]
        )
        tangent_error = max(abs(minkowski_dot(points[index], transported[:, j])) for j in range(2))
        transport_errors.append(max(float(np.linalg.norm(returned - frames[0])), tangent_error))
    generator = octagon_generators()[0]
    isometry_errors = []
    for index in range(1, min(len(points), 20)):
        before = hyperbolic_distance(points[0], points[index], radius)
        after = hyperbolic_distance(generator @ points[0], generator @ points[index], radius)
        isometry_errors.append(abs(before - after))
    return {
        "hyperboloid_constraint_max": max(constraints, default=0.0),
        "coordinate_distance_residual_max": max(coordinate_errors, default=0.0),
        "frame_orthonormality_residual_max": max(gram_errors, default=0.0),
        "parallel_transport_residual_max": max(transport_errors, default=0.0),
        "group_isometry_residual_max": max(isometry_errors, default=0.0),
    }


def build_and_save(config_path: Path, output_h5: Path, certificate_path: Path, run_id: str) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    geometry = config["geometry"]
    radius = float(geometry["curvature_radius"])
    points, words = build_orbit(int(geometry["orbit_depth"]), radius)
    frames = np.stack([frame_at(point, radius) for point in points])
    disk = np.stack([hyperboloid_to_disk(point, radius) for point in points])
    checks = validation_checks(points, frames, radius)
    tolerance = float(config["numerics"]["distance_tolerance"])
    status = "PASS_CONVERGED" if all(float(value) <= 10 * tolerance for value in checks.values()) else "FAIL_IMPLEMENTATION"
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as handle:
        handle.attrs["run_id"] = run_id
        handle.attrs["task_id"] = "I-07"
        handle.attrs["curvature_radius"] = radius
        handle.create_dataset("hyperboloid", data=points)
        handle.create_dataset("poincare_disk", data=disk)
        handle.create_dataset("frames", data=frames)
        string_type = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("group_words", data=[" ".join(map(str, word)) for word in words], dtype=string_type)
    certificate = {
        "task_id": "I-07",
        "run_id": run_id,
        "status": status,
        "site_count": len(points),
        "orbit_depth": int(geometry["orbit_depth"]),
        "checks": checks,
        "tolerance": tolerance,
        "coordinate_systems": ["hyperboloid", "Poincare disk", "intrinsic geodesic"],
        "transport": "closed-form Levi-Civita transport on the hyperboloid",
        "raw_output": output_h5.name,
    }
    certificate_path.parent.mkdir(parents=True, exist_ok=True)
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    return certificate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    certificate = build_and_save(args.config, args.output_h5, args.certificate, args.run_id)
    print(args.certificate)
    return 0 if certificate["status"] == "PASS_CONVERGED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

