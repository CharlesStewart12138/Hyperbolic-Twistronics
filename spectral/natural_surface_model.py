from __future__ import annotations

import ast
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from geometry.build_orbit_and_frames import ETA, hyperbolic_distance, octagon_generators


I0 = np.array([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0]])
J0 = np.array([[0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]])


def quaternion_group() -> list[np.ndarray]:
    identity = np.eye(4)
    ij = I0 @ J0
    return [identity, -identity, I0, -I0, J0, -J0, ij, -ij]


def natural_parameters(config: dict) -> dict[str, float]:
    cfg = config["natural_surface_model"]
    q1 = float(Fraction(str(cfg["q1_rational"])))
    radius = float(cfg["curvature_radius"])
    mu = 1.0 / float(cfg["lambda_perp"])
    d1 = 2.0 * radius * math.acosh(1.0 + math.sqrt(2.0))
    ell1 = -math.log(q1) / mu
    height = (d1 * d1 - ell1 * ell1) / (2.0 * ell1)
    return {"m": float(cfg["m"]), "t": float(cfg["t"]), "q1": q1, "R": radius, "mu": mu, "d1": d1, "ell1": ell1, "height": height}


def export_normal_forms(root: Path, config: dict, output: Path) -> dict[str, object]:
    backend = config["gap_backend"]
    bash = str(backend["gap_bash"])
    gap = str(backend["gap_binary_cygwin"])
    script = root / "src" / "exact" / "surface_group_normal_forms.g"
    cyg_script = "/cygdrive/" + script.drive[0].lower() + script.as_posix()[2:]
    cyg_output = "/cygdrive/" + output.drive[0].lower() + output.as_posix()[2:]
    env = os.environ.copy()
    env["VALIDATION_NORMAL_FORMS"] = cyg_output
    env["VALIDATION_MAX_LENGTH"] = str(int(config["natural_surface_model"]["normal_form_cutoff"]))
    completed = subprocess.run([bash, "--login", "-c", f'"{gap}" -q "{cyg_script}"'], env=env, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or "AUTOMATIC=true" not in completed.stdout:
        raise RuntimeError(f"GAP/KBMAG failed: {completed.stdout}\n{completed.stderr}")
    fields = {}
    for line in completed.stdout.splitlines():
        if "=" in line and line.split("=", 1)[0] in {"GAP_VERSION", "AUTOMATIC", "GROWTH", "NORMAL_FORM_COUNT"}:
            key, value = line.split("=", 1)
            fields[key.lower()] = value
    fields["stdout"] = completed.stdout
    return fields


def parse_normal_forms(path: Path) -> list[tuple[int, ...]]:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ext = ast.literal_eval(line.strip())
        letters: list[int] = []
        for index in range(0, len(ext), 2):
            generator, exponent = int(ext[index]), int(ext[index + 1])
            letters.extend([generator if exponent > 0 else -generator] * abs(exponent))
        words.append(tuple(letters))
    return words


def word_point(word: tuple[int, ...], radius: float) -> np.ndarray:
    value = np.eye(3)
    generators = octagon_generators()
    inverses = tuple(np.linalg.inv(g) for g in generators)
    for letter in word:
        value = value @ (generators[letter - 1] if letter > 0 else inverses[-letter - 1])
    return value @ np.array([radius, 0.0, 0.0])


def kernel_records(words: list[tuple[int, ...]], params: dict[str, float]) -> list[dict[str, object]]:
    origin = np.array([params["R"], 0.0, 0.0])
    records = []
    for word in words:
        point = word_point(word, params["R"])
        distance = hyperbolic_distance(origin, point, params["R"])
        radial = math.sqrt(params["height"] ** 2 + distance * distance) - params["height"]
        weight = math.exp(-params["mu"] * radial)
        abelian = np.zeros(4)
        for letter in word:
            abelian[abs(letter) - 1] += 1.0 if letter > 0 else -1.0
        records.append({"word": word, "word_length": len(word), "distance": distance, "radial_distance": radial, "weight": weight, "abelian": abelian})
    return records


def symmetry_average(matrix: np.ndarray) -> np.ndarray:
    group = quaternion_group()
    return sum(action.T @ matrix @ action for action in group) / len(group)


def packing_tail_bounds(params: dict[str, float], ca: float) -> dict[str, float]:
    radius, d1, height, mu = params["R"], params["d1"], params["height"], params["mu"]
    # The next distinct center distance follows from the exact regular-octagon two-letter orbit.
    points = []
    generators = octagon_generators()
    origin = np.array([radius, 0.0, 0.0])
    for left in generators + tuple(np.linalg.inv(g) for g in generators):
        for right in generators + tuple(np.linalg.inv(g) for g in generators):
            point = left @ right @ origin
            distance = hyperbolic_distance(origin, point, radius)
            if distance > d1 + 1.0e-8:
                points.append(distance)
    d2 = min(points)
    packing_area = 2.0 * math.pi * radius * radius * (math.cosh(d1 / (2.0 * radius)) - 1.0)
    def radial(distance: float) -> float:
        return math.sqrt(height * height + distance * distance) - height
    def count_tail(distance: float) -> float:
        ball = 2.0 * math.pi * radius * radius * (math.cosh((distance + d1 / 2.0) / radius) - 1.0)
        return max(0.0, ball / packing_area - 9.0)
    def minus_derivative(distance: float) -> float:
        return mu * distance / math.sqrt(height * height + distance * distance) * math.exp(-mu * radial(distance))
    upper = max(100.0 * radius, d2 + 100.0)
    scalar = math.exp(-mu * radial(d2)) * count_tail(d2) + quad(lambda d: minus_derivative(d) * count_tail(d), d2, upper, epsabs=1.0e-10)[0]
    weighted = math.exp(-mu * radial(d2)) * count_tail(d2) * ca * ca * (1.0 + d2 / radius) ** 2
    weighted += quad(lambda d: minus_derivative(d) * count_tail(d) * ca * ca * (1.0 + d / radius) ** 2, d2, upper, epsabs=1.0e-9)[0]
    return {"d2": d2, "scalar_l1_upper": scalar, "hodge_trace_upper": weighted, "packing_area": packing_area, "CA": ca}

