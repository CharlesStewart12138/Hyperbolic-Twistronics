from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tower_height import inverse_sl2, marked_generators


def _unit_inverse(values: np.ndarray, p: int, modulus: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values % p == 0):
        raise ValueError("attempted to invert a nonunit")
    lookup = np.zeros(p, dtype=np.int64)
    for residue in range(1, p):
        lookup[residue] = pow(residue, -1, p)
    inverse = lookup[values % p]
    precision = p
    while precision < modulus:
        next_precision = min(precision * precision, modulus)
        inverse = (inverse * (2 - (values % next_precision) * inverse)) % next_precision
        precision = next_precision
    if np.any((values * inverse) % modulus != 1):
        raise ArithmeticError("vectorized modular inverse failed")
    return inverse


def projective_permutation(matrix: tuple[int, int, int, int], p: int, level: int) -> np.ndarray:
    modulus = p**level
    small = p ** (level - 1)
    dimension = modulus + small
    indices = np.arange(dimension, dtype=np.int64)
    chart_a = indices < modulus
    x = indices[chart_a]
    y = indices[~chart_a] - modulus
    a, b, c, d = (np.int64(value % modulus) for value in matrix)
    u = np.empty(dimension, dtype=np.int64)
    v = np.empty(dimension, dtype=np.int64)
    u[chart_a] = (a * x + b) % modulus
    v[chart_a] = (c * x + d) % modulus
    py = (p * y) % modulus
    u[~chart_a] = (a + b * py) % modulus
    v[~chart_a] = (c + d * py) % modulus
    output = np.empty(dimension, dtype=np.int64)
    v_unit = v % p != 0
    output[v_unit] = (u[v_unit] * _unit_inverse(v[v_unit], p, modulus)) % modulus
    u_nonzero = u[~v_unit]
    if np.any(u_nonzero % p == 0):
        raise ArithmeticError("projective vector has no unit coordinate")
    ratio = (v[~v_unit] * _unit_inverse(u_nonzero, p, modulus)) % modulus
    if np.any(ratio % p != 0):
        raise ArithmeticError("second projective chart normalization failed")
    output[~v_unit] = modulus + ratio // p
    if len(np.unique(output)) != dimension:
        raise ArithmeticError("projective action is not a permutation")
    return output


@dataclass(frozen=True)
class ProjectiveAction:
    p: int
    level: int
    modulus: int
    dimension: int
    permutations: np.ndarray
    inverse_permutations: np.ndarray
    bijection_pass: bool
    inverse_pair_pass: bool

    def apply(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError("operator-vector dimension mismatch")
        result = np.zeros_like(vector)
        for permutation in self.permutations:
            result += vector[permutation]
        return result / len(self.permutations)


def build_projective_action(p: int, root: int, level: int) -> ProjectiveAction:
    modulus = p**level
    positive = marked_generators(p, root, level)
    matrices = positive + tuple(inverse_sl2(matrix, modulus) for matrix in positive)
    permutations = np.asarray([projective_permutation(matrix, p, level) for matrix in matrices], dtype=np.int64)
    inverse = np.empty_like(permutations)
    base = np.arange(permutations.shape[1], dtype=np.int64)
    for index, permutation in enumerate(permutations):
        inverse[index, permutation] = base
    inverse_pair_pass = all(np.array_equal(inverse[index], permutations[index + 4]) for index in range(4))
    return ProjectiveAction(
        p=p,
        level=level,
        modulus=modulus,
        dimension=permutations.shape[1],
        permutations=permutations,
        inverse_permutations=inverse,
        bijection_pass=True,
        inverse_pair_pass=bool(inverse_pair_pass),
    )


def symmetry_residual(action: ProjectiveAction, seed: int = 78123) -> float:
    rng = np.random.default_rng(seed)
    left = rng.normal(size=action.dimension)
    right = rng.normal(size=action.dimension)
    return float(abs(np.dot(left, action.apply(right)) - np.dot(action.apply(left), right)))

