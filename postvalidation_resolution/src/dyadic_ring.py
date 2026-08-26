from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import sympy as sp


Coeff = tuple[int, int, int, int]
RingMatrix = tuple[int, int, int, int]


def reduce_y_polynomial(coefficients: list[int]) -> Coeff:
    work = list(coefficients) + [0] * max(0, 4 - len(coefficients))
    for degree in range(len(work) - 1, 3, -1):
        value = work[degree]
        if value:
            work[degree] = 0
            work[degree - 4] += 2 * value
            work[degree - 2] -= 4 * value
            work[degree - 1] -= 4 * value
    return tuple(int(value) for value in work[:4])


def multiply_coefficients(left: Coeff, right: Coeff) -> Coeff:
    product = [0] * 7
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] += a * b
    return reduce_y_polynomial(product)


def y_power(power: int) -> Coeff:
    value: Coeff = (1, 0, 0, 0)
    y: Coeff = (0, 1, 0, 0)
    for _ in range(power):
        value = multiply_coefficients(value, y)
    return value


@dataclass(frozen=True)
class RamifiedDyadicRing:
    depth: int
    representatives: tuple[Coeff, ...]
    key_to_index: dict[tuple[int, int, int, int], int]
    determinant: int
    adjugate: np.ndarray
    addition: np.ndarray
    multiplication: np.ndarray
    negation: np.ndarray
    zero: int
    one: int
    y: int
    x: int

    @property
    def size(self) -> int:
        return len(self.representatives)

    def key(self, coefficients: Coeff) -> tuple[int, int, int, int]:
        vector = np.asarray(coefficients, dtype=object)
        values = self.adjugate @ vector
        return tuple(int(value) % self.determinant for value in values)

    def index(self, coefficients: Coeff) -> int:
        return self.key_to_index[self.key(coefficients)]

    def add(self, left: int, right: int) -> int:
        return int(self.addition[left, right])

    def mul(self, left: int, right: int) -> int:
        return int(self.multiplication[left, right])

    def neg(self, value: int) -> int:
        return int(self.negation[value])

    def sub(self, left: int, right: int) -> int:
        return self.add(left, self.neg(right))

    def reduce_to(self, target: "RamifiedDyadicRing", value: int) -> int:
        if target.depth >= self.depth:
            raise ValueError("target depth must be smaller")
        return target.index(self.representatives[value])


def build_ring(depth: int) -> RamifiedDyadicRing:
    if depth < 1:
        raise ValueError("depth must be positive")
    columns = [y_power(depth + offset) for offset in range(4)]
    ideal = sp.Matrix(4, 4, lambda row, column: columns[column][row])
    signed_determinant = int(ideal.det())
    determinant = abs(signed_determinant)
    if determinant != 2**depth:
        raise ArithmeticError(f"Norm((x-1)^{depth})={determinant}, expected {2**depth}")
    adjugate = np.asarray(ideal.adjugate().tolist(), dtype=object)

    def key(coefficients: Coeff) -> tuple[int, int, int, int]:
        vector = np.asarray(coefficients, dtype=object)
        values = adjugate @ vector
        return tuple(int(value) % determinant for value in values)

    basis: tuple[Coeff, ...] = (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )
    representatives: list[Coeff] = [(0, 0, 0, 0)]
    key_to_index = {key(representatives[0]): 0}
    queue: deque[int] = deque([0])
    while queue:
        index = queue.popleft()
        current = representatives[index]
        for generator in basis:
            candidate = tuple(a + b for a, b in zip(current, generator, strict=True))
            candidate_key = key(candidate)
            if candidate_key not in key_to_index:
                key_to_index[candidate_key] = len(representatives)
                representatives.append(candidate)
                queue.append(len(representatives) - 1)
    if len(representatives) != determinant:
        raise ArithmeticError("additive quotient enumeration is incomplete")

    dtype = np.uint16 if determinant > 255 else np.uint8
    addition = np.empty((determinant, determinant), dtype=dtype)
    multiplication = np.empty((determinant, determinant), dtype=dtype)
    for i, left in enumerate(representatives):
        for j, right in enumerate(representatives):
            addition[i, j] = key_to_index[key(tuple(a + b for a, b in zip(left, right, strict=True)))]
            multiplication[i, j] = key_to_index[key(multiply_coefficients(left, right))]
    zero = key_to_index[key((0, 0, 0, 0))]
    one = key_to_index[key((1, 0, 0, 0))]
    y_index = key_to_index[key((0, 1, 0, 0))]
    negation = np.asarray(
        [key_to_index[key(tuple(-value for value in representative))] for representative in representatives],
        dtype=dtype,
    )
    x_index = int(addition[one, y_index])
    ring = RamifiedDyadicRing(
        depth=depth,
        representatives=tuple(representatives),
        key_to_index=key_to_index,
        determinant=determinant,
        adjugate=adjugate,
        addition=addition,
        multiplication=multiplication,
        negation=negation,
        zero=zero,
        one=one,
        y=y_index,
        x=x_index,
    )
    if ring.mul(ring.x, ring.x) == ring.zero and depth > 1:
        raise ArithmeticError("unexpected nilpotent marked parameter")
    return ring


def matrix_multiply(ring: RamifiedDyadicRing, left: RingMatrix, right: RingMatrix) -> RingMatrix:
    return (
        ring.add(ring.mul(left[0], right[0]), ring.mul(left[1], right[2])),
        ring.add(ring.mul(left[0], right[1]), ring.mul(left[1], right[3])),
        ring.add(ring.mul(left[2], right[0]), ring.mul(left[3], right[2])),
        ring.add(ring.mul(left[2], right[1]), ring.mul(left[3], right[3])),
    )


def matrix_determinant(ring: RamifiedDyadicRing, value: RingMatrix) -> int:
    return ring.sub(ring.mul(value[0], value[3]), ring.mul(value[1], value[2]))


def matrix_inverse(ring: RamifiedDyadicRing, value: RingMatrix) -> RingMatrix:
    if matrix_determinant(ring, value) != ring.one:
        raise ValueError("matrix determinant is not one")
    return (value[3], ring.neg(value[1]), ring.neg(value[2]), value[0])


def marked_generators(ring: RamifiedDyadicRing) -> tuple[RingMatrix, ...]:
    x = ring.x
    x2 = ring.mul(x, x)
    x3 = ring.mul(x2, x)
    u = ring.sub(x3, x)
    zero = ring.zero
    generators = (
        (x2, u, u, x2),
        (ring.add(x2, x), x, x, ring.sub(x2, x)),
        (ring.add(x2, u), zero, zero, ring.sub(x2, u)),
        (ring.add(x2, x), ring.neg(x), ring.neg(x), ring.sub(x2, x)),
    )
    if any(matrix_determinant(ring, value) != ring.one for value in generators):
        raise ArithmeticError("marked dyadic generator determinant check failed")
    return generators


def evaluate_word(
    ring: RamifiedDyadicRing, generators: tuple[RingMatrix, ...], word: tuple[int, ...]
) -> RingMatrix:
    identity = (ring.one, ring.zero, ring.zero, ring.one)
    inverses = tuple(matrix_inverse(ring, generator) for generator in generators)
    value = identity
    for letter in word:
        move = generators[letter - 1] if letter > 0 else inverses[-letter - 1]
        value = matrix_multiply(ring, value, move)
    return value


def expected_sl2_order(depth: int) -> int:
    return 6 * 8 ** (depth - 1)
