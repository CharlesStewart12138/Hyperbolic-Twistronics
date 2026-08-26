from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, getcontext
from typing import Iterable

import numpy as np


RELATOR = (1, -2, 3, -4, -1, 2, -3, 4)


def polynomial(value: int) -> int:
    return value**4 - 2 * value**2 - 1


def polynomial_derivative(value: int) -> int:
    return 4 * value**3 - 4 * value


def hensel_root(p: int, root_mod_p: int, level: int) -> int:
    """Lift a simple root of x^4-2x^2-1 to Z/p^level Z exactly."""
    if level < 1:
        raise ValueError("level must be positive")
    root = int(root_mod_p) % p
    if polynomial(root) % p != 0 or polynomial_derivative(root) % p == 0:
        raise ValueError("the supplied residue is not a simple root")
    modulus = p
    for _ in range(1, level):
        quotient = polynomial(root) // modulus
        correction = (-quotient * pow(polynomial_derivative(root), -1, p)) % p
        root += correction * modulus
        modulus *= p
        root %= modulus
        if polynomial(root) % modulus != 0:
            raise ArithmeticError("Hensel lift verification failed")
    return root


def matmul2(left: tuple[int, int, int, int], right: tuple[int, int, int, int], modulus: int) -> tuple[int, int, int, int]:
    a, b, c, d = left
    e, f, g, h = right
    return (
        (a * e + b * g) % modulus,
        (a * f + b * h) % modulus,
        (c * e + d * g) % modulus,
        (c * f + d * h) % modulus,
    )


def inverse_sl2(matrix: tuple[int, int, int, int], modulus: int) -> tuple[int, int, int, int]:
    a, b, c, d = matrix
    if (a * d - b * c) % modulus != 1:
        raise ValueError("matrix is not in SL(2)")
    return d % modulus, (-b) % modulus, (-c) % modulus, a % modulus


def marked_generators(p: int, root_mod_p: int, level: int) -> tuple[tuple[int, int, int, int], ...]:
    modulus = p**level
    x = hensel_root(p, root_mod_p, level)
    x2 = x * x % modulus
    u = (x * x2 - x) % modulus
    generators = (
        (x2, u, u, x2),
        ((x2 + x) % modulus, x, x, (x2 - x) % modulus),
        ((x2 + u) % modulus, 0, 0, (x2 - u) % modulus),
        ((x2 + x) % modulus, (-x) % modulus, (-x) % modulus, (x2 - x) % modulus),
    )
    for matrix in generators:
        if (matrix[0] * matrix[3] - matrix[1] * matrix[2]) % modulus != 1:
            raise ArithmeticError("marked generator determinant is not one")
    return generators


def exact_group_audit(p: int, root_mod_p: int, level: int) -> dict[str, object]:
    modulus = p**level
    positive = marked_generators(p, root_mod_p, level)
    letters = positive + tuple(inverse_sl2(matrix, modulus) for matrix in positive)

    def letter(index: int) -> tuple[int, int, int, int]:
        return letters[index - 1] if index > 0 else letters[4 + (-index - 1)]

    value = (1, 0, 0, 1)
    for index in RELATOR:
        value = matmul2(value, letter(index), modulus)
    commutator = matmul2(
        matmul2(matmul2(positive[0], positive[1], modulus), inverse_sl2(positive[0], modulus), modulus),
        inverse_sl2(positive[1], modulus),
        modulus,
    )
    root = hensel_root(p, root_mod_p, level)
    return {
        "p": p,
        "level": level,
        "modulus_decimal": str(modulus),
        "modulus_digits": len(str(modulus)),
        "lifted_root_decimal": str(root),
        "lifted_root_sha256": hashlib.sha256(str(root).encode("ascii")).hexdigest(),
        "polynomial_residual_modulus": polynomial(root) % modulus,
        "relator_matrix": list(value),
        "relator_pass": value == (1, 0, 0, 1),
        "commutator_matrix": list(commutator),
        "nonabelian_witness_pass": commutator != (1, 0, 0, 1),
    }


def archimedean_letter_bound() -> dict[str, object]:
    """Rigorous analytic bound from |x| at the four embeddings.

    The two real embeddings have |x|=sqrt(1+sqrt(2)); the complex pair has
    |x|=sqrt(sqrt(2)-1).  Triangle inequalities give the displayed row-sum
    bounds for every marked generator and inverse.
    """
    getcontext().prec = 90
    two = Decimal(2)
    sqrt2 = two.sqrt()
    real_modulus = (Decimal(1) + sqrt2).sqrt()
    complex_modulus = (sqrt2 - Decimal(1)).sqrt()
    real_bound = real_modulus**2 + real_modulus**3 + real_modulus
    complex_bound = complex_modulus**2 + Decimal(2) * complex_modulus
    product = real_bound**2 * complex_bound**2
    quantum = Decimal("1e-70")
    real_upper = real_bound.quantize(quantum, rounding=ROUND_CEILING)
    complex_upper = complex_bound.quantize(quantum, rounding=ROUND_CEILING)
    product_upper = (real_upper**2 * complex_upper**2).quantize(quantum, rounding=ROUND_CEILING)
    return {
        "real_root_modulus_upper": str(real_modulus.quantize(quantum, rounding=ROUND_CEILING)),
        "complex_root_modulus_upper": str(complex_modulus.quantize(quantum, rounding=ROUND_CEILING)),
        "real_embedding_letter_row_sum_upper": str(real_upper),
        "complex_embedding_letter_row_sum_upper": str(complex_upper),
        "C_product_upper": str(product_upper),
        "proof": "C_real=r^2+r^3+r; C_complex=s^2+2s; C_product=C_real^2*C_complex^2",
        "rounding": "Decimal 90-digit evaluation rounded upward to 70 decimal places after exact radical formula",
    }


def kernel_word_lower(p: int, level: int, c_product_upper: float) -> float:
    numerator = level * math.log(float(p)) - math.log(16.0)
    return max(0.0, numerator / math.log(c_product_upper) - 1.0e-12)


def quotient_order(p: int, level: int) -> int:
    return p ** (3 * (level - 1)) * p * (p * p - 1)


def projective_dimension(p: int, level: int) -> int:
    return p**level + p ** (level - 1)


@dataclass(frozen=True)
class SelectedLevel:
    tower_id: str
    p: int
    root: int
    threshold: float
    level: int
    word_systole_lower: float
    injectivity_radius_lower: float


def select_levels(
    tower_id: str,
    p: int,
    root: int,
    thresholds: Iterable[float],
    c_product_upper: float,
    maximum_level: int,
) -> list[SelectedLevel]:
    selected: list[SelectedLevel] = []
    prior = 0
    for threshold in thresholds:
        found = None
        for level in range(max(1, prior), maximum_level + 1):
            systole = kernel_word_lower(p, level, c_product_upper)
            radius = 0.5 * systole
            if radius >= float(threshold):
                found = SelectedLevel(tower_id, p, root, float(threshold), level, systole, radius)
                break
        if found is None:
            raise RuntimeError(f"no level reaches threshold {threshold} for {tower_id}")
        selected.append(found)
        prior = found.level + 1
    return selected


def selected_level_record(level: SelectedLevel) -> dict[str, object]:
    order = quotient_order(level.p, level.level)
    return {
        "tower_id": level.tower_id,
        "p": level.p,
        "root_mod_p": level.root,
        "level": level.level,
        "threshold": level.threshold,
        "word_systole_lower": level.word_systole_lower,
        "injectivity_radius_lower": level.injectivity_radius_lower,
        "quotient_order_decimal": str(order),
        "quotient_order_digits": len(str(order)),
        "genus_decimal": str(1 + order),
        "projective_sector_dimension_decimal": str(projective_dimension(level.p, level.level)),
        "hamiltonian_structural_nnz_decimal": str(8 * projective_dimension(level.p, level.level)),
    }


def verify_monotone_levels(records: list[dict[str, object]]) -> bool:
    for tower_id in sorted({str(row["tower_id"]) for row in records}):
        rows = [row for row in records if row["tower_id"] == tower_id]
        levels = [int(row["level"]) for row in rows]
        radii = [float(row["injectivity_radius_lower"]) for row in rows]
        if not all(right > left for left, right in zip(levels, levels[1:])):
            return False
        if not all(right > left for left, right in zip(radii, radii[1:])):
            return False
    return True

