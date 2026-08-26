from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


RELATOR_WORD = (1, -2, 3, -4, -1, 2, -3, 4)


def psl_rotation(angle: sp.Expr) -> sp.Matrix:
    """PSL(2,R) representative rotating the disk by ``angle``."""
    return sp.Matrix(
        [
            [sp.cos(angle / 2), sp.sin(angle / 2)],
            [-sp.sin(angle / 2), sp.cos(angle / 2)],
        ]
    )


def regular_octagon_generators() -> tuple[sp.Matrix, ...]:
    """Exact opposite-side pairings for the centered Bolza octagon."""
    c = 1 + sp.sqrt(2)
    s = sp.sqrt(c**2 - 1)
    base = sp.Matrix([[c, s], [s, c]])
    generators = []
    for k in range(4):
        rotation = psl_rotation(sp.pi * sp.Rational(k, 4))
        generators.append(sp.simplify(rotation * base * rotation.inv()))
    return tuple(generators)


def evaluate_word(generators: tuple[sp.Matrix, ...], word: tuple[int, ...]) -> sp.Matrix:
    value = sp.eye(2)
    for letter in word:
        generator = generators[abs(letter) - 1]
        value = value * (generator if letter > 0 else generator.inv())
    return value.applyfunc(sp.trigsimp).applyfunc(sp.simplify)


def exact_certificate() -> dict[str, object]:
    generators = regular_octagon_generators()
    determinants = [sp.simplify(generator.det()) for generator in generators]
    relation = evaluate_word(generators, RELATOR_WORD)
    determinant_pass = all(value == 1 for value in determinants)
    residuals = [relation[i, j] - (1 if i == j else 0) for i in range(2) for j in range(2)]
    relation_pass = all(residual.equals(0) is True for residual in residuals)
    c = 1 + sp.sqrt(2)
    inradius = sp.acosh(c)
    return {
        "task_id": "I-04",
        "status": "PASS_EXACT" if determinant_pass and relation_pass else "FAIL_IMPLEMENTATION",
        "backend": "SymPy exact algebraic/trigonometric expressions",
        "model": "centered regular octagon with opposite-side pairing",
        "presentation": "<g1,g2,g3,g4 | g1 g2^-1 g3 g4^-1 g1^-1 g2 g3^-1 g4 = e>",
        "standard_surface_group": "Tietze-equivalent to <a1,b1,a2,b2 | [a1,b1][a2,b2]=e>",
        "relator_word": list(RELATOR_WORD),
        "relator_matrix": [["1" if i == j else "0" for j in range(2)] for i in range(2)],
        "relator_residual_minimal_polynomials": [sp.sstr(sp.minpoly(residual)) for residual in residuals],
        "determinants": [sp.sstr(value) for value in determinants],
        "inradius_over_R": sp.sstr(inradius),
        "center_neighbor_distance_over_R": sp.sstr(2 * inradius),
        "generators": [
            [[sp.sstr(generator[i, j]) for j in range(2)] for i in range(2)]
            for generator in generators
        ],
        "checks": {
            "all_determinants_one": determinant_pass,
            "octagon_cycle_relator_identity": relation_pass,
        },
        "scope": "exact PSL(2,R) side-pairing certificate; no finite-cover spectral claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    certificate = exact_certificate()
    certificate["run_id"] = args.run_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0 if certificate["status"] == "PASS_EXACT" else 1


if __name__ == "__main__":
    raise SystemExit(main())

