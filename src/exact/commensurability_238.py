from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


def quaternion_embedding_checks() -> dict[str, object]:
    sqrt2 = sp.sqrt(2)
    sqrt3 = sp.sqrt(3)
    fourth_root2 = sp.root(2, 4)
    iota = sp.Matrix([[0, -sqrt3], [sqrt3, 0]])
    jmath = sp.Matrix([[fourth_root2, 0], [0, -fourth_root2]])
    zero = sp.zeros(2)
    identity = sp.eye(2)
    checks = {
        "iota_square_minus_three": sp.simplify(iota * iota + 3 * identity) == zero,
        "jmath_square_sqrt_two": sp.simplify(jmath * jmath - sqrt2 * identity) == zero,
        "anticommutator_zero": sp.simplify(iota * jmath + jmath * iota) == zero,
    }
    a, b, c, d = sp.symbols("a b c d", real=True)
    rho = a * identity + b * iota + c * jmath + d * iota * jmath
    norm = sp.expand(rho.det())
    expected = a**2 + 3 * b**2 - sqrt2 * c**2 - 3 * sqrt2 * d**2
    checks["reduced_norm_formula"] = sp.simplify(norm - expected) == 0
    return {
        "checks": checks,
        "embedding_iota": [[sp.sstr(iota[i, j]) for j in range(2)] for i in range(2)],
        "embedding_jmath": [[sp.sstr(jmath[i, j]) for j in range(2)] for i in range(2)],
        "reduced_norm": sp.sstr(expected),
    }


def centered_sequence_checks(j_values: tuple[int, ...] = (1, 2, 3, 8, 21)) -> dict[str, object]:
    sqrt3 = sp.sqrt(3)
    rows = []
    passed = True
    for j in j_values:
        normalizer = sp.sqrt(j * j + 3)
        matrix = sp.Matrix([[j, sqrt3], [-sqrt3, j]]) / normalizer
        fixed_residual = sp.simplify((j * sp.I + sqrt3) / (-sqrt3 * sp.I + j) - sp.I)
        determinant = sp.simplify(matrix.det())
        theta = 2 * sp.atan(sqrt3 / j)
        half_angle_squared = sp.simplify(sp.sin(theta / 2) ** 2)
        row_pass = determinant == 1 and fixed_residual == 0 and half_angle_squared == sp.Rational(3, j * j + 3)
        passed &= row_pass
        rows.append(
            {
                "j": j,
                "determinant": sp.sstr(determinant),
                "fixed_point_residual": sp.sstr(fixed_residual),
                "theta": sp.sstr(theta),
                "sin_half_squared": sp.sstr(half_angle_squared),
                "pass": row_pass,
            }
        )
    return {"pass": passed, "samples": rows}


def exact_certificate() -> dict[str, object]:
    embedding = quaternion_embedding_checks()
    sequence = centered_sequence_checks()
    orbifold_euler = sp.Rational(2) - (1 - sp.Rational(1, 2)) - (1 - sp.Rational(1, 3)) - (1 - sp.Rational(1, 8))
    all_checks = all(embedding["checks"].values()) and sequence["pass"] and orbifold_euler == -sp.Rational(1, 24)
    return {
        "task_id": "I-05",
        "status": "PASS_EXACT" if all_checks else "FAIL_IMPLEMENTATION",
        "backend": "SymPy exact algebraic numbers",
        "triangle_group": "Delta+(2,3,8)",
        "invariant_trace_field": "Q(sqrt(2))",
        "invariant_quaternion_algebra": "(-3,sqrt(2)) over Q(sqrt(2))",
        "distinguished_real_place": "split (explicit 2x2 real embedding verified)",
        "conjugate_real_place": "ramified because both Hilbert-symbol entries are negative",
        "orbifold_euler_characteristic": sp.sstr(orbifold_euler),
        "classification_reference": {
            "result": "Takeuchi arithmetic triangle-group classification, compact class (2,3,8)",
            "manuscript_pages": [67, 69, 70],
            "scope": "classification theorem is cited; algebraic specialization is recomputed here",
        },
        "embedding": embedding,
        "centered_sequence": sequence,
        "scope": "commensurability-class and centered-sequence certificate; no exact coincidence index for an unspecified torsion-free subgroup",
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

