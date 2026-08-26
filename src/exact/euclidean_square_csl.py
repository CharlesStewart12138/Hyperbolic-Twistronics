from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import pandas as pd
import sympy as sp

from audit.data_io import write_json


def csl_record(m: int, n: int) -> dict[str, object]:
    if not (m > n >= 1 and math.gcd(m, n) == 1):
        raise ValueError("require coprime integers m > n >= 1")
    raw_norm = m * m + n * n
    parity_reduction = 2 if (m % 2 == 1 and n % 2 == 1) else 1
    index = raw_norm // parity_reduction
    cos_theta = Fraction(m * m - n * n, raw_norm)
    sin_theta = Fraction(2 * m * n, raw_norm)
    rotation = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
    gaussian = f"{m}+{n}*I"
    return {
        "m": m,
        "n": n,
        "gaussian_integer": gaussian,
        "gaussian_norm": raw_norm,
        "primitive_parity_reduction": parity_reduction,
        "csl_index_single_layer": index,
        "bilayer_atoms": 2 * index,
        "cos_theta": str(cos_theta),
        "sin_theta": str(sin_theta),
        "rotation_matrix": [[str(value) for value in row] for row in rotation],
    }


def exact_certificate() -> dict[str, object]:
    ten = csl_record(2, 1)
    thirty_four = csl_record(4, 1)
    parity = csl_record(3, 1)
    matrix = sp.Matrix([[sp.Rational(ten["cos_theta"]), -sp.Rational(ten["sin_theta"])], [sp.Rational(ten["sin_theta"]), sp.Rational(ten["cos_theta"])]])
    checks = {
        "ten_site_bilayer": ten["bilayer_atoms"] == 10,
        "thirty_four_site_bilayer": thirty_four["bilayer_atoms"] == 34,
        "odd_odd_parity_reduction": parity["primitive_parity_reduction"] == 2 and parity["csl_index_single_layer"] == 5,
        "orthogonal_rotation": matrix.T * matrix == sp.eye(2),
        "unit_determinant": sp.factor(matrix.det()) == 1,
    }
    return {"status": "PASS_EXACT" if all(checks.values()) else "FAIL_IMPLEMENTATION", "checks": checks, "examples": [ten, thirty_four, parity]}


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    data = exact_certificate()
    data.update({"task_id": "G-07", "run_id": run_id, "exact_backend": "SymPy rational arithmetic"})
    exact = run_dir / "exact" / "g07_euclidean_square_csl.json"
    write_json(exact, data)
    derived = run_dir / "derived" / "euclidean_square_csl_examples.parquet"
    pd.DataFrame([{key: value for key, value in row.items() if key != "rotation_matrix"} for row in data["examples"]]).to_parquet(derived, index=False)
    return str(data["status"]), {"raw": exact, "derived": derived, "certificate": exact}

