from __future__ import annotations

from pathlib import Path

import pandas as pd
import sympy as sp

from audit.data_io import write_json


def exact_record(j: int) -> dict[str, object]:
    if j < 1:
        raise ValueError("j must be positive")
    z = sp.I
    root3 = sp.sqrt(3)
    norm = j * j + 3
    matrix = sp.Matrix([[j, root3], [-root3, j]]) / sp.sqrt(norm)
    image = sp.simplify((j * z + root3) / (-root3 * z + j))
    s2 = sp.simplify(sp.sin(sp.atan(root3 / j)) ** 2)
    checks = {
        "determinant_one": sp.simplify(matrix.det() - 1) == 0,
        "fixed_point_i": sp.simplify(image - sp.I) == 0,
        "half_angle_square": sp.simplify(s2 - sp.Rational(3, norm)) == 0,
        "reduced_norm": norm == j * j + 3,
    }
    return {
        "j": j,
        "x_j": f"{j}-iota",
        "reduced_norm": norm,
        "matrix": [[str(sp.simplify(value)) for value in row] for row in matrix.tolist()],
        "fixed_point_image": str(image),
        "theta_exact": f"2*atan(sqrt(3)/{j})",
        "sin_half_angle_squared": f"3/{norm}",
        "checks": checks,
    }


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    records = [exact_record(j) for j in range(1, 33)]
    passed = all(all(record["checks"].values()) for record in records)
    status = "PASS_EXACT" if passed else "FAIL_IMPLEMENTATION"
    exact = run_dir / "exact" / "g09_commensurator_sequence.json"
    write_json(exact, {"task_id": "G-09", "run_id": run_id, "status": status, "number_field_generators": {"sqrt_3": {"minimal_polynomial": "x^2-3", "real_embedding": "positive"}, "iota": {"square": -3}}, "records": records})
    derived = run_dir / "derived" / "commensurator_sequence.parquet"
    pd.DataFrame([{"j": row["j"], "reduced_norm": row["reduced_norm"], "theta_exact": row["theta_exact"], "sin_half_angle_squared": row["sin_half_angle_squared"]} for row in records]).to_parquet(derived, index=False)
    return status, {"raw": exact, "derived": derived, "certificate": exact}

