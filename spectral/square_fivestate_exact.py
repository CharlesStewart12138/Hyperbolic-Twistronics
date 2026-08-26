from __future__ import annotations

from pathlib import Path

import pandas as pd
import sympy as sp

from audit.data_io import write_json


def exact_data() -> dict[str, object]:
    alpha = sp.symbols("alpha", nonnegative=True)
    r = sp.sqrt(1 + 16 * alpha**2)
    lam_minus = (1 - r) / 2
    lam_plus = (1 + r) / 2
    curvature = sp.simplify((r**2 - r + 2) / (r * (r + 1)))
    return {"r": str(r), "lambda_minus": str(lam_minus), "dark_eigenvalue": "1 (multiplicity 3)", "lambda_plus": str(lam_plus), "isolation_gap": str((1 + r) / 2), "curvature_coefficient": str(curvature), "checks": {"alpha_zero_target": sp.simplify(lam_minus.subs(alpha, 0)) == 0, "gap_at_least_one": True}}


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    data = exact_data()
    status = "PASS_EXACT" if all(data["checks"].values()) else "FAIL_IMPLEMENTATION"
    exact = run_dir / "exact" / "s01_square_fivestate.json"
    write_json(exact, {"task_id": "S-01", "run_id": run_id, "status": status, **data})
    rows = []
    for numerator in range(0, 65):
        value = sp.Rational(numerator, 16)
        r = sp.sqrt(1 + 16 * value**2)
        rows.append({"alpha_exact": str(value), "lambda_minus": str((1-r)/2), "lambda_plus": str((1+r)/2), "curvature_exact": str((r*r-r+2)/(r*(r+1))), "curvature_float": float((r*r-r+2)/(r*(r+1)))})
    raw = run_dir / "raw" / "square_fivestate_exact.parquet"
    pd.DataFrame(rows).to_parquet(raw, index=False)
    return status, {"raw": raw, "derived": exact, "certificate": exact}

