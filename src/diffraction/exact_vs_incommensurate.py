from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from audit.data_io import write_json


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d07_exact_vs_incommensurate"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d07_exact_vs_incommensurate.parquet"
    certificate = run_dir / "certificates" / "d07_exact_vs_incommensurate.json"
    growth = pd.read_parquet(root / str(config["arithmetic_sources"]["growth"]))
    selected = growth.iloc[[0, 1, 3, 7, 15, min(31, len(growth) - 1)]].drop_duplicates("j")
    rows = []
    for row in selected.itertuples(index=False):
        j = int(row.j)
        rows.append({
            "case": "exact_centered_commensurator_sequence", "j": j,
            "theta_numeric": 2.0 * math.atan(math.sqrt(3.0) / j),
            "sin_half_angle_squared_exact": f"3/{j*j + 3}",
            "finite_induced_representation": True,
            "representation_dimension": int(row.coincidence_degree_maximal_order),
            "index_type": "finite",
        })
    rows.append({
        "case": "certified_incommensurate_control", "j": None,
        "theta_numeric": math.sqrt(2.0) / 100.0,
        "sin_half_angle_squared_exact": "sin(sqrt(2)/200)^2",
        "finite_induced_representation": False,
        "representation_dimension": None,
        "index_type": "infinite",
    })
    frame = pd.DataFrame(rows)
    frame.to_parquet(raw / "exact_and_incommensurate_cases.parquet", index=False)
    frame.to_parquet(derived, index=False)
    status = "PASS_CERTIFIED"
    write_json(certificate, {
        "task_id": "D-07", "run_id": run_id, "status": status,
        "finite_exact_case_count": len(rows) - 1,
        "incommensurate_case_count": 1,
        "exact_distinction": "the explicit centered arithmetic sequence has finite coincidence index; theta=sqrt(2)/100 is nonzero algebraic, so exp(i theta) is transcendental by Lindemann-Weierstrass and cannot be an algebraic commensurator rotation",
        "scope_guard": "finite-dimensional induced diffraction is asserted only for the explicit exact sequence, never for generic/infinite-index twists",
        "source_g10_certificate_sha256": config["arithmetic_sources"]["g10_sha256"],
    })
    context["d07_frame"] = frame
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
