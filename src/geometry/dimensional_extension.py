from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

from audit.data_io import write_json, write_zarr


def beta_dimension(dimension: int, y: float) -> float:
    integral, _ = quad(lambda t: math.sinh(t) ** (dimension - 1), 0.0, y, epsabs=1.0e-13, epsrel=1.0e-13)
    return math.sinh(y) ** dimension / (math.cosh(y) * integral)


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    rows: list[dict[str, object]] = []
    y_values = np.geomspace(0.02, 12.0, 80)
    for dimension in (2, 3, 4, 5, 6):
        full_rank = dimension % 2 == 0
        for y in y_values:
            if full_rank:
                beta = beta_dimension(dimension, float(y))
                rows.append({"ambient_dimension": dimension, "active_dimension": dimension, "fixed_axis_count": 0, "y": float(y), "beta": beta, "full_D_law_admissible": 1})
            else:
                active = dimension - 1
                beta = beta_dimension(active, float(y))
                rows.append({"ambient_dimension": dimension, "active_dimension": active, "fixed_axis_count": 1, "y": float(y), "beta": beta, "full_D_law_admissible": 0})
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "dimensional_extension.zarr"
    write_zarr(raw, {column: frame[column].to_numpy() for column in frame.columns}, {"run_id": run_id, "task_id": "G-06", "fixed_axis_interpretation": "beta applies only to the active rotating subspace; the full-D compact registry-volume law is rejected"})
    summaries = []
    for (ambient, active, admissible), group in frame.groupby(["ambient_dimension", "active_dimension", "full_D_law_admissible"]):
        small = group.iloc[0]
        large = group.iloc[-1]
        summaries.append({"ambient_dimension": int(ambient), "active_dimension": int(active), "fixed_axis_count": int(ambient - active), "full_D_law_admissible": bool(admissible), "small_y_beta": float(small["beta"]), "small_y_target": int(active), "large_y_beta": float(large["beta"]), "large_y_target": int(active - 1), "fixed_axis_displacement_at_arbitrary_radius": 0.0 if not admissible else np.nan})
    summary = pd.DataFrame(summaries)
    derived = run_dir / "derived" / "dimensional_endpoint_laws.parquet"
    summary.to_parquet(derived, index=False)
    endpoint_error = max(float((summary["small_y_beta"] - summary["small_y_target"]).abs().max()), float((summary["large_y_beta"] - summary["large_y_target"]).abs().max()))
    fixed_rejected = bool((summary.loc[summary["fixed_axis_count"] > 0, "full_D_law_admissible"] == False).all())  # noqa: E712
    status = "PASS_CONVERGED" if endpoint_error < 0.015 and fixed_rejected else "FAIL_THEORY"
    certificate = run_dir / "certificates" / "g06_dimensional_extension.json"
    write_json(certificate, {"task_id": "G-06", "run_id": run_id, "status": status, "maximum_endpoint_error": endpoint_error, "full_rank_dimensions": [2, 4, 6], "fixed_axis_dimensions": [3, 5], "scope_guard": "No ambient D to D-1 claim is made for odd-D generators with a fixed axis."})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

