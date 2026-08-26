from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from audit.data_io import write_json, write_zarr
from geometry.crossover_core import displacement, moire_length


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    grid = config["registry_grid"]
    rows: list[dict[str, float]] = []
    for radius in map(float, grid["radii"]):
        for threshold in map(float, grid["thresholds"]):
            for theta in map(float, grid["angles"]):
                analytic = moire_length(radius, theta, threshold)
                upper = max(radius, 1.25 * analytic + threshold)
                numeric = brentq(
                    lambda r: displacement(radius, theta, r) - threshold,
                    0.0,
                    upper,
                    xtol=1.0e-13,
                    rtol=1.0e-14,
                )
                rows.append(
                    {
                        "R": radius,
                        "threshold": threshold,
                        "theta": theta,
                        "xi_numeric": numeric,
                        "xi_analytic": analytic,
                        "registry_residual": displacement(radius, theta, numeric) - threshold,
                        "absolute_error": abs(numeric - analytic),
                    }
                )
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "moire_length.zarr"
    write_zarr(raw, {column: frame[column].to_numpy() for column in frame.columns}, {"run_id": run_id, "task_id": "G-02"})
    derived = run_dir / "derived" / "moire_length_errors.parquet"
    frame[["R", "threshold", "theta", "registry_residual", "absolute_error"]].to_parquet(derived, index=False)
    tolerance = float(config["certification"]["numerical_tolerance"])
    maximum = float(frame["absolute_error"].max())
    status = "PASS_CONVERGED" if maximum <= tolerance else "FAIL_IMPLEMENTATION"
    certificate = run_dir / "certificates" / "g02_moire_length.json"
    write_json(certificate, {"task_id": "G-02", "run_id": run_id, "status": status, "solver": "scipy.optimize.brentq", "maximum_absolute_error": maximum, "tolerance": tolerance})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

