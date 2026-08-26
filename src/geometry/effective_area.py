from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from scipy.integrate import quad

from audit.data_io import write_json, write_zarr
from geometry.crossover_core import effective_area, moire_length


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    grid = config["registry_grid"]
    rows: list[dict[str, float]] = []
    for radius in map(float, grid["radii"]):
        for threshold in map(float, grid["thresholds"]):
            for theta in map(float, grid["angles"]):
                xi = moire_length(radius, theta, threshold)
                numeric, quadrature_error = quad(
                    lambda r: 2.0 * math.pi * radius * math.sinh(r / radius),
                    0.0,
                    xi,
                    epsabs=1.0e-12,
                    epsrel=1.0e-12,
                )
                analytic = effective_area(radius, theta, threshold)
                rows.append({"R": radius, "threshold": threshold, "theta": theta, "xi": xi, "area_numeric": numeric, "area_analytic": analytic, "quadrature_error_bound": quadrature_error, "absolute_residual": abs(numeric - analytic)})
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "effective_area.zarr"
    write_zarr(raw, {column: frame[column].to_numpy() for column in frame.columns}, {"run_id": run_id, "task_id": "G-03"})
    derived = run_dir / "derived" / "effective_area_errors.parquet"
    frame[["R", "threshold", "theta", "quadrature_error_bound", "absolute_residual"]].to_parquet(derived, index=False)
    relative = (frame["absolute_residual"] / frame["area_analytic"].clip(lower=1.0e-300)).max()
    status = "PASS_CONVERGED" if float(relative) <= 5.0e-11 else "FAIL_IMPLEMENTATION"
    certificate = run_dir / "certificates" / "g03_effective_area.json"
    write_json(certificate, {"task_id": "G-03", "run_id": run_id, "status": status, "method": "adaptive quadrature of geodesic-polar area element", "maximum_relative_residual": float(relative)})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

