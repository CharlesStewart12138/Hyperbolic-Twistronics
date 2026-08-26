from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json
from geometry.build_orbit_and_frames import (
    centered_rotation,
    hyperbolic_distance,
    lorentz_boost,
)
from geometry.crossover_core import displacement


def _arb_certificate(rows: list[dict[str, float]], bits: int) -> dict[str, object]:
    from flint import arb, ctx

    previous = ctx.prec
    balls: list[str] = []
    contains_zero = True
    try:
        ctx.prec = bits
        for row in rows:
            radius = arb(str(row["R"]))
            theta = arb(str(row["theta"]))
            radial = arb(str(row["r"]))
            sh = (radial / radius).sinh()
            s = (theta / 2).sin()
            closed = 2 * radius * (sh * s).asinh()
            # Independent hyperboloid inner product after a centered rotation.
            x0 = radius * (radial / radius).cosh()
            x1 = radius * sh
            rotated_inner = -(x0 * x0) + x1 * x1 * theta.cos()
            coordinate = radius * (-rotated_inner / (radius * radius)).acosh()
            residual = coordinate - closed
            balls.append(str(residual))
            contains_zero = contains_zero and bool(residual.contains(arb(0)))
    finally:
        ctx.prec = previous
    return {
        "backend": "python-flint Arb",
        "precision_bits": bits,
        "all_residual_balls_contain_zero": contains_zero,
        "residual_balls": balls,
    }


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    grid = config["distance_grid"]
    rows: list[dict[str, float]] = []
    for radius in map(float, grid["radii"]):
        origin = np.array([radius, 0.0, 0.0])
        for theta in map(float, grid["angles"]):
            rotation = centered_rotation(theta)
            for radial_ratio in map(float, grid["radial_ratios"]):
                radial = radius * radial_ratio
                base_point = lorentz_boost(radial_ratio, 0.37) @ origin
                for center_rapidity in map(float, grid["center_rapidities"]):
                    h = lorentz_boost(center_rapidity, 1.13)
                    point = h @ base_point
                    about_center = h @ rotation @ np.linalg.inv(h)
                    coordinate = hyperbolic_distance(point, about_center @ point, radius)
                    closed = displacement(radius, theta, radial)
                    rows.append(
                        {
                            "R": radius,
                            "theta": theta,
                            "r": radial,
                            "center_rapidity": center_rapidity,
                            "coordinate_distance": coordinate,
                            "closed_form_distance": closed,
                            "absolute_residual": abs(coordinate - closed),
                        }
                    )
    raw = run_dir / "raw" / "distance_identity.parquet"
    derived = run_dir / "derived" / "distance_identity_residuals.parquet"
    pd.DataFrame(rows).to_parquet(raw, index=False)
    summary = (
        pd.DataFrame(rows)
        .groupby(["R", "theta"], as_index=False)["absolute_residual"]
        .max()
    )
    summary.to_parquet(derived, index=False)
    representative: list[dict[str, float]] = []
    for radius in map(float, grid["radii"]):
        for theta in map(float, grid["angles"]):
            for radial_ratio in map(float, grid["radial_ratios"]):
                representative.append({"R": radius, "theta": theta, "r": radius * radial_ratio})
    arb_data = _arb_certificate(representative, int(config["certification"]["arb_bits"]))
    tolerance = float(config["certification"]["numerical_tolerance"])
    maximum = max(row["absolute_residual"] for row in rows)
    passed = maximum <= tolerance and bool(arb_data["all_residual_balls_contain_zero"])
    status = "PASS_CERTIFIED" if passed else "FAIL_IMPLEMENTATION"
    certificate = run_dir / "certificates" / "g01_distance_identity.json"
    write_json(
        certificate,
        {
            "task_id": "G-01",
            "run_id": run_id,
            "status": status,
            "identity": "sinh(d_theta(r)/(2R)) = sinh(r/R)|sin(theta/2)|",
            "independent_method": "Lorentz-hyperboloid isometry conjugated to three twist centers",
            "sample_count": len(rows),
            "maximum_absolute_residual": maximum,
            "numerical_tolerance": tolerance,
            "arb": arb_data,
            "raw_output": raw.relative_to(run_dir.parent.parent).as_posix(),
            "derived_output": derived.relative_to(run_dir.parent.parent).as_posix(),
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

