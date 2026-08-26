from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import sympy as sp

from audit.data_io import write_json, write_zarr
from geometry.crossover_core import beta_m, crossover_f, effective_area, flow_rhs, mismatch_chi


def _exact_identities() -> dict[str, object]:
    x = sp.symbols("x", positive=True)
    f = 2 / (1 + sp.sqrt(1 + x**2))
    beta = 1 + 1 / sp.sqrt(1 + x**2)
    return {
        "flow_residual": str(sp.simplify(x * sp.diff(f, x) + 2 * f * (1 - f) / (2 - f))),
        "amplitude_beta_residual": str(sp.simplify(beta - 2 / (2 - f))),
        "transformed_residual": str(sp.simplify((2 / f - 1) ** 2 - 1 - x**2)),
    }


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    grid = config["registry_grid"]
    rows: list[dict[str, float]] = []
    h = 2.0e-5
    for radius in map(float, grid["radii"]):
        for threshold in map(float, grid["thresholds"]):
            for theta in map(float, grid["angles"]):
                s = abs(math.sin(theta / 2.0))
                chi = mismatch_chi(radius, theta, threshold)
                f = crossover_f(chi)
                area = effective_area(radius, theta, threshold)
                reference = math.pi * radius * radius * chi * chi
                def log_area(log_s: float) -> float:
                    local_chi = math.sinh(threshold / (2 * radius)) / math.exp(log_s)
                    return math.log(2 * math.pi * radius * radius * local_chi * local_chi / (math.sqrt(1 + local_chi * local_chi) + 1))
                beta_numeric = -(log_area(math.log(s) + h) - log_area(math.log(s) - h)) / (2 * h)
                fp = crossover_f(chi * math.exp(h))
                fm = crossover_f(chi * math.exp(-h))
                flow_numeric = (fp - fm) / (2 * h)
                transformed = (2.0 / f - 1.0) ** 2 - 1.0
                rows.append({"R": radius, "threshold": threshold, "theta": theta, "chi": chi, "area": area, "reference_area": reference, "normalized_area": area / reference, "F": f, "collapse_residual": area / reference - f, "beta_numeric": beta_numeric, "beta_exact": beta_m(chi), "beta_residual": beta_numeric - beta_m(chi), "flow_numeric": flow_numeric, "flow_rhs": flow_rhs(f), "flow_residual": flow_numeric - flow_rhs(f), "transformed_lhs": transformed, "transformed_rhs": chi * chi, "transformed_relative_residual": (transformed - chi * chi) / max(1.0, chi * chi)})
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "crossover.zarr"
    write_zarr(raw, {column: frame[column].to_numpy() for column in frame.columns}, {"run_id": run_id, "task_id": "G-04"})
    derived = run_dir / "derived" / "flow_residual.parquet"
    frame[["R", "threshold", "theta", "chi", "collapse_residual", "beta_residual", "flow_residual", "transformed_relative_residual"]].to_parquet(derived, index=False)
    exact = _exact_identities()
    max_residual = float(frame[["collapse_residual", "beta_residual", "flow_residual", "transformed_relative_residual"]].abs().max().max())
    exact_pass = all(value == "0" for value in exact.values())
    status = "PASS_CONVERGED" if exact_pass and max_residual <= 2.0e-8 else "FAIL_IMPLEMENTATION"
    certificate = run_dir / "certificates" / "g04_crossover.json"
    write_json(certificate, {"task_id": "G-04", "run_id": run_id, "status": status, "symbolic_identities": exact, "symbolic_status": "PASS_EXACT" if exact_pass else "FAIL_IMPLEMENTATION", "maximum_numerical_residual": max_residual, "finite_difference_log_step": h})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

