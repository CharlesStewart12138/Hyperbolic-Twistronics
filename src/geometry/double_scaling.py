from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json, write_zarr


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    settings = config["double_scaling"]
    rows: list[dict[str, float]] = []
    for u in map(float, settings["u"]):
        target = math.asinh(1.0 / u)
        for alpha in map(float, settings["alpha"]):
            theta = 2.0 * alpha * u
            chi = math.sinh(alpha) / math.sin(alpha * u)
            scaled_length = math.asinh(chi)
            predicted_correction = alpha * alpha * math.sqrt(1.0 + u * u) / 6.0
            rows.append({"u": u, "alpha": alpha, "theta": theta, "chi": chi, "xi_over_R": scaled_length, "limit_xi_over_R": target, "absolute_error": abs(scaled_length - target), "predicted_O_alpha2_correction": predicted_correction, "normalized_length_G": u * scaled_length, "limit_G": u * target})
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "double_scaling.zarr"
    write_zarr(raw, {column: frame[column].to_numpy() for column in frame.columns}, {"run_id": run_id, "task_id": "G-05"})
    orders = []
    for u, group in frame.groupby("u"):
        selected = group.sort_values("alpha").head(4)
        slope, intercept = np.polyfit(np.log(selected["alpha"]), np.log(selected["absolute_error"]), 1)
        orders.append({"u": float(u), "convergence_order": float(slope), "log_prefactor": float(intercept)})
    order_frame = pd.DataFrame(orders)
    derived = run_dir / "derived" / "double_scaling_convergence.parquet"
    order_frame.to_parquet(derived, index=False)
    passed = bool(((order_frame["convergence_order"] > 1.94) & (order_frame["convergence_order"] < 2.06)).all())
    status = "PASS_CONVERGED" if passed else "FAIL_THEORY"
    certificate = run_dir / "certificates" / "g05_double_scaling.json"
    write_json(certificate, {"task_id": "G-05", "run_id": run_id, "status": status, "expected_order": 2, "observed_order_min": float(order_frame["convergence_order"].min()), "observed_order_max": float(order_frame["convergence_order"].max())})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

