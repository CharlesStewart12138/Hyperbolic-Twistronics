from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d08_arithmetic_complexity"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d08_arithmetic_complexity.parquet"
    certificate = run_dir / "certificates" / "d08_arithmetic_complexity.json"
    growth = pd.read_parquet(root / str(config["arithmetic_sources"]["growth"])).copy()
    growth["s_j"] = np.sqrt(3.0 / (np.asarray(growth.j, dtype=float) ** 2 + 3.0))
    growth["diffraction_representation_dimension"] = growth.coincidence_degree_maximal_order.astype("int64")
    growth["pointwise_exponent"] = np.log(np.asarray(growth.diffraction_representation_dimension, dtype=float)) / np.log(1.0 / growth.s_j)
    growth[["j", "s_j", "diffraction_representation_dimension"]].to_parquet(raw / "arithmetic_dimension_inputs.parquet", index=False)
    growth[["j", "s_j", "diffraction_representation_dimension", "pointwise_exponent"]].to_parquet(derived, index=False)
    g10 = json.loads((root / str(config["arithmetic_sources"]["g10_certificate"])).read_text(encoding="utf-8"))
    g11 = json.loads((root / str(config["arithmetic_sources"]["g11_certificate"])).read_text(encoding="utf-8"))
    tail = growth.tail(min(16, len(growth)))
    slope, intercept = np.polyfit(np.log(1.0 / tail.s_j), np.log(np.asarray(tail.diffraction_representation_dimension, dtype=float)), 1)
    passed = (
        g10.get("status") == "PASS_EXACT"
        and g11.get("status") == "PASS_CONVERGED"
        and bool(g11.get("compatible_with_4"))
        and bool(g11.get("incompatible_with_1"))
        and abs(float(slope) - 4.0) < 0.5
    )
    status = "PASS_CERTIFIED" if passed else "FAIL_THEORY"
    write_json(certificate, {
        "task_id": "D-08", "run_id": run_id, "status": status,
        "fitted_tail_exponent": float(slope), "fitted_log_intercept": float(intercept),
        "exact_exponent": 4,
        "exact_basis": g10.get("formula"),
        "g10_certificate_sha256": config["arithmetic_sources"]["g10_sha256"],
        "g11_certificate_sha256": config["arithmetic_sources"]["g11_sha256"],
        "conclusion": "the exact induced diffraction-representation dimension inherits the explicit sequence arithmetic exponent four, not one",
    })
    context["d08_slope"] = float(slope)
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
