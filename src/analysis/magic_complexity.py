from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["g14_magic_complexity"]
    g10 = json.loads((root / str(config["phase_g_source"]["g10_certificate"])).read_text(encoding="utf-8"))
    g11 = json.loads((root / str(config["phase_g_source"]["g11_certificate"])).read_text(encoding="utf-8"))
    sampling = pd.read_parquet(run_dir / "derived" / "magic_subsequence_sampling.parquet")
    growth = pd.read_parquet(root / str(config["phase_g_source"]["growth"]))
    q_column = "coincidence_degree_maximal_order"
    merged = sampling.merge(growth[["j", q_column, "product_identity"]], on="j", how="left", validate="one_to_one")
    if merged[q_column].isna().any():
        raise RuntimeError("frozen G-10 growth table does not contain every preregistered G-14 index")
    merged["sqrt_omega_log_q"] = np.sqrt(merged["omega_j"].astype(float)) * np.log(merged[q_column].astype(float))
    merged["target_4c"] = 4.0 * merged["c_nu"].astype(float)
    merged["relative_error_to_4c"] = (merged["sqrt_omega_log_q"] - merged["target_4c"]) / merged["target_4c"]
    merged["theorem_matched_residual_over_lambda"] = (
        np.log(merged[q_column].astype(float)) - 4.0 * merged["lambda_arithmetic"].astype(float)
    ) / merged["lambda_arithmetic"].astype(float)
    raw = run_dir / "raw" / "g14_magic_complexity_inputs.parquet"
    merged[["j", q_column, "product_identity", "omega_j", "c_nu", "lambda_arithmetic"]].to_parquet(raw, index=False)
    derived = run_dir / "derived" / "magic_complexity.parquet"
    merged.to_parquet(derived, index=False)

    tail_count = int(task["tail_count"])
    tail = merged.tail(tail_count)
    x = 1.0 / np.log(tail["j"].to_numpy(dtype=float))
    y = tail["sqrt_omega_log_q"].to_numpy(dtype=float)
    extrapolated = float(np.polyfit(x, y, 1)[1])
    target = float(4.0 * merged["c_nu"].iloc[0])
    extrapolated_relative_error = abs(extrapolated - target) / target
    final_relative_error = abs(float(y[-1]) - target) / target
    acceptance = task["acceptance"]
    exact_formula = bool(merged["product_identity"].all())
    passed = (
        g10.get("status") == acceptance["g10_status_required"]
        and g11.get("status") == acceptance["g11_status_required"]
        and exact_formula
        and extrapolated_relative_error <= float(acceptance["extrapolated_relative_error_maximum"])
        and final_relative_error <= float(acceptance["final_direct_relative_error_maximum"])
    )
    status = "PASS_CONVERGED" if passed else "INCONCLUSIVE"
    certificate = run_dir / "certificates" / "g14_magic_complexity.json"
    write_json(
        certificate,
        {
            "task_id": "G-14",
            "run_id": run_id,
            "status": status,
            "theorem_equations": task["theorem_equations"],
            "source_g10_status": g10.get("status"),
            "source_g11_status": g11.get("status"),
            "all_exact_product_identities": exact_formula,
            "target_4c": target,
            "tail_count": tail_count,
            "extrapolated_sqrt_omega_log_q": extrapolated,
            "extrapolated_relative_error": extrapolated_relative_error,
            "last_direct_value": float(y[-1]),
            "last_direct_relative_error": final_relative_error,
            "fixed_group_scope": "Only the frozen bounded C_Gamma comparison is inherited; no exact arbitrary-Gamma degree is fabricated.",
            "estimator": task["estimator"],
            "reason_if_inconclusive": "The preregistered finite-scale coefficient criteria did not both close; exact inputs and residuals remain saved.",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
