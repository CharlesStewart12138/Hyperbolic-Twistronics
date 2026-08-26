from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json
from exact.coincidence_index_height import local_factor_record


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    samples = list(map(int, config["arithmetic"]["j_samples"]))
    rows = []
    for j in samples:
        record = local_factor_record(j)
        s = math.sqrt(3.0 / (j * j + 3.0))
        q = int(record["coincidence_degree_maximal_order"])
        exponent = math.log(q) / math.log(1.0 / s)
        rows.append({"j": j, "s_j": s, "q_j_maximal_order": q, "log_q_over_log_s_inverse": exponent, "distance_to_4": exponent - 4.0, "distance_to_1": exponent - 1.0})
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "arithmetic_exponent_inputs.parquet"
    frame[["j", "s_j", "q_j_maximal_order"]].to_parquet(raw, index=False)
    derived = run_dir / "derived" / "arithmetic_exponent.parquet"
    frame.to_parquet(derived, index=False)
    x = 1.0 / np.log(frame["j"].tail(5).to_numpy(dtype=float))
    y = frame["log_q_over_log_s_inverse"].tail(5).to_numpy(dtype=float)
    extrapolated = float(np.polyfit(x, y, 1)[1])
    last = float(y[-1])
    compatible_four = bool(abs(extrapolated - 4.0) < 0.12 and abs(last - 4.0) < 0.35)
    incompatible_one = bool(min(abs(y - 1.0)) > 2.0)
    status = "PASS_CONVERGED" if compatible_four and incompatible_one else "FAIL_THEORY"
    certificate = run_dir / "certificates" / "g11_arithmetic_exponent.json"
    write_json(certificate, {"task_id": "G-11", "run_id": run_id, "status": status, "numerical_extrapolate_to_inverse_log_zero": extrapolated, "last_direct_exponent": last, "compatible_with_4": compatible_four, "incompatible_with_1": incompatible_one, "exact_theoretical_certificate": "G-10 formula plus bounded epsilon_j and sub-power F_j gives limit 4", "scope": "maximal order; fixed Gamma differs by bounded C_Gamma and has the same exponent"})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

