from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json
from exact.coincidence_index_height import local_factor_record


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    alpha = float(config["arithmetic"]["alpha_registry"])
    b = math.sinh(alpha)
    samples = list(map(int, config["arithmetic"]["j_samples"]))
    rows = []
    for j in samples:
        q = int(local_factor_record(j)["coincidence_degree_maximal_order"])
        s = math.sqrt(3.0 / (j * j + 3.0))
        xi_over_r = math.asinh(b / s)
        # Maximal-order normalization A0=pi*R^2/12 gives cosh(r_sc/R)-1=q/24.
        rsc_over_r = math.acosh(1.0 + q / 24.0)
        rows.append({"j": j, "q_j_maximal_order": q, "xi_over_R": xi_over_r, "r_sc_over_R": rsc_over_r, "Delta_over_xi": (rsc_over_r - xi_over_r) / xi_over_r, "r_sc_over_xi": rsc_over_r / xi_over_r})
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "radial_locking.parquet"
    frame.to_parquet(raw, index=False)
    fits = {}
    x = 1.0 / np.log(frame["j"].tail(5).to_numpy(dtype=float))
    for column, target in (("Delta_over_xi", 3.0), ("r_sc_over_xi", 4.0)):
        y = frame[column].tail(5).to_numpy(dtype=float)
        fits[column] = {"target": target, "extrapolated": float(np.polyfit(x, y, 1)[1]), "last": float(y[-1])}
    derived = run_dir / "derived" / "radial_locking_extrapolation.parquet"
    pd.DataFrame([{"observable": key, **value} for key, value in fits.items()]).to_parquet(derived, index=False)
    passed = abs(fits["Delta_over_xi"]["extrapolated"] - 3.0) < 0.12 and abs(fits["r_sc_over_xi"]["extrapolated"] - 4.0) < 0.12
    status = "PASS_CONVERGED" if passed else "FAIL_THEORY"
    certificate = run_dir / "certificates" / "g12_radial_locking.json"
    write_json(certificate, {"task_id": "G-12", "run_id": run_id, "status": status, "alpha_registry": alpha, "fits": fits, "area_normalization": "maximal-order A0=pi R^2/12", "fixed_group_scope": "bounded comparison C_Gamma leaves the limiting radial ratios unchanged"})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

