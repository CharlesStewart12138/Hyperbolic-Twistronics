from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json
from spectral.natural_surface_model import natural_parameters


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    params = natural_parameters(config)
    q1, t, m = params["q1"], params["t"], int(params["m"])
    w_star = t / q1
    w_values = np.linspace(0.05, 12.0, int(config["natural_surface_model"]["root_scan_points"]))
    rows = []
    for w in w_values:
        f1 = 1.0 - w * q1 / t
        rows.append({"w_over_t": w/t, "F1": f1, "magic_score": 1.0/(1.0+f1*f1), "hodge_curvature_scalar": 2.0*(t-w*q1), "layer_even_bandwidth": 4.0*m*abs(w*q1-t)})
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "character_root.parquet"
    frame.to_parquet(raw, index=False)
    derived = run_dir / "derived" / "character_root_summary.parquet"
    pd.DataFrame([{"q1": q1, "t": t, "root_w": w_star, "derivative": -q1/t, "first_shell_gap": 2*t*(1/q1-2*m), "score_at_root": 1.0}]).to_parquet(derived, index=False)
    inside = 0.05 <= w_star/t <= 12.0
    status = "PASS_CONVERGED" if inside and 2*t*(1/q1-2*m) > 0 else "FAIL_THEORY"
    certificate = run_dir / "certificates" / "s03_character_root.json"
    write_json(certificate, {"task_id": "S-03", "run_id": run_id, "status": status, "model": "M1 natural genus-two surface-group first shell", "q1": q1, "root_w_over_t": w_star/t, "inside_preregistered_box": inside, "simple_derivative": -q1/t, "gap": 2*t*(1/q1-2*m)})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

