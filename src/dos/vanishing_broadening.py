from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dos.common import finish, task_paths


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw, derived, certificate = task_paths(run_dir, "d03_vanishing_broadening")
    frame = context["d02_records"].copy()
    alpha = float(config["vanishing_broadening"]["alpha"])
    floor = np.maximum(1.0 / np.asarray(frame.retained_dimension, dtype=float), np.asarray(frame.kappa_N, dtype=float))
    frame["eta_N"] = floor ** (1.0 / (alpha + 1.0))
    frame["kappa_over_eta"] = frame.kappa_N / frame.eta_N
    frame["eta_to_alpha"] = frame.eta_N ** alpha
    frame["combined_term"] = frame.kappa_over_eta + frame.eta_to_alpha
    frame["terms_stored_separately"] = True
    frame.to_parquet(raw / "vanishing_broadening_terms.parquet", index=False)
    frame.to_parquet(derived, index=False)
    minimum = int(config["vanishing_broadening"]["minimum_levels_per_tower"])
    diagnostics = []
    enough_depth = True
    for tower_id, group in frame.sort_values("level").groupby("tower_id"):
        values = np.asarray(group.combined_term, dtype=float)
        levels = len(group)
        monotone = bool(levels > 1 and np.all(np.diff(values) < 0.0))
        diagnostics.append({
            "tower_id": str(tower_id), "level_count": levels,
            "minimum_required": minimum, "combined_term_strictly_decreasing": monotone,
        })
        enough_depth &= levels >= minimum and monotone
    status = "PASS_CONVERGED" if context.get("d02_status") == "PASS_CONVERGED" and enough_depth else "INCONCLUSIVE"
    finish(certificate, {
        "alpha": alpha, "eta_rule": "max(kappa_N,1/retained_dimension)^(1/(alpha+1))",
        "all_terms_stored_separately": True, "tower_depth_diagnostics": diagnostics,
        "minimum_levels_per_tower": minimum,
        "reason_if_inconclusive": None if status == "PASS_CONVERGED" else "fewer than three within-tower levels prevent a defensible asymptotic vanishing-broadening claim",
    }, status, run_id, "D-03")
    context["d03_terms"] = frame
    context["d03_status"] = status
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
