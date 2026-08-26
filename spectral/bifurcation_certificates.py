from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["s21_bifurcation"]
    landscape = pd.read_parquet(run_dir / "derived" / "magic_landscape_flat.parquet")
    derivative_rows = []
    stationary_rows = []
    for (curvature, w_value), group in landscape.groupby(["K", "w_over_t"]):
        ordered = group.sort_values("theta")
        theta = ordered["theta"].to_numpy(dtype=float)
        score = ordered["score_M"].to_numpy(dtype=float)
        coordinate = (theta - theta[0]) / (theta[-1] - theta[0])
        first = np.gradient(score, coordinate, edge_order=2)
        second = np.gradient(first, coordinate, edge_order=2)
        third = np.gradient(second, coordinate, edge_order=2)
        for index in range(len(theta)):
            derivative_rows.append(
                {
                    "K": curvature,
                    "w_over_t": w_value,
                    "theta": theta[index],
                    "score": score[index],
                    "F_dM_du": first[index],
                    "F_theta_d2M_du2": second[index],
                    "F_theta_theta_d3M_du3": third[index],
                }
            )
        for index in range(1, len(theta) - 1):
            sign_change = first[index - 1] > 0.0 and first[index + 1] < 0.0
            near_stationary = abs(first[index]) <= float(task["stationary_derivative_tolerance"])
            if not (sign_change or near_stationary):
                continue
            stationary_rows.append(
                {
                    "K": curvature,
                    "w_over_t": w_value,
                    "theta": theta[index],
                    "F": first[index],
                    "F_theta": second[index],
                    "F_theta_theta": third[index],
                    "fold_candidate": abs(second[index]) <= float(task["fold_second_derivative_tolerance"]),
                    "cusp_candidate": abs(second[index]) <= float(task["fold_second_derivative_tolerance"])
                    and abs(third[index]) <= float(task["cusp_third_derivative_margin"]),
                }
            )
    derivatives = pd.DataFrame(derivative_rows)
    stationary = pd.DataFrame(stationary_rows)
    raw = run_dir / "raw" / "s21_landscape_derivatives.parquet"
    derived = run_dir / "derived" / "bifurcation_diagnostics.parquet"
    derivatives.to_parquet(raw, index=False)
    stationary.to_parquet(derived, index=False)
    if stationary.empty:
        nonfold_margin = float("inf")
        fold_count = 0
        cusp_count = 0
    else:
        nonfold_margin = float(stationary["F_theta"].abs().min())
        fold_count = int(stationary["fold_candidate"].sum())
        cusp_count = int(stationary["cusp_candidate"].sum())
    no_bifurcations_certified = (
        fold_count == 0
        and cusp_count == 0
        and nonfold_margin >= float(task["certified_nonfold_second_derivative_margin"])
    )
    status = "PASS_CERTIFIED" if no_bifurcations_certified else "INCONCLUSIVE"
    certificate = run_dir / "certificates" / "s21_bifurcation_certificates.json"
    write_json(
        certificate,
        {
            "task_id": "S-21",
            "run_id": run_id,
            "status": status,
            "stationary_point_count": len(stationary),
            "fold_candidate_count": fold_count,
            "cusp_candidate_count": cusp_count,
            "minimum_nonfold_second_derivative_margin": nonfold_margin if np.isfinite(nonfold_margin) else None,
            "physical_box_conclusion": "NO_CERTIFIED_FOLD_OR_CUSP" if no_bifurcations_certified else "NO_EVENT_PROMOTED; DERIVATIVE_EXCLUSION_INCONCLUSIVE",
            "model_changed_after_scan": False,
            "acceptance": task,
            "scope": "preregistered finite ARO-3B operational landscape only",
            "reason_if_inconclusive": "No fold/cusp is asserted, but the frozen derivative margin did not certify exclusion throughout the sampled physical box.",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
