from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from audit.data_io import write_json


def _branches(group: pd.DataFrame) -> list[dict[str, float]]:
    ordered = group.sort_values("theta").reset_index(drop=True)
    records = []
    for index in range(1, len(ordered) - 1):
        center = float(ordered.loc[index, "score_M"])
        if center >= float(ordered.loc[index - 1, "score_M"]) and center >= float(ordered.loc[index + 1, "score_M"]):
            records.append(
                {
                    "theta": float(ordered.loc[index, "theta"]),
                    "score": center,
                    "X": float(ordered.loc[index, "X"]),
                    "gap": float(ordered.loc[index, "Delta"]),
                    "coherence": float(ordered.loc[index, "C_coh"]),
                    "tracking_overlap": float(ordered.loc[index, "tracking_overlap"]),
                }
            )
    return records


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["s22_curvature_born"]
    landscape = pd.read_parquet(run_dir / "derived" / "magic_landscape_flat.parquet")
    bifurcation = json.loads((run_dir / "certificates" / "s21_bifurcation_certificates.json").read_text(encoding="utf-8"))
    rows = []
    counts: dict[tuple[float, float], int] = {}
    for (curvature, w_value), group in landscape.groupby(["K", "w_over_t"]):
        branches = _branches(group)
        counts[(float(curvature), float(w_value))] = len(branches)
        for branch_index, branch in enumerate(branches, start=1):
            rows.append(
                {
                    "K": float(curvature),
                    "w_over_t": float(w_value),
                    "branch_index": branch_index,
                    **branch,
                }
            )
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "s22_branch_catalog.parquet"
    frame.to_parquet(raw, index=False)
    comparisons = []
    w_values = sorted(set(float(value) for value in landscape["w_over_t"]))
    negative_curvatures = sorted(value for value in set(float(value) for value in landscape["K"]) if value < 0.0)
    branch_excess = False
    for w_value in w_values:
        baseline_count = counts.get((0.0, w_value), 0)
        for curvature in negative_curvatures:
            curved_count = counts.get((curvature, w_value), 0)
            excess = curved_count > baseline_count
            branch_excess = branch_excess or excess
            comparisons.append(
                {
                    "w_over_t": w_value,
                    "K": curvature,
                    "euclidean_branch_count": baseline_count,
                    "curved_branch_count": curved_count,
                    "branch_count_increase": excess,
                    "interior_fold_certified": int(bifurcation.get("fold_candidate_count", 0)) > 0,
                }
            )
    derived = run_dir / "derived" / "curvature_born_branch_comparison.parquet"
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_parquet(derived, index=False)
    interior_fold = int(bifurcation.get("fold_candidate_count", 0)) > 0
    curvature_born = branch_excess and interior_fold
    decisive_no_birth = not branch_excess or bifurcation.get("status") == "PASS_CERTIFIED"
    status = "PASS_CERTIFIED" if decisive_no_birth and not curvature_born else "INCONCLUSIVE"
    certificate = run_dir / "certificates" / "s22_curvature_born_branch.json"
    write_json(
        certificate,
        {
            "task_id": "S-22",
            "run_id": run_id,
            "status": status,
            "same_preregistered_target": True,
            "bifurcation_certificate_status": bifurcation.get("status"),
            "any_branch_count_increase": branch_excess,
            "interior_fold_certified": interior_fold,
            "curvature_born_branch_detected": curvature_born,
            "conclusion": "NO_GENUINELY_CURVATURE_BORN_BRANCH_CERTIFIED" if not curvature_born else "CURVATURE_BORN_CANDIDATE_REQUIRES_BULK_CONFIRMATION",
            "boundary_entry_not_counted_as_birth": True,
            "scope": "same finite active-fiber target at K=0 and K<0",
            "reason_if_inconclusive": "A branch-count change occurred without a certified interior fold exclusion.",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
