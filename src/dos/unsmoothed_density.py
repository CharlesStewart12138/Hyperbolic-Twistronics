from __future__ import annotations

from pathlib import Path

import pandas as pd

from dos.common import finish, task_paths


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw, derived, certificate = task_paths(run_dir, "d04_unsmoothed_density")
    candidates = []
    for (tower_id, level), subset in context["blocks"].groupby(["tower_id", "level"]):
        candidates.append({
            "tower_id": str(tower_id), "level": int(level),
            "retained_only": True,
            "uniform_regular_density_hypothesis_certified": False,
            "vanishing_broadening_prerequisite": context.get("d03_status"),
            "atomic_finite_cover_density_claimed": False,
            "admitted_to_unsmoothed_theorem": False,
        })
    frame = pd.DataFrame(candidates)
    frame.to_parquet(raw / "regularity_gate_candidates.parquet", index=False)
    frame.to_parquet(derived, index=False)
    admitted = int(frame.admitted_to_unsmoothed_theorem.sum())
    status = "PASS_CERTIFIED" if admitted > 0 else "INCONCLUSIVE"
    finish(certificate, {
        "admitted_sector_count": admitted,
        "atomic_finite_cover_density_claim": False,
        "guard_enforced": True,
        "reason_if_inconclusive": None if admitted else "no retained sector has both the required uniform regularity certificate and a passed D-03 asymptotic gate",
        "scientific_conclusion": "finite covers remain atomic; no unsmoothed density convergence is asserted",
    }, status, run_id, "D-04")
    context["d04_status"] = status
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
