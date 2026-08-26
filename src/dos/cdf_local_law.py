from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from dos.common import finish, retained_group, task_paths, weighted_cdf_distance


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw, derived, certificate = task_paths(run_dir, "d02_cdf_local_law")
    blocks = context["blocks"]
    groups = []
    for tower_id, tower in blocks.groupby("tower_id"):
        level = int(tower.level.max())
        subset = retained_group(blocks, str(tower_id), level)
        groups.append({
            "tower_id": str(tower_id), "level": level,
            "values": np.asarray(subset.adjacency_eigenvalue, dtype=float) / 8.0,
            "weights": np.asarray(subset.regular_multiplicity, dtype=float),
            "retained_dimension": int(np.sum(subset.regular_multiplicity)),
        })
    proxy_values = np.concatenate([row["values"] for row in groups])
    proxy_weights = np.concatenate([
        row["weights"] / np.sum(row["weights"]) / len(groups) for row in groups
    ])
    records = []
    for tower_id, level in blocks[["tower_id", "level"]].drop_duplicates().itertuples(index=False):
        subset = retained_group(blocks, str(tower_id), int(level))
        kappa = weighted_cdf_distance(
            np.asarray(subset.adjacency_eigenvalue) / 8.0,
            np.asarray(subset.regular_multiplicity), proxy_values, proxy_weights,
        )
        records.append({
            "tower_id": str(tower_id), "level": int(level),
            "quotient_order": int(subset.quotient_order.iloc[0]),
            "retained_dimension": int(np.sum(subset.regular_multiplicity)),
            "kappa_N": kappa, "kernel": "NONE",
            "reference": "equal-weight empirical CDF barycenter of the latest certified level of each of three non-Abelian towers",
        })
    pairwise = []
    for left, right in combinations(groups, 2):
        distance = weighted_cdf_distance(left["values"], left["weights"], right["values"], right["weights"])
        pairwise.append({
            "cover_a": f"{left['tower_id']}_L{left['level']}",
            "cover_b": f"{right['tower_id']}_L{right['level']}",
            "kolmogorov_distance": distance,
        })
    pd.DataFrame(records).to_parquet(raw / "retained_cdf_errors.parquet", index=False)
    pd.DataFrame(pairwise).to_parquet(raw / "latest_level_pairwise_cdf.parquet", index=False)
    pd.DataFrame(records).to_parquet(derived, index=False)
    limit = float(config["cdf_local_law"]["cross_tower_kolmogorov_limit"])
    maximum = max(row["kolmogorov_distance"] for row in pairwise)
    status = "PASS_CONVERGED" if len(groups) >= 3 and maximum <= limit else "INCONCLUSIVE"
    finish(certificate, {
        "kernel_independent": True, "smoothing_used": False,
        "certified_nonabelian_tower_count": len(groups),
        "latest_level_pairwise_records": pairwise,
        "maximum_latest_level_pairwise_kolmogorov_distance": maximum,
        "acceptance_limit": limit,
        "scope": "finite retained-sector CDF convergence diagnostic; not by itself an unsmoothed density theorem",
    }, status, run_id, "D-02")
    context["d02_records"] = pd.DataFrame(records)
    context["d02_status"] = status
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
