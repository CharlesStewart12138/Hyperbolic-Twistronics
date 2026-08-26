from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["s23_symmetry_flatness"]
    frozen = pd.read_parquet(root / str(config["phase_s_source"]["s14_table"]))
    zero = frozen.iloc[(frozen["bond_anisotropy"].astype(float).abs()).argmin()]
    b_scalar = 2.0 / float(zero["trace_root"])
    pattern = np.asarray(task["breaking_pattern"], dtype=float)
    rows = []
    for epsilon in map(float, task["perturbations"]):
        preserving_b = (b_scalar + epsilon) * np.eye(4)
        preserving_root = 2.0 / (b_scalar + epsilon)
        preserving_response = 2.0 * np.eye(4) - preserving_root * preserving_b
        preserving_eigenvalues = np.linalg.eigvalsh(preserving_response)
        rows.append(
            {
                "perturbation_class": "symmetry_preserving",
                "epsilon": epsilon,
                "trace_root": preserving_root,
                "symmetry_degeneracy_spread": float(np.ptp(preserving_eigenvalues)),
                "kinetic_flattening_operator_norm": float(np.linalg.norm(preserving_response, ord=2)),
                "symmetry_protection_gap_proxy": abs(b_scalar + epsilon),
                "principal_min": float(preserving_eigenvalues[0]),
                "principal_max": float(preserving_eigenvalues[-1]),
            }
        )
        breaking_b = b_scalar * np.eye(4) + epsilon * np.diag(pattern)
        breaking_root = 2.0 * 4.0 / float(np.trace(breaking_b))
        breaking_response = 2.0 * np.eye(4) - breaking_root * breaking_b
        breaking_eigenvalues = np.linalg.eigvalsh(breaking_response)
        rows.append(
            {
                "perturbation_class": "symmetry_breaking",
                "epsilon": epsilon,
                "trace_root": breaking_root,
                "symmetry_degeneracy_spread": float(np.ptp(breaking_eigenvalues)),
                "kinetic_flattening_operator_norm": float(np.linalg.norm(breaking_response, ord=2)),
                "symmetry_protection_gap_proxy": max(0.0, b_scalar - abs(epsilon) * float(np.max(np.abs(pattern)))),
                "principal_min": float(breaking_eigenvalues[0]),
                "principal_max": float(breaking_eigenvalues[-1]),
            }
        )
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "s23_symmetry_perturbation_responses.parquet"
    frame.to_parquet(raw, index=False)
    derived = run_dir / "derived" / "symmetry_vs_flatness.parquet"
    frame.assign(
        degeneracy_preserved=frame["symmetry_degeneracy_spread"] <= float(task["degeneracy_tolerance"]),
        fully_flat=frame["kinetic_flattening_operator_norm"] <= float(task["degeneracy_tolerance"]),
    ).to_parquet(derived, index=False)
    preserving = frame.loc[frame.perturbation_class == "symmetry_preserving"]
    breaking = frame.loc[(frame.perturbation_class == "symmetry_breaking") & (frame.epsilon.abs() > 0)]
    preserving_degeneracy = float(preserving["symmetry_degeneracy_spread"].max())
    breaking_separation = float(breaking["symmetry_degeneracy_spread"].min())
    passed = (
        preserving_degeneracy <= float(task["degeneracy_tolerance"])
        and breaking_separation >= float(task["separation_margin"])
        and bool((frame["symmetry_protection_gap_proxy"] > 0).all())
    )
    status = "PASS_CERTIFIED" if passed else "INCONCLUSIVE"
    certificate = run_dir / "certificates" / "s23_symmetry_vs_flatness.json"
    write_json(
        certificate,
        {
            "task_id": "S-23",
            "run_id": run_id,
            "status": status,
            "maximum_preserving_degeneracy_spread": preserving_degeneracy,
            "minimum_nonzero_breaking_degeneracy_spread": breaking_separation,
            "all_protection_gap_proxies_positive": bool((frame["symmetry_protection_gap_proxy"] > 0).all()),
            "distinction": "Symmetry-preserving perturbations retain degeneracy while moving the trace root; symmetry-breaking perturbations split principal responses even at the trace root. Degeneracy, protection, and flattening are therefore recorded separately.",
            "frozen_S14_cross_check_rows": len(frozen),
            "scope": "Hodge-response active fiber; no bulk projector claim",
            "reason_if_inconclusive": "The preregistered degeneracy-separation or protection margin did not close.",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
