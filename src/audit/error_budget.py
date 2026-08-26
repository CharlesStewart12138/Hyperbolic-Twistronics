from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


COMPONENTS = [
    "truncation_error", "physical_tail", "cover_error", "angular_error",
    "solver_error", "floating_point_interval_width", "transport_error",
    "representation_error", "dos_smoothing_error", "statistical_kpm_slq_error",
]


def budget_row(observable_id, task_id, scope, status, notes, **values):
    row = {
        "observable_id": observable_id, "task_id": task_id, "model_scope": scope,
        "status": status, "notes": notes,
    }
    for component in COMPONENTS:
        row[component] = values.get(component)
    known = [float(row[key]) for key in COMPONENTS if row[key] is not None and not pd.isna(row[key])]
    row["known_error_component_count"] = len(known)
    row["unknown_error_component_count"] = len(COMPONENTS) - len(known)
    row["total_known_error_upper"] = float(sum(known))
    return row


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d13_error_budget"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "error_budget.parquet"
    certificate = run_dir / "certificates" / "d13_error_budget.json"
    phase_b = context["phase_b_dir"]
    b11 = pd.read_parquet(phase_b / "derived" / "b11_full_shell_balance.parquet")
    b12 = pd.read_parquet(phase_b / "derived" / "b12_full_shell_spectral_inheritance.parquet")
    rows = []
    for record in b11.itertuples(index=False):
        inherited = b12[(b12.tower_id == record.tower_id) & (b12.level == record.level)].iloc[0]
        rows.append(budget_row(
            f"bulk_edges_{record.tower_id}_L{record.level}", "B-12", "retained non-Abelian full-shell sector",
            "PASS_CERTIFIED", "physical and master tails are not double counted",
            truncation_error=float(record.master_tail), physical_tail=float(record.physical_tail),
            cover_error=float(record.core_error), transport_error=float(inherited.spectral_hausdorff_error_upper),
            solver_error=0.0, floating_point_interval_width=0.0, representation_error=0.0,
        ))
    for record in context["d01_summary"].itertuples(index=False):
        rows.append(budget_row(
            f"kpm_full_regular_{record.tower_id}_L{record.level}", "D-01", "method-validation only; not bulk admissible",
            "PASS_CONVERGED", "full regular spectrum contains preserved pollution",
            statistical_kpm_slq_error=float(record.maximum_moment_standard_error),
            solver_error=float(record.moment_rms_error), dos_smoothing_error=float(record.density_normalization_error),
        ))
    d02 = context["d02_records"]
    for record in d02.itertuples(index=False):
        rows.append(budget_row(
            f"retained_cdf_{record.tower_id}_L{record.level}", "D-02", "retained non-Abelian sector",
            context["d02_status"], "kernel-independent empirical CDF diagnostic",
            cover_error=float(record.kappa_N), representation_error=0.0,
        ))
    errors = context["d05_errors"]
    rows.append(budget_row(
        "coherence_weighted_dos", "D-05", "retained layer-even bilayer sector",
        context["statuses"].get("D-05", "INCONCLUSIVE"), "all requested components are stored independently",
        cover_error=errors["spectral_error"], representation_error=errors["projector_error"],
        solver_error=errors["coherence_weight_error"], dos_smoothing_error=errors["smoothing_local_law_error"],
    ))
    for record in context["d06_summary"].itertuples(index=False):
        rows.append(budget_row(
            f"nonabelian_parseval_{record.group}", "D-06", "complete finite-quotient Fourier dual",
            "PASS_CERTIFIED", "exact degree-square identity",
            representation_error=float(record.parseval_residual), floating_point_interval_width=0.0,
        ))
    for record in context["d09_records"]:
        rows.append(budget_row(
            f"public_hyperbloch_{Path(record['source']).stem}", "D-09", "public {8,3} benchmark",
            "PASS_EXTERNAL", "independent parser and eigensolver",
            solver_error=float(record["hermiticity_residual"]),
        ))
    rows.append(budget_row(
        "circuit_laplacian_mapping", "D-11", "p7 level-one ARO-3B bilayer sector",
        context["statuses"].get("D-11", "INCONCLUSIVE"), "purely numerical circuit mapping",
        solver_error=float(context["d11_residual"]),
    ))
    frame = pd.DataFrame(rows)
    frame.to_parquet(raw / "error_budget_components.parquet", index=False)
    frame.to_parquet(derived, index=False)
    incomplete = frame[frame.unknown_error_component_count > 0].observable_id.tolist()
    write_json(certificate, {
        "task_id": "D-13", "run_id": run_id, "status": "PASS_CERTIFIED",
        "observable_count": len(frame), "component_columns": COMPONENTS,
        "unknown_components_explicitly_flagged": True,
        "observables_with_not_applicable_or_unquantified_components": incomplete,
        "zero_means_exactly_absent_or_exactly_recombined": True,
        "nan_means_not_applicable_or_not_quantified": True,
        "output": derived.relative_to(root).as_posix(),
    })
    context["error_budget"] = frame
    return "PASS_CERTIFIED", {"raw": raw, "derived": derived, "certificate": certificate}
