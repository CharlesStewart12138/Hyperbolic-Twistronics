from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from common import initialize_run, load_yaml, sha256_file, write_json  # noqa: E402
from report_text import CURVATURE_THEORY_TEX, DEEP_REPORT_MD, MANUSCRIPT_REVISION_TEX  # noqa: E402


def verify_sources(config: dict[str, object]) -> dict[str, object]:
    checks = {}
    for family, entry in config["source_runs"].items():
        run_dir = EXTENSION_ROOT / "results" / str(entry["run_id"])
        actual = sha256_file(run_dir / "freeze_certificate.json")
        checks[family] = {
            "run_id": str(entry["run_id"]),
            "expected": str(entry["freeze_certificate_sha256"]),
            "actual": actual,
            "pass": actual == str(entry["freeze_certificate_sha256"]),
        }
    if not all(bool(value["pass"]) for value in checks.values()):
        raise RuntimeError(f"final source verification failed: {checks}")
    return checks


def copy_data(source: Path, target: Path) -> str:
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return sha256_file(target)


def main() -> int:
    config_path = EXTENSION_ROOT / "configs" / "final_source_runs_preregistration.yaml"
    config = load_yaml(config_path)
    checks = verify_sources(config)
    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT, "FINAL_AUDIT")
    runs = {name: EXTENSION_ROOT / "results" / value["run_id"] for name, value in config["source_runs"].items()}
    copies = {
        "figure_8_deep_cover_levels.parquet": runs["R8_R9"] / "figure_data" / "figure_8_deep_cover_levels.parquet",
        "figure_8_matched_levels.parquet": runs["R8_R9"] / "figure_data" / "figure_8_matched_levels.parquet",
        "figure_9_balanced_diagonal.parquet": runs["R8_R9"] / "figure_data" / "figure_9_balanced_diagonal.parquet",
        "figure_10_KPM_SLQ_CDF.parquet": runs["R10"] / "figure_data" / "figure_10_KPM_SLQ_CDF.parquet",
        "figure_10_vanishing_schedule.parquet": runs["R10"] / "figure_data" / "figure_10_vanishing_schedule.parquet",
        "figure_10_holdout_coherence_density.parquet": runs["R10"] / "figure_data" / "figure_10_holdout_coherence_density.parquet",
        "figure_16_hypothesis_holdout.parquet": runs["R16_holdout"] / "figure_data" / "figure_16_hypothesis_holdout.parquet",
        "figure_16_old_radius_shape_field.parquet": runs["R16_holdout"] / "figure_data" / "figure_16_old_radius_shape_field.parquet",
        "figure_18_curvature_scaling.parquet": runs["R16_holdout"] / "figure_data" / "figure_18_curvature_scaling.parquet",
        "figure_18_hypothesis_summary.parquet": runs["R16_holdout"] / "figure_data" / "figure_18_hypothesis_summary.parquet",
        "r16_tangent_gram.parquet": runs["R16_design"] / "derived" / "r16_tangent_gram.parquet",
        "r16_block_rank.parquet": runs["R16_design"] / "derived" / "r16_block_rank.parquet",
        "r16_observable_jacobian.parquet": runs["R16_design"] / "derived" / "r16_observable_jacobian.parquet",
    }
    figure_hashes = {name: copy_data(source, run_dir / "figure_data" / name) for name, source in copies.items()}

    holdout = pd.read_parquet(run_dir / "figure_data" / "figure_16_hypothesis_holdout.parquet")
    channel_summary = []
    for channel, group in holdout[holdout.channel != "X"].groupby("channel"):
        for hypothesis in ("H0", "H2", "H3", "H4", "wrong_K"):
            channel_summary.append(
                {
                    "channel": channel,
                    "hypothesis": hypothesis,
                    "max_C0": float(group[f"{hypothesis}_C0"].max()),
                    "max_C1": float(group[f"{hypothesis}_C1"].max()),
                    "max_C2": float(group[f"{hypothesis}_C2"].max()),
                    "median_C0": float(group[f"{hypothesis}_C0"].median()),
                }
            )
    channel_path = run_dir / "figure_data" / "figure_16_channel_summary.parquet"
    pd.DataFrame(channel_summary).to_parquet(channel_path, index=False)
    figure_hashes[channel_path.name] = sha256_file(channel_path)
    design_certificate = json.loads((runs["R16_design"] / "certificates" / "r16_design_certificate.json").read_text(encoding="utf-8"))
    singular_rows = []
    for family, values in (
        ("operator_primary", design_certificate["primary_gram"]["singular_values"]),
        ("observable_jacobian", design_certificate["observable_jacobian"]["singular_values"]),
    ):
        for index, value in enumerate(values, start=1):
            singular_rows.append({"family": family, "index": index, "singular_value": float(value)})
    singular_path = run_dir / "figure_data" / "figure_18_singular_values.parquet"
    pd.DataFrame(singular_rows).to_parquet(singular_path, index=False)
    figure_hashes[singular_path.name] = sha256_file(singular_path)

    error_rows = []
    r9 = pd.read_parquet(runs["R8_R9"] / "derived" / "r9_balanced_error_budget.parquet")
    for row in r9.itertuples():
        for component in ("epsilon_physical_tail", "epsilon_master_tail", "epsilon_cover", "epsilon_transport", "epsilon_solver", "epsilon_representation", "epsilon_total_C0", "epsilon_total_C1", "epsilon_total_C2"):
            error_rows.append({"family": "R9", "case": f"{row.tower_id}:n{row.level}:L{row.L_j}", "component": component, "value": float(getattr(row, component)), "status": "INCONCLUSIVE"})
    r10 = pd.read_parquet(runs["R10"] / "derived" / "r10_error_budget.parquet")
    for row in r10.itertuples():
        for component in ("kappa_N", "eta_N", "kappa_over_eta", "KPM_truncation_resolution", "SLQ_quadrature_resolution", "finite_cover_error", "reference_error", "KPM_SLQ_disagreement"):
            error_rows.append({"family": "R10", "case": f"{row.tower_id}:n{row.level}", "component": component, "value": float(getattr(row, component)), "status": "PASS_WEAK_CDF" if row.split == "holdout" else "FINITE_PILOT"})
    for row in holdout.itertuples():
        if row.channel == "X":
            continue
        for hypothesis in ("H0", "H2", "H3", "H4"):
            for tier in ("C0", "C1", "C2"):
                error_rows.append({"family": "R16", "case": row.case_id, "component": f"{hypothesis}_{tier}", "value": float(getattr(row, f"{hypothesis}_{tier}")), "status": "PASS_RESTRICTED_CLASS"})
    error_frame = pd.DataFrame(error_rows)
    error_frame.to_parquet(run_dir / "reports" / "deep_error_budget.parquet", index=False)

    claims = [
        {"claim_id": "F8-WEAK", "figure": 8, "task_id": "R8-E/R10-B", "status": "PASS_WEAK_BULK", "scope": "retained projective-sector local spectral measures only", "run_id": config["source_runs"]["R10"]["run_id"], "certificate": "r10_dos_certificate.json", "source_data": "figure_10_vanishing_schedule.parquet"},
        {"claim_id": "F8-STRONG", "figure": 8, "task_id": "R8-C", "status": "INCONCLUSIVE", "scope": "full regular spectrum edge/gap no-pollution", "run_id": config["source_runs"]["R8_R9"]["run_id"], "certificate": "r8_deep_cover_certificate.json", "source_data": "figure_8_deep_cover_levels.parquet"},
        {"claim_id": "F9-BALANCED", "figure": 9, "task_id": "R9-A", "status": "PASS_CERTIFIED", "scope": "L=1..5 and analytic balanced-law limits", "run_id": config["source_runs"]["R8_R9"]["run_id"], "certificate": "r9_balanced_certificate.json", "source_data": "figure_9_balanced_diagonal.parquet"},
        {"claim_id": "F9-CLOSURE", "figure": 9, "task_id": "R9-D/R9-E", "status": "INCONCLUSIVE", "scope": "C0/C1/C2 physical-tail closure", "run_id": config["source_runs"]["R8_R9"]["run_id"], "certificate": "r9_balanced_certificate.json", "source_data": "figure_9_balanced_diagonal.parquet"},
        {"claim_id": "F10-WEAK", "figure": 10, "task_id": "R10-B/R10-D", "status": "PASS_CONVERGED", "scope": "weak CDF plus finite holdout eta=sqrt(kappa) schedule", "run_id": config["source_runs"]["R10"]["run_id"], "certificate": "r10_dos_certificate.json", "source_data": "figure_10_vanishing_schedule.parquet"},
        {"claim_id": "F10-LOCAL", "figure": 10, "task_id": "R10-C/R10-E", "status": "INCONCLUSIVE", "scope": "tower-uniform local/unsmoothed/coherence DOS", "run_id": config["source_runs"]["R10"]["run_id"], "certificate": "r10_dos_certificate.json", "source_data": "figure_10_holdout_coherence_density.parquet"},
        {"claim_id": "F16-CLASS", "figure": 16, "task_id": "R16-V08", "status": "PASS_RESTRICTED_CLASS", "scope": "fixed comparison class", "run_id": config["source_runs"]["R16_holdout"]["run_id"], "certificate": "r16_holdout_certificate.json", "source_data": "figure_16_hypothesis_holdout.parquet"},
        {"claim_id": "F18-CURVATURE", "figure": 18, "task_id": "R16-V02/R16-V03/R16-V04", "status": "INCONCLUSIVE", "scope": "local operator rank-2 evidence but no global observable identifiability/holdout", "run_id": config["source_runs"]["R16_holdout"]["run_id"], "certificate": "r16_holdout_certificate.json", "source_data": "figure_18_singular_values.parquet"},
    ]
    for claim in claims:
        claim["source_data_sha256"] = figure_hashes[claim["source_data"]]
        claim["figure_data_sha256"] = figure_hashes[claim["source_data"]]
    claim_frame = pd.DataFrame(claims)
    claim_frame.to_parquet(run_dir / "reports" / "theorem_to_computation_matrix.parquet", index=False)
    claim_frame.to_csv(run_dir / "reports" / "theorem_to_computation_matrix.csv", index=False)
    write_json(run_dir / "reports" / "publication_claim_ledger.json", {"schema_version": 1, "claims": claims})

    (run_dir / "reports" / "curvature_relevant_universality_theory.tex").write_text(CURVATURE_THEORY_TEX, encoding="utf-8")
    (run_dir / "reports" / "manuscript_revision_proposal.tex").write_text(MANUSCRIPT_REVISION_TEX, encoding="utf-8")
    (run_dir / "reports" / "deep_resolution_report.md").write_text(DEEP_REPORT_MD, encoding="utf-8")
    report_tex = r"""\documentclass[11pt]{article}
\usepackage{booktabs,geometry}\geometry{margin=1in}
\title{Deep Resolution Validation Report}\date{}
\begin{document}\maketitle
\section*{Outcome}
Figure 8: weak retained-sector spectral-measure closure passes; strong full-spectrum no-pollution is inconclusive.
Figure 9: the balanced diagonal is certified, while C0/C1/C2 tail closure is inconclusive.
Figure 10: weak CDF and the frozen broadening schedule pass, but local unsmoothed DOS is inconclusive.
Figure 16: \texttt{PASS\_RESTRICTED\_CLASS}; curvature is not globally certified as an independent relevant coordinate.
No genuine \texttt{FAIL\_THEORY} was found.  One separate R10 \texttt{FAIL\_IMPLEMENTATION} run is preserved and superseded only operationally by a new recovery run.
\end{document}
"""
    (run_dir / "reports" / "deep_resolution_report.tex").write_text(report_tex, encoding="utf-8")
    write_json(
        run_dir / "certificates" / "final_audit_prefigure.json",
        {
            "run_id": run_id,
            "source_checks": checks,
            "anchor_checks": identity["anchor_checks"],
            "source_preregistration_sha256": sha256_file(config_path),
            "figure_data_hashes": figure_hashes,
            "claim_count": len(claims),
            "error_budget_rows": len(error_frame),
            "final_scientific_conclusions": {
                "Figure_8": "PASS_WEAK_BULK_RETAINED_PROJECTIVE_SECTOR_STRONG_INCONCLUSIVE",
                "Figure_9": "INCONCLUSIVE_TAIL_CLOSURE_WITH_TRUE_BALANCED_DIAGONAL",
                "Figure_10": "PASS_WEAK_CDF_AND_SCHEDULE_LOCAL_DOS_INCONCLUSIVE",
                "Figure_16": "PASS_RESTRICTED_CLASS",
                "curvature_independent_relevant_coordinate": "LOCAL_OPERATOR_EVIDENCE_NOT_GLOBAL_CERTIFICATION",
                "genuine_FAIL_THEORY": False,
            },
        },
    )
    print(json.dumps({"run_id": run_id, "figure_data_count": len(figure_hashes), "claim_count": len(claims), "error_budget_rows": len(error_frame)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

