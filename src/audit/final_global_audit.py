from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json
from audit.error_budget import COMPONENTS
from audit.run_manifest import sha256_file
from audit.validation_matrix import COLUMNS, clean_records
from representation.b07_recovery_seed import _tree_inventory, inventory_digest


NEW_CLAIMS = {
    "G-13": ("Theorem 4 / Eqs. 1115-1124", "exact arithmetic sampling of the magic surface", "geometry/arithmetic", "certified M1 root + exact centered sequence", "FIGURE 13"),
    "G-14": ("Theorem 5 / Eqs. 1127-1131", "exponent-four magic-supercell complexity", "geometry/arithmetic", "maximal-order degree with fixed-group bounds", "FIGURE 13"),
    "G-15": ("Eq. 6018", "incommensurate joint-limit criterion and falsification", "geometry/bulk interface", "certified non-Abelian towers", "FIGURE 14"),
    "S-17": ("Eq. 5613 / 5956", "same-target operational magic metrics", "spectral", "fixed finite ARO-3B active fiber", "FIGURE 15"),
    "S-18": ("Theorem 160", "complete-spectrum master-curve collapse", "spectral", "fixed finite ARO-3B active fiber", "FIGURE 16"),
    "S-19": ("Theorem 165", "geometry-spectrum exponent factorization", "spectral", "fixed finite ARO-3B active fiber", "FIGURE 16"),
    "S-20": ("Theorem 142 / Eq. 5965", "operational magic landscape", "spectral", "fixed finite ARO-3B active fiber", "FIGURE 17"),
    "S-21": ("Theorems 142-143", "fold/cusp derivative certificates", "spectral", "fixed finite ARO-3B active fiber", "FIGURE 17"),
    "S-22": ("Theorem 143", "curvature-born branch diagnostic", "spectral", "same transported finite target", "FIGURE 17"),
    "S-23": ("symmetry/flatness audit", "symmetry degeneracy versus kinetic flattening", "spectral/Hodge", "finite Hodge-response fiber", "FIGURE 15"),
    "S-24": ("Eqs. 6010-6013", "reverse universality falsification", "negative control", "saved complete spectra", "NEGATIVE CONTROLS"),
}


def _artifact_digest(path: Path) -> dict[str, object]:
    if path.is_file():
        return {"kind": "file", "sha256": sha256_file(path), "bytes": path.stat().st_size}
    inventory = _tree_inventory(path)
    return {"kind": "directory", "tree_inventory_sha256": inventory_digest(inventory), "file_count": len(inventory)}


def _predecessor_inventory(root: Path, current_run_id: str) -> pd.DataFrame:
    rows = []
    for directory in sorted((root / "results").iterdir()):
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or directory.name == current_run_id or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inventory = _tree_inventory(directory)
        temp_files = [name for name in inventory if str(name).endswith(".tmp")]
        rows.append(
            {
                "run_id": directory.name,
                "manifest_run_id": manifest.get("run_id"),
                "manifest_status": manifest.get("status"),
                "manifest_sha256": sha256_file(manifest_path),
                "tree_inventory_sha256": inventory_digest(inventory),
                "file_count": len(inventory),
                "directory_matches_manifest": manifest.get("run_id") == directory.name,
                "manifest_terminal": manifest.get("status") in {"COMPLETE", "INCOMPLETE"},
                "temporary_file_count": len(temp_files),
            }
        )
    return pd.DataFrame(rows)


def _matrix(root: Path, run_dir: Path, run_id: str, config: dict) -> pd.DataFrame:
    old = pd.read_parquet(root / str(config["frozen_predecessor"]["validation_matrix"]))
    old_by_task = {str(row.code_id): row._asdict() for row in old.itertuples(index=False)}
    with (root / "TASK_MANIFEST.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    rows = []
    for item in manifest:
        task = item["task_id"]
        row = {column: None for column in COLUMNS}
        if task in old_by_task:
            row.update({key: value for key, value in old_by_task[task].items() if key in row})
        if task in NEW_CLAIMS:
            theorem, claim, layer, model, figure = NEW_CLAIMS[task]
            row.update(
                {
                    "theorem_id": theorem,
                    "claim_name": claim,
                    "claim_layer": layer,
                    "model_level": model,
                    "future_figure_id": figure,
                    "parameter_set": "configs/final_remaining.yaml + preregistration amendment",
                }
            )
        row.update(
            {
                "code_id": task,
                "run_id": item["run_id"],
                "validation_type": item["status"],
                "status": item["status"],
                "raw_data_file": item["raw_output"] or "MISSING",
                "derived_data_file": item["derived_output"] or "NOT_APPLICABLE",
                "certificate_file": item["certificate"] or "MISSING",
            }
        )
        if item["status"] in {"INCONCLUSIVE", "FAIL_EXPECTED"}:
            prior = "" if row.get("notes") is None or pd.isna(row.get("notes")) else str(row.get("notes"))
            row["notes"] = (prior + " INCONCLUSIVE and FAIL_EXPECTED are preserved without relabelling.").strip()
        rows.append(row)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    raw_dir = run_dir / "raw" / "final_global_audit"
    derived_dir = run_dir / "derived" / "final_global_audit"
    raw_dir.mkdir(parents=True, exist_ok=False)
    derived_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(raw_dir / "theorem_validation_matrix.csv", index=False)
    frame.to_parquet(derived_dir / "theorem_validation_matrix.parquet", index=False)
    return frame


def _certificate_value(path: Path, key: str, default: float | None = None) -> float | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get(key, default)
    return None if value is None else float(value)


def _new_budget_rows(root: Path, manifest: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    value_keys = {
        "G-13": ("maximum_sampling_identity_residual", "solver_error"),
        "G-14": ("extrapolated_relative_error", "angular_error"),
        "G-15": (None, None),
        "S-17": ("maximum_refinement_relative_change", "solver_error"),
        "S-18": ("maximum_pairwise_normalized_spectrum_residual", "truncation_error"),
        "S-19": ("maximum_absolute_factorization_residual", "solver_error"),
        "S-20": (None, None),
        "S-21": (None, None),
        "S-22": (None, None),
        "S-23": ("maximum_preserving_degeneracy_spread", "solver_error"),
        "S-24": ("correct_control_residual", "solver_error"),
    }
    for item in manifest:
        task = item["task_id"]
        if task not in NEW_CLAIMS:
            continue
        certificate = root / item["certificate"]
        key, component = value_keys[task]
        values = {name: None for name in COMPONENTS}
        if key is not None and component is not None:
            values[component] = _certificate_value(certificate, key, 0.0)
        known = [float(value) for value in values.values() if value is not None]
        rows.append(
            {
                "observable_id": f"final_{task.lower().replace('-', '_')}",
                "task_id": task,
                "model_scope": NEW_CLAIMS[task][3],
                "status": item["status"],
                "notes": "New final-stage observable; unquantified components remain explicit nulls.",
                **values,
                "known_error_component_count": len(known),
                "unknown_error_component_count": len(COMPONENTS) - len(known),
                "total_known_error_upper": float(sum(known)),
            }
        )
    return rows


def _error_budget(root: Path, run_dir: Path, config: dict) -> pd.DataFrame:
    inherited = pd.read_parquet(root / str(config["final_audit"]["inherited_error_budget"]))
    with (root / "TASK_MANIFEST.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    combined = pd.concat([inherited, pd.DataFrame(_new_budget_rows(root, manifest))], ignore_index=True, sort=False)
    output = run_dir / "derived" / "final_global_audit" / str(config["final_audit"]["error_budget_filename"])
    combined.to_parquet(output, index=False)
    return combined


def _build_workbook(
    root: Path,
    run_dir: Path,
    run_id: str,
    config: dict,
    matrix: pd.DataFrame,
    budget: pd.DataFrame,
    node_executable: Path,
) -> tuple[Path, dict[str, object]]:
    raw_dir = run_dir / "raw" / "final_global_audit"
    derived_dir = run_dir / "derived" / "final_global_audit"
    provenance = json.loads((root / "public_data" / "provenance.json").read_text(encoding="utf-8"))
    input_path = raw_dir / "final_workbook_input.json"
    write_json(
        input_path,
        {
            "matrix": clean_records(matrix),
            "error_budget": clean_records(budget),
            "provenance": provenance["resources"],
            "run_id": run_id,
        },
    )
    workbook = derived_dir / str(config["final_audit"]["workbook_filename"])
    preview_dir = derived_dir / "workbook_previews"
    verification = derived_dir / "final_workbook_verification.json"
    result = subprocess.run(
        [
            str(node_executable),
            str(root / "src" / "audit" / "theorem_validation_workbook.mjs"),
            str(input_path),
            str(workbook),
            str(preview_dir),
            str(verification),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    (run_dir / "logs" / "final_workbook.stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "logs" / "final_workbook.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"final artifact-tool workbook failed: {result.stderr[-3000:]}")
    audit = json.loads(verification.read_text(encoding="utf-8"))
    expected_preview_ranges = config.get("final_audit_workbook_render_recovery", {}).get(
        "repair_contract", {}
    ).get("preview_ranges", {})
    checks = {
        "formula_error_count_zero": audit.get("formula_error_count") == 0,
        "rendered_sheet_count_four": audit.get("rendered_sheet_count") == 4,
        "preview_ranges_exact": audit.get("rendered_ranges") == expected_preview_ranges,
        "summary_reconciled": audit.get("summary_reconciled") is True,
        "validation_rows_88": audit.get("summary_actual", {}).get("validation_rows") == 88,
        "provenance_date_format": audit.get("provenance_date_number_format") == "yyyy-mm-dd hh:mm:ss",
    }
    if not all(checks.values()):
        raise RuntimeError(f"final workbook verification failed: {checks}")
    return workbook, {"verification": verification, "preview_dir": preview_dir, "checks": checks, "audit": audit}


def _render_figures(root: Path, run_dir: Path, scientific_run_dir: Path) -> Path:
    output = run_dir / "derived" / "final_publication_figures"
    result = subprocess.run(
        [sys.executable, str(root / "src" / "plots" / "render_final_publication_figures.py"), str(scientific_run_dir), str(output)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    (run_dir / "logs" / "final_figures.stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "logs" / "final_figures.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"final publication figure rendering failed: {result.stderr[-3000:]}")
    figures = sorted(output.glob("*.svg"))
    if len(figures) != 5:
        raise RuntimeError(f"expected five final publication figures, found {len(figures)}")
    return output


def _write_reports(
    root: Path,
    run_dir: Path,
    run_id: str,
    config: dict,
    matrix: pd.DataFrame,
    budget: pd.DataFrame,
    predecessor: pd.DataFrame,
    workbook: Path,
    figure_dir: Path,
) -> tuple[Path, Path, Path]:
    counts = matrix["status"].value_counts().sort_index().to_dict()
    inconclusive = matrix.loc[matrix.status == "INCONCLUSIVE", "code_id"].tolist()
    expected_failures = matrix.loc[matrix.status == "FAIL_EXPECTED", "code_id"].tolist()
    lines = [
        "# Final computer-assisted validation report",
        "",
        f"- Final run ID: `{run_id}`",
        "- Global audit: `88/88 TERMINAL`",
        "- Project state: `PROJECT_COMPLETE`",
        "- Completed D/NC tasks rerun: `false`",
        "- Scientific calculations in plotting scripts: `false`",
        "",
        "## A. Model and reproducibility protocol",
        "",
        "All predecessor runs remain in their original immutable directories. New work starts at G-13 and uses frozen inputs only after SHA-256 and tree-inventory verification. The S-17--S-24 claims are restricted to the preregistered fixed transported finite-rank ARO-3B active fiber; the frozen S-07/S-08 infinite-regular INCONCLUSIVE results are not promoted to bulk conclusions.",
        "",
        "## B. Geometry and exact sampled magic sequence",
        "",
        "G-13 verifies the exact sampling identities of manuscript Eqs. (1115)--(1124). G-14 tests the exponent-four magic complexity coefficient without replacing the exact Euler/parity factors. G-15 retains both a passing and a deliberately failing joint-limit sequence on three certified non-Abelian towers.",
        "",
        "## C. Operational spectrum and universality diagnostics",
        "",
        "S-17 computes W, Delta, Omega_max, rho_coh, and C_coh from one transported target. S-18 stores complete normalized spectra. S-19 stores geometric and spectral derivatives independently. S-20 stores the full preregistered three-dimensional landscape. S-21--S-23 separate bifurcation, branch-birth, symmetry, protection, and flattening diagnostics. S-24 preserves the five reverse tests as expected failures.",
        "",
        "## D. Global adjudication",
        "",
        f"Status counts: `{json.dumps(counts, sort_keys=True)}`.",
        f"Preserved INCONCLUSIVE tasks: `{', '.join(inconclusive)}`.",
        f"Preserved FAIL_EXPECTED tasks: `{', '.join(expected_failures)}`.",
        f"Immutable predecessor runs inventoried: `{len(predecessor)}`.",
        f"Error-budget observables: `{len(budget)}`; null components remain explicitly unquantified.",
        "",
        "## E. Theorem-to-computation matrix",
        "",
        "| Task | Status | Theorem / claim | Certificate |",
        "|---|---|---|---|",
    ]
    for row in matrix.itertuples(index=False):
        lines.append(f"| {row.code_id} | {row.status} | {row.theorem_id} | `{row.certificate_file}` |")
    lines.extend(
        [
            "",
            "## F. Final artifacts",
            "",
            f"- Workbook: `{workbook.relative_to(root).as_posix()}`",
            f"- Error budget: `{(run_dir / 'derived' / 'final_global_audit' / config['final_audit']['error_budget_filename']).relative_to(root).as_posix()}`",
            f"- Publication figures: `{figure_dir.relative_to(root).as_posix()}`",
            "",
        ]
    )
    markdown_text = "\n".join(lines)
    markdown_run = run_dir / "derived" / "final_global_audit" / str(config["final_audit"]["report_markdown"])
    markdown_run.write_text(markdown_text, encoding="utf-8")
    markdown_root = root / "reports" / str(config["final_audit"]["report_markdown"])
    markdown_root.write_text(markdown_text, encoding="utf-8")

    tex_rows = []
    for row in matrix.itertuples(index=False):
        theorem = str(row.theorem_id).replace("_", "\\_").replace("&", "\\&")
        status_tex = str(row.status).replace("_", "\\_")
        tex_rows.append(f"{row.code_id} & {status_tex} & {theorem} \\\\")
    latex = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{longtable}",
            r"\begin{document}",
            r"\section*{Final Computer-Assisted Validation Report}",
            f"Final run: \\texttt{{{run_id}}}. The global audit closes 88/88 tasks.",
            r"\begin{longtable}{lll}",
            r"Task & Status & Theorem / claim \\\hline",
            *tex_rows,
            r"\end{longtable}",
            f"Preserved inconclusive tasks: {', '.join(inconclusive).replace('_', '\\_')}.\\",
            f"Preserved expected failures: {', '.join(expected_failures).replace('_', '\\_')}.",
            r"\end{document}",
        ]
    )
    latex_run = run_dir / "derived" / "final_global_audit" / str(config["final_audit"]["report_latex"])
    latex_run.write_text(latex, encoding="utf-8")
    latex_root = root / "reports" / str(config["final_audit"]["report_latex"])
    latex_root.write_text(latex, encoding="utf-8")
    return markdown_root, latex_root, markdown_run


def run(
    config: dict,
    run_dir: Path,
    run_id: str,
    root: Path,
    node_executable: Path,
) -> dict[str, Path | object]:
    final_cfg = config["final_audit"]
    terminal = set(final_cfg["terminal_statuses"])
    with (root / "TASK_MANIFEST.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    statuses = {row["task_id"]: row["status"] for row in manifest}
    recovery = config.get("final_audit_recovery", {}).get("repair_contract", {})
    legacy_blank_derived = set(recovery.get("tasks_with_blank_legacy_derived_output", []))

    def output_paths_valid(row: dict[str, str]) -> bool:
        if not row["raw_output"] or not row["certificate"]:
            return False
        if not (root / row["raw_output"]).exists() or not (root / row["certificate"]).exists():
            return False
        if row["derived_output"]:
            return (root / row["derived_output"]).exists()
        return row["task_id"] in legacy_blank_derived

    checks = {
        "task_count_88": len(manifest) == int(final_cfg["expected_tasks"]),
        "unique_task_ids_88": len(statuses) == int(final_cfg["expected_tasks"]),
        "all_terminal": all(status in terminal for status in statuses.values()),
        "no_blockers": not any(status in set(final_cfg["blockers"]) for status in statuses.values()),
        "all_recorded_paths_present_and_legacy_blank_derived_explicit": all(output_paths_valid(row) for row in manifest),
        "preserved_statuses": all(statuses.get(task) == expected for task, expected in final_cfg["preserve_statuses"].items()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"88/88 manifest audit failed: {checks}")

    predecessor = _predecessor_inventory(root, run_id)
    predecessor_ok = bool(
        not predecessor.empty
        and predecessor["directory_matches_manifest"].all()
        and predecessor["manifest_terminal"].all()
        and (predecessor["temporary_file_count"] == 0).all()
    )
    checks["all_predecessor_hashes_inventoried"] = predecessor_ok
    if not predecessor_ok:
        raise RuntimeError("immutable predecessor inventory audit failed")
    predecessor_path = run_dir / "derived" / "predecessor_hash_inventory.parquet"
    predecessor.to_parquet(predecessor_path, index=False)

    matrix = _matrix(root, run_dir, run_id, config)
    matrix_ok = len(matrix) == 88 and matrix.code_id.nunique() == 88 and matrix.status.to_dict() == pd.Series(statuses).reindex(matrix.code_id).reset_index(drop=True).to_dict()
    checks["matrix_manifest_reconciled"] = bool(matrix_ok)
    if not matrix_ok:
        raise RuntimeError("final validation matrix did not reconcile to TASK_MANIFEST")
    budget = _error_budget(root, run_dir, config)
    workbook, workbook_audit = _build_workbook(root, run_dir, run_id, config, matrix, budget, node_executable)
    scientific_run_dir = root / str(
        config.get("scientific_source_run_dir", run_dir.relative_to(root).as_posix())
    )
    figure_dir = _render_figures(root, run_dir, scientific_run_dir)

    final_figure_root = root / "figures" / "final_validation"
    if final_figure_root.exists():
        raise FileExistsError(f"final figure directory already exists: {final_figure_root}")
    shutil.copytree(figure_dir, final_figure_root)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    workbook_root = reports_dir / str(final_cfg["workbook_filename"])
    budget_root = reports_dir / str(final_cfg["error_budget_filename"])
    shutil.copy2(workbook, workbook_root)
    shutil.copy2(run_dir / "derived" / "final_global_audit" / str(final_cfg["error_budget_filename"]), budget_root)

    figure_sources = {
        "figure13_magic_complexity.svg": [scientific_run_dir / "derived" / "magic_subsequence_sampling.parquet", scientific_run_dir / "derived" / "magic_complexity.parquet"],
        "figure14_joint_limit.svg": [scientific_run_dir / "derived" / "incommensurate_joint_limit.parquet"],
        "figure15_operational_spectrum.svg": [scientific_run_dir / "raw" / "operational_magic.zarr"],
        "figure16_master_collapse.svg": [scientific_run_dir / "raw" / "master_curve_complete_spectra.zarr"],
        "figure17_magic_landscape.svg": [scientific_run_dir / "raw" / "magic_landscape.zarr"],
    }
    index_rows = []
    for filename, sources in figure_sources.items():
        figure = final_figure_root / filename
        index_rows.append(
            {
                "figure": figure.relative_to(root).as_posix(),
                "figure_sha256": sha256_file(figure),
                "source_data": ";".join(path.relative_to(root).as_posix() for path in sources),
                "source_digests": json.dumps([_artifact_digest(path) for path in sources], sort_keys=True),
                "scientific_calculation_in_renderer": False,
            }
        )
    figure_index = pd.DataFrame(index_rows)
    figure_index_run = run_dir / "derived" / "final_global_audit" / str(final_cfg["figure_index"])
    figure_index_root = reports_dir / str(final_cfg["figure_index"])
    figure_index.to_csv(figure_index_run, index=False)
    figure_index.to_csv(figure_index_root, index=False)

    markdown_root, latex_root, _ = _write_reports(
        root, run_dir, run_id, config, matrix, budget, predecessor, workbook, figure_dir
    )
    counts = {str(key): int(value) for key, value in matrix.status.value_counts().sort_index().items()}
    audit_certificate = run_dir / "certificates" / "final_global_audit.json"
    write_json(
        audit_certificate,
        {
            "run_id": run_id,
            "status": "PASS_CERTIFIED",
            "checks": checks,
            "task_count": 88,
            "terminal_task_count": 88,
            "status_counts": counts,
            "predecessor_run_count": len(predecessor),
            "predecessor_hash_inventory": predecessor_path.relative_to(root).as_posix(),
            "workbook_checks": workbook_audit["checks"],
            "preserved_inconclusive": matrix.loc[matrix.status == "INCONCLUSIVE", "code_id"].tolist(),
            "preserved_expected_failures": matrix.loc[matrix.status == "FAIL_EXPECTED", "code_id"].tolist(),
        },
    )
    final_status = {
        "schema_version": 1,
        "run_id": run_id,
        "state": "PROJECT_COMPLETE",
        "project_complete": True,
        "global_audit_status": "PASS_CERTIFIED",
        "task_count": 88,
        "terminal_task_count": 88,
        "status_counts": counts,
        "blockers": [],
        "preserved_inconclusive": matrix.loc[matrix.status == "INCONCLUSIVE", "code_id"].tolist(),
        "preserved_expected_failures": matrix.loc[matrix.status == "FAIL_EXPECTED", "code_id"].tolist(),
        "artifacts": {
            "validation_report_md": markdown_root.relative_to(root).as_posix(),
            "validation_report_tex": latex_root.relative_to(root).as_posix(),
            "theorem_validation_matrix_xlsx": workbook_root.relative_to(root).as_posix(),
            "error_budget_parquet": budget_root.relative_to(root).as_posix(),
            "publication_figure_data_index": figure_index_root.relative_to(root).as_posix(),
            "publication_figures": final_figure_root.relative_to(root).as_posix(),
            "global_audit_certificate": audit_certificate.relative_to(root).as_posix(),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    final_status_run = run_dir / "derived" / "final_global_audit" / str(final_cfg["status_filename"])
    write_json(final_status_run, final_status)
    final_status_root = root / str(final_cfg["status_filename"])
    write_json(final_status_root, final_status)
    state_path = root / "PROJECT_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "state": "PROJECT_COMPLETE",
            "current_phase": "FINAL_AUDIT",
            "latest_run_id": run_id,
            "latest_run_directory": run_dir.relative_to(root).as_posix(),
            "next_task": None,
            "final_validation_status": final_status_root.relative_to(root).as_posix(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "certificate": audit_certificate,
        "status": final_status_root,
        "workbook": workbook_root,
        "error_budget": budget_root,
        "report_md": markdown_root,
        "report_tex": latex_root,
        "figure_index": figure_index_root,
        "figure_dir": final_figure_root,
        "matrix": matrix,
    }
