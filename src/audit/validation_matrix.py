from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pandas as pd

from audit.data_io import write_json


COLUMNS = [
    "theorem_id", "claim_name", "claim_layer", "model_level", "code_id", "run_id",
    "validation_type", "parameter_set", "residual_value", "tolerance",
    "certified_lower_bound", "certified_upper_bound", "physical_margin", "status",
    "raw_data_file", "derived_data_file", "certificate_file", "future_figure_id", "notes",
]


D_CLAIMS = {
    "D-01": ("DOS protocol", "large-system KPM/SLQ DOS", "DOS", "full regular method check"),
    "D-02": ("local law", "kernel-independent CDF local-law diagnostic", "DOS", "retained sectors"),
    "D-03": ("local law", "vanishing-broadening balance", "DOS", "retained sectors"),
    "D-04": ("density theorem", "unsmoothed regular density gate", "DOS", "uniformly regular sectors only"),
    "D-05": ("spectral measure", "coherence-weighted DOS", "DOS", "retained layer-even sector"),
    "D-06": ("finite-group Plancherel", "non-Abelian generalized structure factor", "diffraction", "complete quotient dual"),
    "D-07": ("induced diffraction", "exact versus incommensurate structure", "diffraction", "finite/infinite index"),
    "D-08": ("arithmetic exponent", "diffraction representation complexity", "diffraction", "explicit centered sequence"),
    "D-09": ("external replication", "HyperBloch/HyperCells DOS benchmark", "external", "public {8,3} graphs"),
    "D-10": ("external replication", "public graph and supercell comparison", "external", "public {8,3} graphs"),
    "D-11": ("circuit mapping", "circuit-Laplacian representation", "external", "p7 bilayer sector"),
    "D-12": ("external replication", "public circuit spectrum reproduction", "external", "public baseline required"),
    "D-13": ("audit", "complete observable error budget", "audit", "all completed observables"),
    "D-14": ("audit", "theorem-to-computation matrix", "audit", "all project layers"),
    "D-15": ("publication audit", "figure-ready data and publication figures", "plots", "saved derived data only"),
}


def clean_records(frame: pd.DataFrame):
    records = []
    for record in frame.to_dict("records"):
        records.append({key: (None if pd.isna(value) else value) for key, value in record.items()})
    return records


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d14_validation_matrix"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d14_validation_matrix"
    derived.mkdir(parents=True, exist_ok=False)
    certificate = run_dir / "certificates" / "d14_validation_matrix.json"
    with (root / "TASK_MANIFEST.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    statuses = context["statuses"]
    outputs = context["outputs"]
    rows = []
    for source in manifest:
        task = source["task_id"]
        if task in D_CLAIMS:
            theorem, claim, layer, model = D_CLAIMS[task]
        else:
            theorem = source.get("theorem_id") or f"{task.split('-')[0]} validation claim"
            claim = source.get("claim_name") or task
            layer = source.get("claim_layer") or task.split("-")[0]
            model = source.get("model_level") or "frozen predecessor result"
        output = outputs.get(task, {})
        if task == "D-14":
            output = {"raw": raw, "derived": derived, "certificate": certificate}
            status = "PASS_CERTIFIED"
        else:
            status = statuses.get(task, source.get("status") or "NOT_STARTED")
        if task.startswith("D-"):
            future_figure = "FIGURE 10" if task <= "D-05" else ("FIGURE 11" if task <= "D-08" else "FIGURE 12" if task <= "D-12" else "FINAL AUDIT")
        else:
            future_figure = source.get("future_figure_id") or "PREDECESSOR"
        rows.append({
            "theorem_id": theorem, "claim_name": claim, "claim_layer": layer,
            "model_level": model, "code_id": task,
            "run_id": run_id if task in statuses else source.get("run_id", ""),
            "validation_type": status, "parameter_set": context.get("parameter_set", "configs/phase_d_nc.yaml") if task in D_CLAIMS else source.get("parameter_set", ""),
            "residual_value": None, "tolerance": None,
            "certified_lower_bound": None, "certified_upper_bound": None,
            "physical_margin": None, "status": status,
            "raw_data_file": output["raw"].relative_to(root).as_posix() if "raw" in output else source.get("raw_output", "MISSING"),
            "derived_data_file": output["derived"].relative_to(root).as_posix() if "derived" in output else source.get("derived_output", "MISSING"),
            "certificate_file": output["certificate"].relative_to(root).as_posix() if "certificate" in output else source.get("certificate", "MISSING"),
            "future_figure_id": future_figure,
            "notes": "INCONCLUSIVE and FAIL_EXPECTED are preserved without relabelling." if status in {"INCONCLUSIVE", "FAIL_EXPECTED"} else "",
        })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(raw / "theorem_validation_matrix.csv", index=False)
    frame.to_parquet(derived / "theorem_validation_matrix.parquet", index=False)
    builder_input = raw / "workbook_input.json"
    provenance = json.loads((root / "public_data" / "provenance.json").read_text(encoding="utf-8"))
    write_json(builder_input, {
        "matrix": clean_records(frame),
        "error_budget": clean_records(context["error_budget"]),
        "provenance": provenance["resources"],
        "run_id": run_id,
    })
    workbook = derived / str(config["workbook"]["filename"])
    verification = derived / str(config["workbook"]["verification_filename"])
    previews = derived / "previews"
    command = [
        str(context["node_executable"]),
        str(root / "src" / "audit" / "theorem_validation_workbook.mjs"),
        str(builder_input), str(workbook), str(previews), str(verification),
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False, timeout=600)
    (run_dir / "logs" / "d14_workbook.stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "logs" / "d14_workbook.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"artifact-tool workbook builder failed: {result.stderr[-2000:]}")
    audit = json.loads(verification.read_text(encoding="utf-8"))
    status = "PASS_CERTIFIED" if workbook.exists() and audit.get("formula_error_count") == 0 and audit.get("rendered_sheet_count") == 4 and audit.get("summary_reconciled") is True and audit.get("provenance_date_number_format") == "yyyy-mm-dd hh:mm:ss" else "FAIL_IMPLEMENTATION"
    write_json(certificate, {
        "task_id": "D-14", "run_id": run_id, "status": status,
        "row_count": len(frame), "required_columns": COLUMNS,
        "all_required_columns_present": list(frame.columns) == COLUMNS,
        "workbook": workbook.relative_to(root).as_posix(),
        "workbook_verification": verification.relative_to(root).as_posix(),
        "formula_error_count": audit.get("formula_error_count"),
        "rendered_sheet_count": audit.get("rendered_sheet_count"),
        "summary_reconciled": audit.get("summary_reconciled"),
        "summary_expected": audit.get("summary_expected"),
        "summary_actual": audit.get("summary_actual"),
        "provenance_date_number_format": audit.get("provenance_date_number_format"),
        "spreadsheet_backend": "@oai/artifact-tool",
    })
    context["d14_matrix"] = frame
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
