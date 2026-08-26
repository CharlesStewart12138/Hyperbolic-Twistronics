from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from audit.data_io import write_json
from audit.run_manifest import sha256_file


def _save(frame: pd.DataFrame, directory: Path, stem: str) -> list[Path]:
    parquet = directory / f"{stem}.parquet"
    csv = directory / f"{stem}.csv"
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    return [parquet, csv]


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d15_export_figure_data"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d15_publication_outputs"
    figure_data = derived / "figure_ready_data"
    figures = derived / "publication_figures"
    figure_data.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    certificate = run_dir / "certificates" / "d15_export_figure_data.json"
    sources = {
        "d02": context["outputs"]["D-02"]["derived"],
        "d05": context["outputs"]["D-05"]["derived"],
        "d08_pair_blocks": context["outputs"]["D-08"]["derived"] / "pair_block_summaries.parquet",
        "d08_envelopes": context["outputs"]["D-08"]["derived"] / "dyadic_arithmetic_envelopes.parquet",
        "d09": context["outputs"]["D-09"]["derived"],
        "d11": context["outputs"]["D-11"]["derived"],
        "d13": context["outputs"]["D-13"]["derived"],
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"D-15 frozen derived source missing: {label}={path}")
    inventory = {
        label: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for label, path in sources.items()
    }
    write_json(raw / "derived_source_inventory.json", {
        "run_id": run_id, "scientific_recalculation": False, "sources": inventory,
    })
    generated = []
    generated += _save(pd.read_parquet(sources["d02"]), figure_data, "figure10_cdf_convergence")
    generated += _save(pd.read_parquet(sources["d05"]), figure_data, "figure10_coherence_weighted_dos")
    generated += _save(pd.read_parquet(sources["d08_pair_blocks"]), figure_data, "figure11_scale_separated_exponents")
    generated += _save(pd.read_parquet(sources["d08_envelopes"]), figure_data, "figure11_dyadic_envelopes")
    generated += _save(pd.read_parquet(sources["d09"]), figure_data, "figure12_public_hyperbloch_dos")
    generated += _save(pd.read_parquet(sources["d11"]), figure_data, "figure12_circuit_reconstruction")
    generated += _save(pd.read_parquet(sources["d13"]), figure_data, "final_error_budget")
    renderer = root / "src" / "plots" / "render_publication_figures.py"
    result = subprocess.run(
        [sys.executable, str(renderer), str(figure_data), str(figures)],
        cwd=root, capture_output=True, text=True, check=False, timeout=300,
    )
    (run_dir / "logs" / "d15_figures.stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "logs" / "d15_figures.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"D-15 SVG renderer failed: {result.stderr[-2000:]}")
    figure_files = sorted(figures.glob("*.svg"))
    required = {
        "figure10_bulk_dos.svg", "figure11_arithmetic_exponent.svg", "figure12_external_reproduction.svg",
    }
    present = {path.name for path in figure_files}
    status = "PASS_CERTIFIED" if present == required and all(path.stat().st_size > 1000 for path in figure_files) else "FAIL_IMPLEMENTATION"
    write_json(certificate, {
        "task_id": "D-15", "run_id": run_id, "status": status,
        "scientific_results_calculated_in_plotting_script": False,
        "source_derived_hashes": inventory,
        "figure_ready_files": [path.relative_to(root).as_posix() for path in generated],
        "publication_figures": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in figure_files
        ],
        "renderer": renderer.relative_to(root).as_posix(),
        "vector_format": "SVG",
    })
    context["d15_figure_data"] = figure_data
    context["d15_figures"] = figure_files
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
