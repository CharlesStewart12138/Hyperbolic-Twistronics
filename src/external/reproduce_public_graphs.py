from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json
from audit.run_manifest import sha256_file
from external.reproduce_hyperbloch_dos import benchmark_paths, parse_graph


def library_paths(root: Path, config):
    revision = str(config["public_data"]["graph_library_revision"])
    base = root / "public_data" / "cell-graph-library" / revision / "repo"
    return [
        base / "model-graphs" / "{8,3}-tess_T2.1_3.hcm",
        base / "supercell-model-graphs" / "{8,3}-tess_T2.1_3_sc-T5.1.hcs",
    ]


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d10_reproduce_public_graphs"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d10_reproduce_public_graphs.parquet"
    certificate = run_dir / "certificates" / "d10_reproduce_public_graphs.json"
    records = []
    for hyperbloch, library in zip(benchmark_paths(root, config), library_paths(root, config)):
        adjacency_a, edges_a = parse_graph(hyperbloch)
        adjacency_b, edges_b = parse_graph(library)
        spectrum_a = np.linalg.eigvalsh(adjacency_a)
        spectrum_b = np.linalg.eigvalsh(adjacency_b)
        residual = float(np.max(np.abs(spectrum_a - spectrum_b))) if len(spectrum_a) == len(spectrum_b) else float("inf")
        record = {
            "benchmark": hyperbloch.stem,
            "hyperbloch_source": hyperbloch.relative_to(root).as_posix(),
            "library_source": library.relative_to(root).as_posix(),
            "hyperbloch_sha256": sha256_file(hyperbloch), "library_sha256": sha256_file(library),
            "byte_identical": sha256_file(hyperbloch) == sha256_file(library),
            "vertex_count_a": adjacency_a.shape[0], "vertex_count_b": adjacency_b.shape[0],
            "edge_identifications_equal": edges_a == edges_b,
            "adjacency_equal": bool(np.array_equal(adjacency_a, adjacency_b)),
            "spectral_sup_residual": residual,
        }
        records.append(record)
    pd.DataFrame(records).to_parquet(raw / "public_graph_comparisons.parquet", index=False)
    pd.DataFrame(records).to_parquet(derived, index=False)
    tolerance = float(config["public_reproduction"]["spectral_tolerance"])
    passed = all(row["edge_identifications_equal"] and row["adjacency_equal"] and row["spectral_sup_residual"] <= tolerance for row in records)
    status = "PASS_EXTERNAL" if passed else "FAIL_IMPLEMENTATION"
    write_json(certificate, {
        "task_id": "D-10", "run_id": run_id, "status": status,
        "records": records, "spectral_tolerance": tolerance,
        "independent_parser": True,
        "comparison_scope": "primitive graph, supercell graph, boundary identifications and spectra",
        "hyperbloch_revision": config["public_data"]["hyperbloch_revision"],
        "graph_library_revision": config["public_data"]["graph_library_revision"],
    })
    context["d10_records"] = records
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
