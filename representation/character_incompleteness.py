from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json
from bulk.finite_cover_model import bilayer_energies


def run(config, run_dir: Path, run_id: str, root: Path, context: dict[str, object]):
    frame: pd.DataFrame = context["blocks"]
    coupling = float(config["bilayer_family"]["interlayer_coupling"])
    rows = []
    demonstrated = True
    for (tower_id, level), subset in frame.groupby(["tower_id", "level"]):
        one_dimensional = subset[subset.degree == 1]
        trivial = one_dimensional.iloc[int(np.argmax(one_dimensional.adjacency_eigenvalue))]
        single = bilayer_energies(np.asarray([trivial.adjacency_eigenvalue]), 1.0, coupling)
        full_values = bilayer_energies(np.asarray(subset.adjacency_eigenvalue), 1.0, coupling)
        full_dimension = 2 * int(np.sum(subset.regular_multiplicity))
        misses = len(np.unique(np.round(full_values, 10))) - len(np.unique(np.round(single, 10)))
        demonstrated = demonstrated and misses > 0 and full_dimension > len(single)
        rows.append({"tower_id": tower_id, "level": int(level), "single_character_rep_index": int(trivial.rep_index), "single_character_energies": [float(x) for x in single], "single_character_dimension": len(single), "full_regular_dimension": full_dimension, "full_distinct_block_energy_count": len(np.unique(np.round(full_values, 10))), "missed_distinct_energies": misses, "single_character_representation_complete": False})
    raw = run_dir / "raw" / "character_incompleteness.parquet"
    pd.DataFrame([{**row, "single_character_energies": str(row["single_character_energies"])} for row in rows]).to_parquet(raw, index=False)
    derived = run_dir / "derived" / "b08_character_incompleteness.parquet"
    pd.DataFrame([{**row, "single_character_energies": str(row["single_character_energies"])} for row in rows]).to_parquet(derived, index=False)
    certificate = run_dir / "certificates" / "b08_character_incompleteness.json"
    status = "FAIL_EXPECTED" if demonstrated else "FAIL_IMPLEMENTATION"
    write_json(certificate, {"task_id": "B-08", "run_id": run_id, "status": status, "negative_control_succeeded": demonstrated, "records": rows, "conclusion": "A single automorphic character is not representation complete and cannot certify the finite regular spectrum."})
    return status, {"raw": raw, "derived": derived, "certificate": certificate}

