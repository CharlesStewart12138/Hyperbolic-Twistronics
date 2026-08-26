from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json
from spectral.natural_surface_model import export_normal_forms, kernel_records, natural_parameters, packing_tail_bounds, parse_normal_forms, symmetry_average


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    params = natural_parameters(config)
    normal_forms = run_dir / "raw" / "surface_group_normal_forms.txt"
    gap_info = export_normal_forms(root, config, normal_forms)
    words = parse_normal_forms(normal_forms)
    records = kernel_records(words, params)
    later = [row for row in records if int(row["word_length"]) >= 2]
    b_later = np.zeros((4,4))
    rows = []
    for row in records:
        n = np.asarray(row["abelian"], dtype=float)
        if int(row["word_length"]) >= 2:
            b_later += float(row["weight"]) * np.outer(n,n)
        rows.append({"word_length": int(row["word_length"]), "distance": float(row["distance"]), "radial_distance": float(row["radial_distance"]), "weight": float(row["weight"]), "abelian_norm_squared": float(n@n)})
    frame = pd.DataFrame(rows)
    shell = frame.groupby("word_length", as_index=False).agg(count=("weight","size"), weight_sum=("weight","sum"), hodge_trace_sum=("abelian_norm_squared", lambda values: 0.0))
    trace_by_length = frame.assign(weighted_trace=frame["weight"]*frame["abelian_norm_squared"]).groupby("word_length")["weighted_trace"].sum()
    shell["hodge_trace_sum"] = shell["word_length"].map(trace_by_length)
    shell_path = run_dir / "derived" / "full_kernel_shells.parquet"
    shell.to_parquet(shell_path, index=False)
    b_sym = symmetry_average(b_later)
    symmetry_residual = float(np.linalg.norm(b_sym - np.trace(b_sym)/4.0*np.eye(4), ord=2))
    npz = run_dir / "raw" / "full_kernel_hodge.npz"
    np.savez_compressed(npz, B_later=b_later, B_later_sym=b_sym, q1=params["q1"], parameters=np.array([params[k] for k in ("R","mu","d1","ell1","height")]))
    tail = packing_tail_bounds(params, float(config["natural_surface_model"]["packing_CA"]))
    beta_finite = float(np.trace(b_sym)/8.0)
    beta_tail_upper = float(tail["hodge_trace_upper"] / 8.0)
    qeff_lower = params["q1"] + beta_finite
    qeff_upper = qeff_lower + beta_tail_upper
    root_lower = params["t"] / qeff_upper
    root_upper = params["t"] / qeff_lower
    passed = gap_info.get("automatic") == "true" and symmetry_residual < 1.0e-10 and root_lower > 0 and root_upper <= 12.0
    status = "PASS_CERTIFIED" if passed else "FAIL_THEORY"
    certificate = run_dir / "certificates" / "schur_tail_certificate.json"
    write_json(certificate, {"task_id": "S-05", "run_id": run_id, "status": status, "model": "M2 full physical product-distance exponential kernel", "gap_backend": {key:value for key,value in gap_info.items() if key != "stdout"}, "normal_form_cutoff": int(config["natural_surface_model"]["normal_form_cutoff"]), "normal_form_count": len(words), "parameters": params, "finite_later_hodge_beta": beta_finite, "packing_tail": tail, "tail_hodge_beta_upper": beta_tail_upper, "full_root_interval": [root_lower, root_upper], "symmetry_average_residual": symmetry_residual, "tail_scope": "The packing bound covers the entire later-shell complement; using it after the finite partial sum is conservative and may double count.", "CA_provenance": "declared octagon tessellation crossing bound ||n(gamma)|| <= CA(1+d/R)"})
    return status, {"raw": normal_forms, "derived": shell_path, "certificate": certificate}

