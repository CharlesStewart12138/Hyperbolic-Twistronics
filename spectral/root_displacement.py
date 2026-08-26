from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from audit.data_io import write_json
from spectral.natural_surface_model import natural_parameters


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    params = natural_parameters(config)
    source = json.loads((run_dir / "certificates" / "schur_tail_certificate.json").read_text(encoding="utf-8"))
    beta = float(source["finite_later_hodge_beta"])
    beta_upper = beta + float(source["tail_hodge_beta_upper"])
    first = params["t"] / params["q1"]
    measured = params["t"] / (params["q1"] + beta)
    displacement = first - measured
    bound = params["t"] * beta_upper / (params["q1"] * (params["q1"] + beta_upper))
    status = "PASS_CONVERGED" if 0 <= displacement <= bound else "FAIL_THEORY"
    raw = run_dir / "raw" / "root_displacement.parquet"
    pd.DataFrame([{"first_shell_root": first, "finite_full_root": measured, "measured_displacement": displacement, "sign_aware_theoretical_bound": bound, "certified_full_root_lower": source["full_root_interval"][0], "certified_full_root_upper": source["full_root_interval"][1]}]).to_parquet(raw,index=False)
    certificate = run_dir / "certificates" / "s06_root_displacement.json"
    write_json(certificate, {"task_id":"S-06","run_id":run_id,"status":status,"measured_displacement":displacement,"theoretical_bound":bound,"bound_derivation":"positive-semidefinite later Hodge tensor gives t*b/[q1(q1+b)]"})
    return status, {"raw":raw,"derived":raw,"certificate":certificate}

