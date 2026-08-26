from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit.data_io import write_json, write_zarr
from spectral.natural_surface_model import natural_parameters


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    params = natural_parameters(config)
    tail = json.loads((run_dir/"certificates"/"schur_tail_certificate.json").read_text(encoding="utf-8"))
    displacement = json.loads((run_dir/"certificates"/"s06_root_displacement.json").read_text(encoding="utf-8"))
    q1,t,m=params["q1"],params["t"],int(params["m"])
    gstar=2*t*(1/q1-2*m)
    qmax=1+2*m*q1
    wfinite=t/(q1+float(tail["finite_later_hodge_beta"]))
    np.load(run_dir/"raw"/"full_kernel_hodge.npz").close()
    shells_path=run_dir/"derived"/"full_kernel_shells.parquet"
    import pandas as pd
    shells=pd.read_parquet(shells_path)
    finite_mass=float(shells.loc[shells.word_length>=2,"weight_sum"].sum())
    epsilon_finite=wfinite*finite_mass+qmax*float(displacement["measured_displacement"])
    epsilon_cert=12.0*float(tail["packing_tail"]["scalar_l1_upper"])+qmax*float(displacement["theoretical_bound"])
    finite_gap_lower=gstar-2*epsilon_finite
    certified_gap_lower=gstar-2*epsilon_cert
    status="PASS_CERTIFIED" if certified_gap_lower>0 else "INCONCLUSIVE"
    w=np.linspace(max(0.05,wfinite-1),min(12,wfinite+1),81)
    bandwidth=2*(epsilon_finite+abs(w-wfinite)*qmax)
    gap=np.maximum(0,gstar-bandwidth)
    raw=run_dir/"raw"/"spectral_island.zarr"
    write_zarr(raw,{"w":w,"bandwidth_upper":bandwidth,"external_gap_lower":gap,"active_overlap_lower":np.full_like(w,t*q1/(q1+float(tail['finite_later_hodge_beta']))),"layer_coherence":np.ones_like(w)},{"run_id":run_id,"task_id":"S-07","target_label":"isolated narrow spectral island continued from an exact point island"})
    certificate=run_dir/"certificates"/"s07_regular_operator_island.json"
    write_json(certificate,{"task_id":"S-07","run_id":run_id,"status":status,"reference_gap":gstar,"finite_normal_form_epsilon":epsilon_finite,"finite_gap_lower":finite_gap_lower,"certified_packing_epsilon":epsilon_cert,"certified_gap_lower":certified_gap_lower,"bandwidth_exactly_zero":False,"required_label":"isolated narrow spectral island continued from an exact point island","reason_if_inconclusive":"The rigorous packing tail is too conservative to prove a positive infinite-regular external gap; finite normal-form estimates are retained but not promoted."})
    return status,{"raw":raw,"derived":certificate,"certificate":certificate}

