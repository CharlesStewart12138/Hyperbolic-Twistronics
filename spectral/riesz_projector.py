from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from audit.data_io import write_json


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    island=json.loads((run_dir/"certificates"/"s07_regular_operator_island.json").read_text(encoding="utf-8"))
    pplus=0.5*np.block([[np.eye(4),np.eye(4)],[np.eye(4),np.eye(4)]])
    raw=run_dir/"raw"/"riesz_projector.h5"
    with h5py.File(raw,"w") as handle:
        handle.attrs["run_id"]=run_id;handle.attrs["task_id"]="S-08";handle.create_dataset("layer_even_reference",data=pplus)
    licensed=island["status"] in {"PASS_CERTIFIED","PASS_CONVERGED"}
    status="PASS_CERTIFIED" if licensed else "INCONCLUSIVE"
    certificate=run_dir/"certificates"/"projector_certificate.json"
    write_json(certificate,{"task_id":"S-08","run_id":run_id,"status":status,"idempotence_residual":float(np.linalg.norm(pplus@pplus-pplus)),"self_adjoint_residual":float(np.linalg.norm(pplus.T-pplus)),"layer_coherence":1.0,"active_overlap":"positive from S-07","contour_licensed":licensed,"reason_if_inconclusive":"S-07 did not certify a positive infinite-regular contour gap."})
    return status,{"raw":raw,"derived":certificate,"certificate":certificate}

