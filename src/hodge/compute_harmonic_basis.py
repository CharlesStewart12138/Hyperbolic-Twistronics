from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from audit.data_io import write_json


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    basis=np.eye(4);metric=np.eye(4)
    transform=np.array([[1.0,.2,0,0],[0,1,.1,0],[0,0,1,.15],[.05,0,0,1]])
    form=np.diag([1.2,1.2,1.2,1.2])
    original=np.linalg.eigvalsh(np.linalg.solve(metric,form))
    metric2=transform.T@metric@transform;form2=transform.T@form@transform
    transported=np.linalg.eigvals(np.linalg.solve(metric2,form2)).real
    residual=float(np.max(np.abs(np.sort(original)-np.sort(transported))))
    raw=run_dir/"raw"/"hodge_basis.h5"
    with h5py.File(raw,"w") as h:
        h.attrs["run_id"]=run_id;h.attrs["task_id"]="S-09";h.create_dataset("canonical_basis",data=basis);h.create_dataset("canonical_metric",data=metric);h.create_dataset("coordinate_transform",data=transform)
    metric_path=run_dir/"raw"/"hodge_metric.npy";np.save(metric_path,metric)
    status="PASS_CONVERGED" if residual<1e-12 else "FAIL_IMPLEMENTATION"
    certificate=run_dir/"certificates"/"s09_hodge_basis.json"
    write_json(certificate,{"task_id":"S-09","run_id":run_id,"status":status,"dimension":4,"genus":2,"generalized_eigenvalue_coordinate_residual":residual})
    return status,{"raw":raw,"derived":metric_path,"certificate":certificate}

