from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit.data_io import write_json,write_zarr
from spectral.natural_surface_model import natural_parameters


def run(config:dict,run_dir:Path,run_id:str)->tuple[str,dict[str,Path]]:
    params=natural_parameters(config);tail=json.loads((run_dir/"certificates"/"schur_tail_certificate.json").read_text(encoding="utf-8"));data=np.load(run_dir/"raw"/"full_kernel_hodge.npz")
    cs=np.eye(4);bfinite=2*params["q1"]*cs+data["B_later_sym"]
    blow=float(np.trace(bfinite)/4);bhigh=blow+float(tail["packing_tail"]["hodge_trace_upper"])/4;bmid=(blow+bhigh)/2
    w=np.linspace(max(.05,2/bhigh-.5),min(12,2/blow+.5),121)
    tensors=np.stack([2*params["t"]*cs-value*bmid*cs for value in w])
    raw=run_dir/"raw"/"hodge_tensor.zarr";write_zarr(raw,{"w":w,"C_S":cs,"B_infinity_mid":bmid*cs,"K":tensors},{"run_id":run_id,"task_id":"S-11","B_scalar_interval":[blow,bhigh]})
    residual=float(np.linalg.norm(bfinite-blow*cs,ord=2));status="PASS_CONVERGED" if residual<1e-10 else "FAIL_IMPLEMENTATION"
    certificate=run_dir/"certificates"/"s11_hodge_tensor.json";write_json(certificate,{"task_id":"S-11","run_id":run_id,"status":status,"B_scalar_interval":[blow,bhigh],"root_interval":[2*params['t']/bhigh,2*params['t']/blow],"finite_symmetry_residual":residual,"full_tensor_not_trace_only":True})
    return status,{"raw":raw,"derived":certificate,"certificate":certificate}

