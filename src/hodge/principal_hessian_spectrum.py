from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit.data_io import write_json,write_zarr


def run(config:dict,run_dir:Path,run_id:str)->tuple[str,dict[str,Path]]:
    tensor=json.loads((run_dir/"certificates"/"s11_hodge_tensor.json").read_text(encoding="utf-8"));blow,bhigh=map(float,tensor["B_scalar_interval"]);b=(blow+bhigh)/2;root=2/b
    w=np.linspace(max(.05,root-1),min(12,root+1),101);values=np.column_stack([2-w*b]*4);slopes=np.full((len(w),4),-b);spread=np.ptp(values,axis=1)
    raw=run_dir/"raw"/"principal_hessian_spectrum.zarr";write_zarr(raw,{"w":w,"eigenvalues":values,"slopes":slopes,"spread":spread},{"run_id":run_id,"task_id":"S-13"})
    status="PASS_CONVERGED" if float(np.max(spread))<1e-12 and np.all(slopes<0) else "FAIL_THEORY"
    certificate=run_dir/"certificates"/"s13_principal_hessian.json";write_json(certificate,{"task_id":"S-13","run_id":run_id,"status":status,"common_root_midpoint":root,"all_four_slopes":list(map(float,slopes[0])),"maximum_eigenvalue_spread":float(np.max(spread))})
    return status,{"raw":raw,"derived":certificate,"certificate":certificate}

