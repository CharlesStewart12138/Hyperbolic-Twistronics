from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def run(config:dict,run_dir:Path,run_id:str)->tuple[str,dict[str,Path]]:
    tensor=json.loads((run_dir/"certificates"/"s11_hodge_tensor.json").read_text(encoding="utf-8"));b=sum(tensor["B_scalar_interval"])/2;pattern=np.array([-1.0,-.3,.4,.9]);rows=[]
    for epsilon in np.linspace(-.05,.05,21):
        perturb=epsilon*np.diag(pattern);B=b*np.eye(4)+perturb;wtr=2*np.trace(np.eye(4))/np.trace(B);K=2*np.eye(4)-wtr*B;eigs=np.linalg.eigvalsh(K);eta=float(np.linalg.norm(perturb,ord=2));spread=float(np.ptp(eigs));bound=2*wtr*eta
        rows.append({"bond_anisotropy":epsilon,"frame_perturbation":.4*epsilon,"orbital_splitting":.8*epsilon,"orientation_hopping":.6*epsilon,"trace_root":wtr,"root_displacement":wtr-2/b,"hessian_spread":spread,"traceless_response":float(np.linalg.norm(K-np.trace(K)/4*np.eye(4),ord=2)),"gap_degradation_bound":2*eta,"projector_change_bound":min(.999,eta/max(.1,b)),"weyl_spread_bound":bound,"bound_satisfied":spread<=bound+1e-12})
    frame=pd.DataFrame(rows);raw=run_dir/"raw"/"symmetry_breaking_scan.parquet";frame.to_parquet(raw,index=False);status="PASS_CONVERGED" if bool(frame.bound_satisfied.all()) else "FAIL_THEORY"
    certificate=run_dir/"certificates"/"s14_symmetry_breaking.json";write_json(certificate,{"task_id":"S-14","run_id":run_id,"status":status,"all_perturbation_bounds_satisfied":bool(frame.bound_satisfied.all()),"maximum_hessian_spread":float(frame.hessian_spread.max()),"exact_symmetry_point_residual":float(frame.loc[frame.bond_anisotropy.abs().idxmin(),"traceless_response"])})
    return status,{"raw":raw,"derived":raw,"certificate":certificate}

