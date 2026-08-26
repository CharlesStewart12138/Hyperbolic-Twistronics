from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def run(config:dict,run_dir:Path,run_id:str)->tuple[str,dict[str,Path]]:
    b=0.4;anisotropy=np.diag([.3,-.2,.1,-.2]);B=b*np.eye(4)+anisotropy;wtr=2*np.trace(np.eye(4))/np.trace(B);K=2*np.eye(4)-wtr*B;eigs=np.linalg.eigvalsh(K);trace=float(np.trace(K));residual=float(np.linalg.norm(K-trace/4*np.eye(4),ord=2));expected=abs(trace)<1e-12 and residual>0 and eigs[0]<0<eigs[-1];status="FAIL_EXPECTED" if expected else "FAIL_IMPLEMENTATION"
    raw=run_dir/"raw"/"negative_control_trace_root.parquet";pd.DataFrame([{"trace_root":wtr,"trace_response":trace,"traceless_operator_norm":residual,"eigenvalue_1":eigs[0],"eigenvalue_2":eigs[1],"eigenvalue_3":eigs[2],"eigenvalue_4":eigs[3]}]).to_parquet(raw,index=False)
    certificate=run_dir/"certificates"/"s16_trace_root_negative_control.json";write_json(certificate,{"task_id":"S-16","run_id":run_id,"status":status,"model_family":"M4 anisotropic ARO-3B Hodge response","trace_zero":abs(trace)<1e-12,"traceless_response_nonzero":residual>0,"operator_norm":residual,"principal_responses":list(map(float,eigs)),"interpretation":"Expected falsification: a scalar trace root does not cancel the full tensor."})
    return status,{"raw":raw,"derived":raw,"certificate":certificate}

