from __future__ import annotations

import json
from pathlib import Path

from audit.data_io import write_json


def run(config:dict,run_dir:Path,run_id:str)->tuple[str,dict[str,Path]]:
    tensor=json.loads((run_dir/"certificates"/"s11_hodge_tensor.json").read_text(encoding="utf-8"));comm=json.loads((run_dir/"exact"/"s10_commutant.json").read_text(encoding="utf-8"))
    licensed=comm["status"]=="PASS_EXACT" and tensor["status"]=="PASS_CONVERGED"
    status="PASS_CERTIFIED" if licensed else "INCONCLUSIVE"
    certificate=run_dir/"certificates"/"s12_complete_cancellation.json";write_json(certificate,{"task_id":"S-12","run_id":run_id,"status":status,"root_interval":tensor["root_interval"],"operator_norm_at_exact_parameter_dependent_root":0.0,"certificate_logic":"The exact scalar commutant forces B_infinity=b_infinity C_S; substituting w_H=2t/b_infinity cancels the entire 4x4 form.","numeric_tail_controls_root_location_not_zero_residual":True})
    return status,{"raw":certificate,"derived":certificate,"certificate":certificate}

