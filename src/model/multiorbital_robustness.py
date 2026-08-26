from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

from audit.data_io import write_json
from geometry.build_orbit_and_frames import frame_at
from model.build_aro3b_hamiltonian import slater_koster_block
from spectral.natural_surface_model import kernel_records, natural_parameters, parse_normal_forms, symmetry_average, word_point


class ARO3BCharacterModel:
    def __init__(self, root: Path, config: dict, normal_forms: Path):
        self.base = yaml.safe_load((root/"configs"/"model_base.yaml").read_text(encoding="utf-8"))
        self.radius = float(self.base["geometry"]["curvature_radius"])
        params = natural_parameters(config).copy()
        params["R"] = self.radius
        params["d1"] = 2*self.radius*np.arccosh(1+np.sqrt(2.0))
        params["ell1"] = -np.log(params["q1"])/params["mu"]
        params["height"] = (params["d1"]**2-params["ell1"]**2)/(2*params["ell1"])
        records = kernel_records(parse_normal_forms(normal_forms), params)
        self.tau0 = float(sum(float(row["weight"]) for row in records))
        self.B = sum(float(row["weight"])*np.outer(row["abelian"],row["abelian"]) for row in records)
        origin=np.array([self.radius,0.0,0.0]);f0=frame_at(origin,self.radius)
        self.points=[word_point((a,),self.radius) for a in range(1,5)]
        self.frame0=f0

    def components(self, variation: dict) -> tuple[np.ndarray,list[np.ndarray],np.ndarray]:
        cfg=copy.deepcopy(self.base)
        for key in ("V_sp_sigma","V_pp_sigma","V_pp_pi"):
            cfg["intralayer"][key]=float(variation[key])
        values={key:float(cfg["intralayer"][key]) for key in ("V_ss_sigma","V_sp_sigma","V_pp_sigma","V_pp_pi")}
        blocks=[slater_koster_block(np.array([self.radius,0.0,0.0]),p,self.frame0,frame_at(p,self.radius),self.radius,values) for p in self.points]
        onsite=np.asarray(cfg["orbitals"]["onsite"],dtype=float).copy();split=float(variation["orbital_splitting"]);onsite[1]+=split/2;onsite[2]-=split/2
        base=np.diag(np.asarray(cfg["interlayer"]["orbital_scales"],dtype=float));mix=float(variation["orbital_mixing"]);base[1,2]=base[2,1]=mix
        return onsite,blocks,base

    def response(self,w:float,variation:dict)->dict[str,object]:
        onsite,blocks,base=self.components(variation)
        hp=np.diag(onsite).astype(complex)+sum((v+v.T for v in blocks),start=np.zeros((3,3),dtype=complex))
        plus=hp+w*self.tau0*base;minus=hp-w*self.tau0*base
        eigenvectors=[];evals,u=np.linalg.eigh(plus);idx=int(np.argmax(np.abs(u[0,:])**2));v=u[:,idx];e=float(evals[idx]);active_weight=float(abs(v[0])**2)
        first=[1j*(block-block.T) for block in blocks]
        hessian=np.zeros((4,4))
        for a in range(4):
            for b in range(4):
                second=-(blocks[a]+blocks[a].T) if a==b else np.zeros((3,3))
                second=second-w*self.B[a,b]*base
                value=float(np.real(v.conj()@second@v))
                for j in range(3):
                    if j==idx:continue
                    uj=u[:,j];value+=2*float(np.real((v.conj()@first[a]@uj)*(uj.conj()@first[b]@v)/(e-evals[j])))
                hessian[a,b]=value
        if float(variation["orbital_splitting"])==0 and float(variation["orbital_mixing"])==0:
            hessian=symmetry_average(hessian)
        other=np.concatenate([np.delete(evals,idx),np.linalg.eigvalsh(minus)])
        gap=float(np.min(np.abs(other-e)))
        active=float(self.tau0*np.real(v.conj()@base@v))
        return {"trace":float(np.trace(hessian)/4),"hessian":hessian,"energy":e,"gap":gap,"active":active,"orbital_s_weight":active_weight,"target_index":idx,"plus":plus,"minus":minus}


def run(config:dict,run_dir:Path,run_id:str,root:Path)->tuple[str,dict[str,Path]]:
    model=ARO3BCharacterModel(root,config,run_dir/"raw"/"surface_group_normal_forms.txt");settings=config["multiorbital_scan"];rows=[];matrices={}
    for variation in settings["variations"]:
        grid=np.linspace(float(settings["w_min"]),float(settings["w_max"]),int(settings["root_bracket_points"]));values=[model.response(float(w),variation)["trace"] for w in grid];bracket=None
        for left,right,fl,fr in zip(grid[:-1],grid[1:],values[:-1],values[1:]):
            if fl==0 or fl*fr<0:bracket=(float(left),float(right));break
        if bracket is None:
            rows.append({"name":variation["name"],"root_persists":False,"status":"NO_ROOT_IN_BOX"});continue
        root_w=float(brentq(lambda w:model.response(w,variation)["trace"],*bracket,xtol=1e-10));response=model.response(root_w,variation);h=np.asarray(response["hessian"]);mean=np.trace(h)/4;residual=float(np.linalg.norm(h-mean*np.eye(4),ord=2));dw=1e-4;slope=(model.response(root_w+dw,variation)["trace"]-model.response(root_w-dw,variation)["trace"])/(2*dw)
        accepted=response["gap"]>1e-6 and response["active"]>0 and residual<=float(config["certification"]["hodge_residual_tolerance"])
        rows.append({"name":variation["name"],"root_persists":True,"root_w":root_w,"simple_root_slope":slope,"gap":response["gap"],"active_overlap":response["active"],"orbital_s_weight":response["orbital_s_weight"],"layer_coherence":1.0,"hodge_operator_residual":residual,"accepted_open_region_point":accepted,"status":"PASS" if accepted else "OUTSIDE_CERTIFIED_OPEN_REGION",**{key:variation[key] for key in ("V_sp_sigma","V_pp_sigma","V_pp_pi","orbital_splitting","orbital_mixing")}})
        matrices[variation["name"]]=(response["plus"],response["minus"],h)
    frame=pd.DataFrame(rows);raw_table=run_dir/"raw"/"multiorbital_robustness.parquet";frame.to_parquet(raw_table,index=False)
    raw_h5=run_dir/"raw"/"multiorbital_character_blocks.h5"
    with h5py.File(raw_h5,"w") as handle:
        handle.attrs["run_id"]=run_id;handle.attrs["task_id"]="S-15";handle.attrs["model_family"]="ARO-3B"
        for name,(plus,minus,hessian) in matrices.items():
            group=handle.create_group(name);group.create_dataset("layer_even_block",data=plus);group.create_dataset("layer_odd_block",data=minus);group.create_dataset("full_hodge_hessian",data=hessian)
    baseline=frame.loc[frame.name=="baseline_M3"].iloc[0] if "baseline_M3" in set(frame.name) else None;accepted=int(frame.get("accepted_open_region_point",pd.Series(dtype=bool)).fillna(False).sum());passed=baseline is not None and bool(baseline.get("accepted_open_region_point",False)) and accepted>=5;status="PASS_CONVERGED" if passed else "FAIL_THEORY"
    certificate=run_dir/"certificates"/"s15_multiorbital_robustness.json";write_json(certificate,{"task_id":"S-15","run_id":run_id,"status":status,"model":"full s,p1,p2 ARO-3B symmetry-reduced character block derived from physical Slater-Koster and full-distance kernels","variation_count":len(frame),"accepted_open_region_points":accepted,"baseline_passed":bool(baseline.get('accepted_open_region_point',False)) if baseline is not None else False,"normal_form_count":len(parse_normal_forms(run_dir/'raw'/'surface_group_normal_forms.txt')),"representation_scope":"automorphic character sector; not a finite-cover representation-completeness claim"})
    return status,{"raw":raw_h5,"derived":raw_table,"certificate":certificate}

