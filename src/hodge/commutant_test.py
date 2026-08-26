from __future__ import annotations

from pathlib import Path

import sympy as sp

from audit.data_io import write_json


def exact_commutant() -> dict[str,object]:
    i0=sp.Matrix([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]])
    j0=sp.Matrix([[0,0,-1,0],[0,0,0,1],[1,0,0,0],[0,-1,0,0]])
    symbols=sp.symbols("x0:10")
    a=sp.zeros(4);index=0
    for row in range(4):
        for col in range(row,4):
            a[row,col]=symbols[index];a[col,row]=symbols[index];index+=1
    equations=list(a*i0-i0*a)+list(a*j0-j0*a)
    solution=sp.linsolve(equations,symbols)
    solved=next(iter(solution))
    scalar=(all(solved[index] == 0 for index in (1,2,3,5,6,8))
            and solved[0] == solved[4] == solved[7] == solved[9])
    return {"I0_square":str(i0*i0),"J0_square":str(j0*j0),"anticommutator":str(i0*j0+j0*i0),"solution":str(solution),"scalar_commutant":scalar}


def run(config:dict,run_dir:Path,run_id:str)->tuple[str,dict[str,Path]]:
    data=exact_commutant();status="PASS_EXACT" if data["scalar_commutant"] else "FAIL_IMPLEMENTATION"
    certificate=run_dir/"exact"/"s10_commutant.json";write_json(certificate,{"task_id":"S-10","run_id":run_id,"status":status,**data})
    return status,{"raw":certificate,"derived":certificate,"certificate":certificate}

