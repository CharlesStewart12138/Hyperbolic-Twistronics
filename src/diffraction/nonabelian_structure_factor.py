from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pandas as pd

from audit.data_io import write_json


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d06_nonabelian_structure_factor"
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / "d06_nonabelian_structure_factor.parquet"
    certificate = run_dir / "certificates" / "d06_nonabelian_structure_factor.json"
    representation_root = root / str(config["source_b07"]["representation_root"])
    intensities = []
    summaries = []
    for group in sorted(path for path in representation_root.iterdir() if path.is_dir()):
        table = json.loads((group / "character_table.json").read_text(encoding="utf-8"))
        order = int(table["order"])
        degrees = [int(value) for value in table["degrees"]]
        exact_numerator = sum(value * value for value in degrees)
        exact_parseval = Fraction(exact_numerator, order)
        for index, degree in enumerate(degrees, start=1):
            intensities.append({
                "group": group.name, "rep_index": index, "degree": degree,
                "delta_identity_fourier_frobenius_squared": degree,
                "parseval_intensity": float(Fraction(degree * degree, order)),
                "normalization_denominator": order,
            })
        summaries.append({
            "group": group.name, "order": order, "irrep_count": len(degrees),
            "nonabelian_irrep_count": sum(value > 1 for value in degrees),
            "sum_degree_squares": exact_numerator,
            "parseval_lhs_delta_identity": 1.0,
            "parseval_rhs_delta_identity": float(exact_parseval),
            "parseval_residual": float(abs(exact_parseval - 1)),
            "exact_equality": exact_parseval == 1,
        })
    pd.DataFrame(intensities).to_parquet(raw / "irrep_structure_factor_intensities.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(derived, index=False)
    tolerance = float(config["diffraction"]["parseval_tolerance"])
    passed = all(row["exact_equality"] and row["nonabelian_irrep_count"] > 0 for row in summaries)
    status = "PASS_CERTIFIED" if passed else "FAIL_THEORY"
    write_json(certificate, {
        "task_id": "D-06", "run_id": run_id, "status": status,
        "signal": "unit point scatterer delta_e on each certified finite quotient",
        "fourier_convention": "hat(f)(rho)=sum_g f(g) rho(g)^*; Parseval=sum_g|f(g)|^2=|G|^-1 sum_rho d_rho ||hat(f)(rho)||_F^2",
        "exact_certificate": "for delta_e, hat(f)(rho)=I_d and the equality reduces exactly to sum d_rho^2=|G|",
        "records": summaries, "tolerance": tolerance,
        "complete_irrep_data_source": "B-07 PASS_CERTIFIED character tables",
    })
    context["d06_summary"] = pd.DataFrame(summaries)
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
