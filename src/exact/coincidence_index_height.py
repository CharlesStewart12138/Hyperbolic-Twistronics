from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pandas as pd
import sympy as sp

from audit.data_io import write_json


def legendre_two(p: int) -> int:
    if p == 2:
        return 0
    return int(sp.legendre_symbol(2, p))


def local_factor_record(j: int) -> dict[str, object]:
    n = j * j + 3
    factors = sp.factorint(n)
    epsilon = Fraction(1, 16) if j % 2 else Fraction(1, 1)
    height = epsilon * n * n
    euler = Fraction(1, 1)
    local: list[dict[str, object]] = []
    for prime, valuation in sorted(factors.items()):
        prime = int(prime)
        valuation = int(valuation)
        if prime == 2:
            local.append({"rational_prime": 2, "valuation_in_n": valuation, "type": "ramified_excluded", "local_index": 1})
            continue
        symbol = legendre_two(prime)
        if symbol == 1:
            single = (prime + 1) * prime ** (valuation - 1)
            local_index = single * single
            euler *= Fraction((prime + 1) ** 2, prime * prime)
            local_type = "split_two_prime_ideals"
        elif symbol == -1:
            norm = prime * prime
            local_index = (norm + 1) * norm ** (valuation - 1)
            euler *= Fraction(norm + 1, norm)
            local_type = "inert_one_prime_ideal"
        else:
            raise AssertionError("unexpected Legendre symbol")
        local.append({"rational_prime": prime, "valuation_in_n": valuation, "legendre_2_over_p": symbol, "type": local_type, "local_index": local_index})
    q = height * euler
    if height.denominator != 1 or q.denominator != 1:
        raise ArithmeticError(f"nonintegral exact invariant at j={j}")
    product = 1
    for row in local:
        product *= int(row["local_index"])
    return {"j": j, "n_j": n, "factorization": {str(k): int(v) for k, v in factors.items()}, "epsilon": str(epsilon), "projective_height": int(height), "euler_factor": str(euler), "coincidence_degree_maximal_order": int(q), "local_index_product": product, "product_identity": product == int(q), "local_factors": local}


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    j_max = int(config["arithmetic"]["j_max_exact"])
    records = [local_factor_record(j) for j in range(1, j_max + 1)]
    passed = all(record["product_identity"] for record in records)
    status = "PASS_EXACT" if passed else "FAIL_IMPLEMENTATION"
    local_path = run_dir / "raw" / "local_factors.json"
    write_json(local_path, {"task_id": "G-10", "run_id": run_id, "status": status, "scope": "maximal-order coincidence degree Q_M(x_j); a fixed torsion-free Gamma is controlled only by the retained C_Gamma comparison bounds", "ramified_prime_rule": "epsilon_j=1 for even j and 2^-4 for odd j", "records": records})
    growth = run_dir / "raw" / "coincidence_growth.parquet"
    pd.DataFrame([{key: value for key, value in row.items() if key not in {"factorization", "local_factors"}} for row in records]).to_parquet(growth, index=False)
    certificate = run_dir / "certificates" / "g10_coincidence_index_height.json"
    write_json(certificate, {"task_id": "G-10", "run_id": run_id, "status": status, "exact_records": len(records), "all_local_products_equal_global_degree": passed, "formula": "Q_M(x_j)=epsilon_j*(j^2+3)^2*F_j", "fixed_group_guard": "Exact q_j for an arbitrary fixed torsion-free subgroup requires its finite coset action and is not inferred from its index alone."})
    return status, {"raw": growth, "derived": local_path, "certificate": certificate}

