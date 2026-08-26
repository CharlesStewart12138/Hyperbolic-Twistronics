from __future__ import annotations

from pathlib import Path

import sympy as sp

from audit.data_io import write_json


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    r = sp.symbols("r", positive=True)
    c = (r**2-r+2)/(r*(r+1))
    rstar = 1 + sp.sqrt(2)
    minimum = sp.simplify(c.subs(r, rstar))
    derivative = sp.factor(sp.diff(c, r))
    alpha2 = sp.simplify((rstar**2 - 1) / 16)
    from flint import arb, ctx
    previous = ctx.prec
    try:
        ctx.prec = int(config["certification"]["arb_bits"])
        ball = 4 * arb(2).sqrt() - 5
        positive = bool(ball > 0)
    finally:
        ctx.prec = previous
    exact_pass = minimum == 4*sp.sqrt(2)-5 and alpha2 == (1+sp.sqrt(2))/8 and positive
    status = "PASS_EXACT" if exact_pass else "FAIL_IMPLEMENTATION"
    certificate = run_dir / "certificates" / "s02_square_no_go.json"
    write_json(certificate, {"task_id": "S-02", "run_id": run_id, "status": status, "domain": "alpha >= 0, equivalently r >= 1", "derivative_factorization": str(derivative), "minimizer_r": str(rstar), "minimizer_alpha_squared": str(alpha2), "sharp_minimum": str(minimum), "arb_positive_ball": str(ball), "no_positive_root": positive})
    return status, {"raw": certificate, "derived": certificate, "certificate": certificate}

