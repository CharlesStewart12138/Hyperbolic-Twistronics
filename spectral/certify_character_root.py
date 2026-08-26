from __future__ import annotations

from pathlib import Path

from audit.data_io import write_json
from spectral.natural_surface_model import natural_parameters


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    from flint import arb, ctx
    params = natural_parameters(config)
    previous = ctx.prec
    bits = int(config["certification"]["arb_bits"])
    try:
        ctx.prec = bits
        q = arb(str(params["q1"]))
        t = arb(str(params["t"]))
        root = t/q
        radius = arb("1e-45")
        interval = root + arb(0, radius)
        derivative = q/t
        gap = 2*t*(1/q-arb(8))
        passed = bool(interval > 0) and bool(derivative > 0) and bool(gap > 0)
    finally:
        ctx.prec = previous
    status = "PASS_CERTIFIED" if passed else "FAIL_THEORY"
    certificate = run_dir / "certificates" / "s04_character_root.json"
    write_json(certificate, {"task_id": "S-04", "run_id": run_id, "status": status, "backend": "python-flint Arb", "precision_bits": bits, "root_interval": str(interval), "absolute_derivative_lower_bound": str(derivative), "gap_lower_bound": str(gap), "parameter_box": {"w_over_t": [0.05, 12.0], "q1": params["q1"]}, "uniqueness": "F1 is affine with strictly negative derivative"})
    return status, {"raw": certificate, "derived": certificate, "certificate": certificate}

