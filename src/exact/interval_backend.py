from __future__ import annotations

import argparse
import json
from pathlib import Path


def backend_certificate(bits: int = 192) -> dict[str, object]:
    try:
        from flint import arb, ctx
    except ImportError as exc:
        return {
            "task_id": "I-09",
            "status": "INCONCLUSIVE",
            "backend": "missing python-flint/Arb",
            "error": str(exc),
        }
    old_precision = ctx.prec
    try:
        ctx.prec = bits
        x = arb(2) / arb(3)
        identity = x.sin() ** 2 + x.cos() ** 2
        one = arb(1)
        contains_one = bool(identity.contains(one))
        radius = identity.rad()
        positive_interval = (arb(2).sqrt() - 1) ** 2
        positivity = bool(positive_interval > 0)
        status = "PASS_CERTIFIED" if contains_one and positivity else "FAIL_IMPLEMENTATION"
        return {
            "task_id": "I-09",
            "status": status,
            "backend": "python-flint Arb",
            "precision_bits": bits,
            "identity": "sin(x)^2+cos(x)^2 at x=2/3",
            "identity_ball": str(identity),
            "identity_radius": str(radius),
            "contains_exact_one": contains_one,
            "positive_ball": str(positive_interval),
            "positive_lower_bound_verified": positivity,
            "scope": "backend self-certificate only; later PASS_CERTIFIED tasks must save their own balls",
        }
    finally:
        ctx.prec = old_precision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bits", type=int, default=192)
    args = parser.parse_args()
    certificate = backend_certificate(args.bits)
    certificate["run_id"] = args.run_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0 if certificate["status"] in {"PASS_CERTIFIED", "INCONCLUSIVE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

