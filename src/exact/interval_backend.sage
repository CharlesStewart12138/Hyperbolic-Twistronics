import json
import sys

bits = 192
RBF = RealBallField(bits)
x = RBF(2) / 3
identity = sin(x)^2 + cos(x)^2
positive = (sqrt(RBF(2)) - 1)^2
checks = {
    "contains_one": identity.contains_exact(1),
    "strictly_positive": positive.lower() > 0,
}
result = {
    "task_id": "I-09",
    "backend": "Sage RealBallField/Arb",
    "precision_bits": bits,
    "identity_ball": str(identity),
    "positive_ball": str(positive),
    "status": "PASS_CERTIFIED" if all(checks.values()) else "FAIL_IMPLEMENTATION",
    "checks": checks,
    "run_id": sys.argv[2] if len(sys.argv) > 2 else "UNSPECIFIED",
}
if len(sys.argv) > 1:
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
else:
    print(json.dumps(result, indent=2, sort_keys=True))

