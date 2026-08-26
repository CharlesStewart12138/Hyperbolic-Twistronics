import json
import sys

K.<r2> = QuadraticField(2)
B.<iota,jmath> = QuaternionAlgebra(K, -3, r2)
chi_orb = QQ(2) - (1-QQ(1)/2) - (1-QQ(1)/3) - (1-QQ(1)/8)
checks = {
    "iota_square": iota*iota == -3,
    "jmath_square": jmath*jmath == r2,
    "anticommutation": iota*jmath == -jmath*iota,
    "orbifold_euler": chi_orb == -QQ(1)/24,
}
result = {
    "task_id": "I-05",
    "backend": "Sage exact QuaternionAlgebra",
    "status": "PASS_EXACT" if all(checks.values()) else "FAIL_IMPLEMENTATION",
    "checks": checks,
    "run_id": sys.argv[2] if len(sys.argv) > 2 else "UNSPECIFIED",
}
if len(sys.argv) > 1:
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
else:
    print(json.dumps(result, indent=2, sort_keys=True))

