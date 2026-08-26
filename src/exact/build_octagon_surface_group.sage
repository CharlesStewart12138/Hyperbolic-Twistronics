import json
import sys

AA = AlgebraicRealField()
c = AA(1) + AA(2).sqrt()
s = (c*c - 1).sqrt()
B = matrix(AA, [[c, s], [s, c]])

def rot(angle):
    return matrix(AA, [[cos(angle/2), sin(angle/2)], [-sin(angle/2), cos(angle/2)]])

generators = []
for k in range(4):
    R = rot(AA.pi() * k / 4)
    generators.append(R * B * R.inverse())

word = [1, -2, 3, -4, -1, 2, -3, 4]
value = identity_matrix(AA, 2)
for letter in word:
    g = generators[abs(letter)-1]
    value *= g if letter > 0 else g.inverse()

passed = all(g.det() == 1 for g in generators) and value == identity_matrix(AA, 2)
result = {
    "task_id": "I-04",
    "backend": "Sage AlgebraicRealField",
    "status": "PASS_EXACT" if passed else "FAIL_IMPLEMENTATION",
    "relator_word": word,
    "run_id": sys.argv[2] if len(sys.argv) > 2 else "UNSPECIFIED",
}
if len(sys.argv) > 1:
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
else:
    print(json.dumps(result, indent=2, sort_keys=True))

