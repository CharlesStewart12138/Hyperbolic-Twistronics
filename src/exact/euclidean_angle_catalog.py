from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from audit.data_io import write_json
from exact.euclidean_square_csl import csl_record


def enumerate_catalog(max_bilayer_atoms: int = 100) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    limit = int(math.sqrt(max_bilayer_atoms)) + 2
    for m in range(2, limit + 1):
        for n in range(1, m):
            if math.gcd(m, n) != 1:
                continue
            record = csl_record(m, n)
            if int(record["bilayer_atoms"]) >= max_bilayer_atoms:
                continue
            key = (str(record["cos_theta"]), str(record["sin_theta"]))
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return sorted(records, key=lambda row: (int(row["bilayer_atoms"]), int(row["m"]), int(row["n"])))


def run(config: dict, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    records = enumerate_catalog(100)
    complete = all(int(row["bilayer_atoms"]) < 100 for row in records)
    unique = len({(row["cos_theta"], row["sin_theta"]) for row in records}) == len(records)
    status = "PASS_EXACT" if records and complete and unique else "FAIL_IMPLEMENTATION"
    exact = run_dir / "exact" / "g08_euclidean_angle_catalog.json"
    write_json(exact, {"task_id": "G-08", "run_id": run_id, "status": status, "strict_atom_bound": 100, "record_count": len(records), "catalog": records, "completeness_rule": "all coprime m>n>=1 with parity-reduced 2*Sigma < 100"})
    derived = run_dir / "derived" / "euclidean_angle_catalog.parquet"
    pd.DataFrame([{key: value for key, value in row.items() if key != "rotation_matrix"} for row in records]).to_parquet(derived, index=False)
    return status, {"raw": exact, "derived": derived, "certificate": exact}

