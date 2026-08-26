from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def generator_permutation(modulus: int, coordinate: int, sign: int) -> np.ndarray:
    index = modulus**4
    states = np.arange(index, dtype=np.int64)
    stride = modulus**coordinate
    digit = (states // stride) % modulus
    target = states + sign * stride
    if sign > 0:
        target = np.where(digit == modulus - 1, target - modulus * stride, target)
    else:
        target = np.where(digit == 0, target + modulus * stride, target)
    return target


def materialize_level(prime: int, level: int, output: Path, run_id: str) -> dict[str, object]:
    modulus = prime**level
    index = modulus**4
    positive = np.stack([generator_permutation(modulus, coordinate, 1) for coordinate in range(4)])
    negative = np.stack([generator_permutation(modulus, coordinate, -1) for coordinate in range(4)])
    permutations = np.concatenate([positive, negative], axis=0)
    if any(np.unique(row).size != index for row in permutations):
        raise RuntimeError("coset action is not a permutation")
    for coordinate in range(4):
        if not np.array_equal(negative[coordinate][positive[coordinate]], np.arange(index)):
            raise RuntimeError("inverse coset action mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, run_id=np.array(run_id), modulus=modulus, permutations=permutations)
    systole = min(modulus, 4)
    return {
        "run_id": run_id,
        "prime": prime,
        "level": level,
        "modulus": modulus,
        "index": index,
        "genus": 1 + index,
        "euler_characteristic": -2 * index,
        "normal": True,
        "quotient": f"(Z/{modulus}Z)^4",
        "presentation": "kernel of Gamma_2 -> (Z/mZ)^4 induced by abelianization",
        "word_systole_exact": systole,
        "injectivity_radius_word_lower": (systole - 1) // 2,
        "injectivity_radius_word_upper": systole / 2,
        "coset_table_file": output.name,
        "bulk_gate_eligible": False,
        "bulk_gate_reason": "commutator word of length four remains in every abelian tower",
    }


def generate(config: Path, output_dir: Path, run_id: str) -> dict[str, object]:
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    records = []
    for tower in data["towers"]:
        for level in tower["materialized_levels"]:
            name = f"{tower['tower_id']}_level_{level}.npz"
            record = materialize_level(
                int(tower["quotient_prime"]), int(level), output_dir / name, run_id
            )
            record["tower_id"] = tower["tower_id"]
            records.append(record)
    summary = {
        "task_id": "I-06",
        "run_id": run_id,
        "status": "PASS_EXACT",
        "towers_declared": len(data["towers"]),
        "levels_materialized": len(records),
        "records": records,
        "scientific_gate": "infrastructure only; all three abelian towers are rejected for later thermodynamic claims",
    }
    (output_dir / "cover_towers.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(args.config, args.output_dir, args.run_id)
    print(json.dumps({"status": summary["status"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

