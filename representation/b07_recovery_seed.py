from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from audit.data_io import write_json
from audit.run_manifest import sha256_file
from representation.lazy_word_representation import LazyWordRepresentation


def _tree_inventory(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def inventory_digest(inventory: dict[str, str]) -> str:
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def audit_frozen_runs(root: Path, config: dict[str, object]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for expected in config["preserved_failed_runs"]:
        run_id = str(expected["run_id"])
        run_dir = root / "results" / run_id
        actual_manifest = sha256_file(run_dir / "manifest.json")
        if actual_manifest != str(expected["manifest_sha256"]):
            raise RuntimeError(f"frozen predecessor manifest changed: {run_id}")
        inventory = _tree_inventory(run_dir)
        tree_hash = inventory_digest(inventory)
        expected_tree_hash = expected.get("tree_inventory_sha256")
        if expected_tree_hash is not None and tree_hash != str(expected_tree_hash):
            raise RuntimeError(f"frozen predecessor tree changed: {run_id}")
        records.append(
            {
                "run_id": run_id,
                "manifest_sha256": actual_manifest,
                "file_count": len(inventory),
                "tree_inventory_sha256": tree_hash,
                "inventory": inventory,
            }
        )
    return {"runs": records}


def verify_b01_hashes(root: Path, config: dict[str, object]) -> None:
    expected = config["preserved_b01"]
    source = root / "results" / str(expected["run_id"])
    checks = {
        "certificate_sha256": sha256_file(source / "certificates" / "b01_finite_cover_spectra.json"),
        "derived_sha256": sha256_file(source / "derived" / "b01_finite_cover_edge_spectra.parquet"),
    }
    for key, value in checks.items():
        if value != str(expected[key]):
            raise RuntimeError(f"B-01 {key} does not match the recovery configuration")
    raw_dir = source / "raw" / "finite_cover_spectra"
    actual_raw = {path.name: sha256_file(path) for path in sorted(raw_dir.glob("*.npz"))}
    if actual_raw != {str(key): str(value) for key, value in expected["raw_sha256"].items()}:
        raise RuntimeError("B-01 raw spectra hashes do not match the recovery configuration")


def seed_recovery_artifacts(
    root: Path,
    run_dir: Path,
    run_id: str,
    config: dict[str, object],
    frozen_before: dict[str, object],
) -> Path:
    seed = config["b07_recovery_seed"]
    source_run_id = str(seed["source_run_id"])
    group = str(seed["group"])
    source = root / "results" / source_run_id / "raw" / "representation" / group
    target = run_dir / "raw" / "representation" / group
    target.mkdir(parents=True, exist_ok=False)
    degrees = [int(value) for value in seed["degrees"]]
    if len(degrees) != 27 or sum(value * value for value in degrees) != 12144:
        raise ArithmeticError("the recorded p23 character degrees are incomplete")
    table = json.loads((source / "character_table.json").read_text(encoding="utf-8"))
    if [int(value) for value in table["degrees"]] != degrees:
        raise ArithmeticError("source character table degrees differ from the recovery record")
    copied: list[dict[str, object]] = []
    for relative, expected_hash in seed["artifacts_sha256"].items():
        relative_path = Path(str(relative))
        source_path = source / relative_path
        actual_hash = sha256_file(source_path)
        if actual_hash != str(expected_hash):
            raise RuntimeError(f"source artifact hash mismatch: {relative}")
        target_path = target / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        target_hash = sha256_file(target_path)
        if target_hash != actual_hash:
            raise RuntimeError(f"copied artifact hash mismatch: {relative}")
        copied.append(
            {
                "relative_path": relative_path.as_posix(),
                "source_sha256": actual_hash,
                "target_sha256": target_hash,
                "bytes": target_path.stat().st_size,
            }
        )
    state = {
        "task_id": "B-07",
        "run_id": run_id,
        "action": group + ".npz",
        "action_sha256": table["action_sha256"],
        "tower_id": table["tower_id"],
        "level": table["level"],
        "current_stage": "individual_irreps",
        "group_audit": "RECOMPUTE_IN_RECOVERY_RUN",
        "character_table": "REUSED_BY_SHA256",
        "total_irreps": 27,
        "last_completed_irrep": 3,
        "last_completed_block": 3,
        "next_irrep": 4,
        "status": "RECOVERY_SEEDED",
    }
    write_json(target / "stage_state.json", state)
    counts = Counter(degrees)
    remaining = degrees[3:]
    certificate = run_dir / "certificates" / "b07_recovery_seed.json"
    write_json(
        certificate,
        {
            "task_id": "B-07",
            "run_id": run_id,
            "status": "RECOVERY_SEEDED",
            "source_run_id": source_run_id,
            "source_run_modified": False,
            "reuse_method": "byte copy after source SHA-256 verification and destination SHA-256 verification",
            "hard_links_used": False,
            "group": group,
            "resume_exactly_at_irrep": 4,
            "completed_irreps": [1, 2, 3],
            "complete_degree_list_before_execution": degrees,
            "degree_counts": {str(key): value for key, value in sorted(counts.items())},
            "degree_square_sum": sum(value * value for value in degrees),
            "remaining_irrep_count": len(remaining),
            "remaining_degree_square_weight": sum(value * value for value in remaining),
            "remaining_degree_cube_weight": sum(value**3 for value in remaining),
            "maximum_remaining_hard_budget_seconds": len(remaining) * int(config["gap_backend"]["recovery_timeout_policy"]["hard_seconds"]),
            "cost_estimate_note": "degree-square/cube weights are workload proxies; Repsn construction cost is representation-dependent and not assumed linear",
            "copied_artifacts": copied,
            "frozen_predecessors_before": frozen_before,
        },
    )
    return certificate


def audit_materialization(run_dir: Path, config: dict[str, object]) -> Path:
    group = str(config["b07_recovery_seed"]["group"])
    block_dir = run_dir / "raw" / "representation" / group / "blocks"
    audits: list[dict[str, object]] = []
    allowed_group = {
        *(f"generator_{index}" for index in range(1, 5)),
        *(f"generator_{index}_inverse" for index in range(1, 5)),
    }
    derived = {"adjacency_matrix", "adjacency_eigenvalues"}
    for path in sorted(block_dir.glob("block_*.h5")):
        with h5py.File(path, "r") as handle:
            datasets = set(handle.keys())
        unexpected = datasets - allowed_group - derived
        if unexpected:
            raise RuntimeError(f"unexpected materialized group-element dataset(s) in {path.name}: {unexpected}")
        audits.append(
            {
                "block": path.name,
                "block_sha256": sha256_file(path),
                "generator_and_inverse_datasets": sorted(datasets & allowed_group),
                "non_generator_group_element_datasets": [],
                "derived_hamiltonian_datasets": sorted(datasets & derived),
            }
        )
    probe = LazyWordRepresentation.from_block(block_dir / "block_0001.h5", maximum_cached_words=8)
    before = probe.cached_words
    first = probe.evaluate((1, 2, -1))
    after_first = probe.cached_words
    second = probe.evaluate((1, 2, -1))
    if not np.array_equal(first, second) or (1, 2, -1) not in after_first:
        raise ArithmeticError("lazy word cache regression probe failed")
    certificate = run_dir / "certificates" / "b07_matrix_materialization.json"
    write_json(
        certificate,
        {
            "task_id": "B-07",
            "status": "PASS_POLICY_AUDIT",
            "policy": "only four generator and four inverse matrices are persisted per irrep",
            "all_group_element_matrices_materialized": False,
            "non_generator_evaluation": "on demand from reduced words with bounded LRU cache",
            "lazy_probe": {
                "word": [1, 2, -1],
                "cache_before": [list(word) for word in before],
                "cache_after_first_evaluation": [list(word) for word in after_first],
                "second_evaluation_reused_cached_value": first is second,
                "materialized_generator_and_inverse_count": probe.materialized_group_element_count,
            },
            "audited_blocks": audits,
        },
    )
    return certificate
