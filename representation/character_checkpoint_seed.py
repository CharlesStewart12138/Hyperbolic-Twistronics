from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from audit.data_io import write_json
from audit.run_manifest import sha256_file
from representation.b07_recovery_seed import _tree_inventory, inventory_digest


def _copy_verified(source: Path, target: Path) -> dict[str, object]:
    source_hash = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    target_hash = sha256_file(target)
    if target_hash != source_hash:
        raise RuntimeError(f"checkpoint copy hash mismatch: {source}")
    return {
        "source": source.as_posix(),
        "target": target.as_posix(),
        "sha256": source_hash,
        "bytes": target.stat().st_size,
    }


def seed_character_checkpoint(
    root: Path,
    run_dir: Path,
    run_id: str,
    config: dict[str, object],
) -> Path:
    checkpoint = config["character_recovery_checkpoint"]
    source_run_id = str(checkpoint["source_run_id"])
    source_run = root / "results" / source_run_id
    source_manifest_hash = sha256_file(source_run / "manifest.json")
    if source_manifest_hash != str(checkpoint["manifest_sha256"]):
        raise RuntimeError("character recovery checkpoint manifest hash differs")
    inventory = _tree_inventory(source_run)
    tree_hash = inventory_digest(inventory)
    if tree_hash != str(checkpoint["tree_inventory_sha256"]):
        raise RuntimeError("character recovery checkpoint tree hash differs")
    source_b07 = source_run / "certificates" / "b07_wedderburn_exact.json"
    if sha256_file(source_b07) != str(checkpoint["b07_certificate_sha256"]):
        raise RuntimeError("character recovery source B-07 certificate hash differs")
    source_b07_payload = json.loads(source_b07.read_text(encoding="utf-8"))
    if source_b07_payload.get("status") != "PASS_CERTIFIED" or not source_b07_payload.get("complete"):
        raise RuntimeError("character recovery source B-07 certificate is not complete and certified")
    source_spectra = source_run / "derived" / "wedderburn_block_spectra.parquet"
    if sha256_file(source_spectra) != str(checkpoint["block_spectra_sha256"]):
        raise RuntimeError("character recovery source block-spectrum hash differs")
    source_pairs = pd.read_parquet(source_spectra)[
        ["tower_id", "level", "rep_index", "raw_irrep_sha256", "raw_block_sha256"]
    ].drop_duplicates()
    if source_pairs.duplicated(["tower_id", "level", "rep_index"]).any():
        raise RuntimeError("character recovery source has ambiguous block/factor hash pairs")
    trusted_pairs: list[dict[str, object]] = []
    for row in source_pairs.itertuples(index=False):
        trusted_pairs.append(
            {
                "group": f"{row.tower_id}_level_{int(row.level)}",
                "rep_index": int(row.rep_index),
                "factor_sha256": str(row.raw_irrep_sha256),
                "block_sha256": str(row.raw_block_sha256),
            }
        )
    source_root = source_run / "raw" / "representation"
    target_root = run_dir / "raw" / "representation"
    target_root.mkdir(parents=True, exist_ok=False)
    copied: list[dict[str, object]] = []
    completed_groups = [str(value) for value in checkpoint["completed_groups"]]
    for group in completed_groups:
        source_group = source_root / group
        state = json.loads((source_group / "stage_state.json").read_text(encoding="utf-8"))
        if state.get("status") != "COMPLETE":
            raise RuntimeError(f"checkpoint group is not complete: {group}")
        target_group = target_root / group
        for source in sorted(source_group.rglob("*")):
            if source.is_file():
                copied.append(_copy_verified(source, target_group / source.relative_to(source_group)))
    trusted_pairs = [record for record in trusted_pairs if str(record["group"]) in completed_groups]
    expected_pair_count = 0
    for group in completed_groups:
        table = json.loads((target_root / group / "character_table.json").read_text(encoding="utf-8"))
        expected_pair_count += len(table["degrees"])
    if len(trusted_pairs) != expected_pair_count:
        raise RuntimeError("trusted source block/factor pair count is incomplete")
    for record in trusted_pairs:
        group = str(record["group"])
        index = int(record["rep_index"])
        target_group = target_root / group
        factor = target_group / "character_factors" / f"factor_{index:04d}.json"
        block = target_group / "blocks" / f"block_{index:04d}.h5"
        if sha256_file(factor) != str(record["factor_sha256"]):
            raise RuntimeError(f"trusted source factor hash mismatch after copy: {group}/{index}")
        if sha256_file(block) != str(record["block_sha256"]):
            raise RuntimeError(f"trusted source block hash mismatch after copy: {group}/{index}")
    partial_value = checkpoint.get("partial_group")
    partial_group = None if partial_value in (None, "") else str(partial_value)
    partial_files: list[str] = []
    partial_resume_stage = None
    if partial_group is not None:
        source_partial = source_root / partial_group
        target_partial = target_root / partial_group
        partial_files = [str(value) for value in checkpoint["partial_completed_files"]]
        for name in partial_files:
            copied.append(_copy_verified(source_partial / name, target_partial / name))
        table = json.loads((target_partial / "character_table.json").read_text(encoding="utf-8"))
        if table.get("status") != "COMPLETE" or sum(int(value) ** 2 for value in table["degrees"]) != int(
            table["order"]
        ):
            raise ArithmeticError("partial checkpoint character table is incomplete")
        write_json(
            target_partial / "stage_state.json",
            {
                "task_id": "B-07",
                "run_id": run_id,
                "action": partial_group + ".npz",
                "action_sha256": table["action_sha256"],
                "tower_id": table["tower_id"],
                "level": table["level"],
                "group_audit": "REUSED_BY_SHA256",
                "character_table": "REUSED_BY_SHA256",
                "current_stage": "compact_character_alignment",
                "last_completed_irrep": 0,
                "last_completed_block": 0,
                "total_irreps": len(table["degrees"]),
                "status": "RECOVERY_SEEDED",
            },
        )
        partial_resume_stage = "compact_character_alignment"
    seed = config["b07_recovery_seed"]
    p23 = target_root / str(seed["group"])
    for relative, expected_hash in seed["artifacts_sha256"].items():
        if sha256_file(p23 / str(relative)) != str(expected_hash):
            raise RuntimeError(f"p23 inherited seed hash mismatch: {relative}")
    certificate = run_dir / "certificates" / "b07_character_checkpoint_seed.json"
    write_json(
        certificate,
        {
            "task_id": "B-07",
            "run_id": run_id,
            "status": "RECOVERY_SEEDED_BY_SHA256",
            "source_run_id": source_run_id,
            "source_manifest_sha256": source_manifest_hash,
            "source_tree_inventory_sha256": tree_hash,
            "source_run_modified": False,
            "completed_groups_reused": completed_groups,
            "partial_group": partial_group,
            "partial_completed_files": partial_files,
            "partial_resume_stage": partial_resume_stage,
            "all_groups_complete": partial_group is None,
            "source_b07_certificate_sha256": sha256_file(source_b07),
            "source_block_spectra_sha256": sha256_file(source_spectra),
            "trusted_block_factor_pair_count": len(trusted_pairs),
            "trusted_block_factor_pairs": trusted_pairs,
            "copied_artifact_count": len(copied),
            "copied_bytes": sum(int(record["bytes"]) for record in copied),
            "copied_artifacts": copied,
            "reuse_method": "byte copy after source and destination SHA-256 verification",
            "hard_links_used": False,
        },
    )
    return certificate
