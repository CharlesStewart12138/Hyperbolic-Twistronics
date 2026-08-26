from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


TASKS = [f"B-{number:02d}" for number in range(1, 16)]
EXECUTION_ORDER = [
    "B-02", "B-03", "B-04", "B-05", "B-06", "B-08", "B-09",
    "B-10", "B-11", "B-12", "B-13", "B-14", "B-15",
]
BLOCKING = {"FAIL_THEORY", "FAIL_IMPLEMENTATION"}


def add_path(root: Path) -> None:
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def artifact_digest(path: Path) -> dict[str, object]:
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest

    if path.is_file():
        return {"kind": "file", "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if path.is_dir():
        inventory = _tree_inventory(path)
        return {
            "kind": "directory",
            "tree_inventory_sha256": inventory_digest(inventory),
            "file_count": len(inventory),
        }
    raise FileNotFoundError(path)


def update_manifest(root: Path, run_id: str, statuses: dict[str, str], outputs: dict[str, dict[str, Path]]) -> None:
    path = root / "TASK_MANIFEST.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        task = row["task_id"]
        if task not in statuses:
            continue
        row["status"] = statuses[task]
        row["run_id"] = run_id
        for field, key in (("raw_output", "raw"), ("derived_output", "derived"), ("certificate", "certificate")):
            value = outputs.get(task, {}).get(key)
            if value is not None:
                row[field] = value.relative_to(root).as_posix()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validation_rows(root: Path, run_id: str, statuses: dict[str, str], outputs: dict[str, dict[str, Path]]) -> list[dict[str, object]]:
    names = {
        "B-01": "finite-cover sparse spectra", "B-02": "lifted Weyl no loss",
        "B-03": "retained-sector no pollution", "B-04": "edge and gap transport",
        "B-05": "operator-tempered classification", "B-06": "cross-tower independence",
        "B-07": "complete resumable Wedderburn decomposition",
        "B-08": "single-character incompleteness", "B-09": "public projector crosscheck",
        "B-10": "common-space embedding", "B-11": "balanced full-shell tails",
        "B-12": "full-shell spectral inheritance", "B-13": "C0/C1/C2 derivative tiers",
        "B-14": "open-patch boundary control", "B-15": "injectivity-radius audit",
    }
    rows = []
    for task in TASKS:
        out = outputs.get(task, {})
        notes = ""
        if task in {"B-01", "B-07"}:
            notes = "Not rerun; reused only after complete frozen-source SHA-256 verification."
        elif statuses.get(task) == "INCONCLUSIVE":
            notes = "Inconclusive result preserved without relabelling."
        elif task == "B-08":
            notes = "Mandatory negative control; FAIL_EXPECTED is the preregistered success state."
        rows.append(
            {
                "theorem_id": "Theorems 128-133 / Definitions 44-45",
                "claim_name": names[task],
                "claim_layer": "finite covers and thermodynamic bulk",
                "model_level": "certified non-Abelian retained scalar bilayer cover family",
                "code_id": task,
                "run_id": run_id,
                "validation_type": statuses.get(task, "NOT_STARTED"),
                "parameter_set": "configs/phase_b_post_b07.yaml",
                "status": statuses.get(task, "NOT_STARTED"),
                "raw_data_file": out["raw"].relative_to(root).as_posix() if "raw" in out else "MISSING",
                "derived_data_file": out["derived"].relative_to(root).as_posix() if "derived" in out else "MISSING",
                "certificate_file": out["certificate"].relative_to(root).as_posix() if "certificate" in out else "MISSING",
                "notes": notes,
            }
        )
    return rows


def save_progress(run_dir: Path, statuses, outputs, errors) -> None:
    atomic_json(
        run_dir / "logs" / "phase_b_post_b07_progress.json",
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_statuses": statuses,
            "outputs": {task: {key: str(value) for key, value in out.items()} for task, out in outputs.items()},
            "errors": errors,
        },
    )


def save_task_checkpoint(root: Path, run_dir: Path, run_id: str, task: str, status: str, outputs) -> Path:
    from audit.data_io import write_json

    certificate = run_dir / "certificates" / "task_checkpoints" / f"{task}.json"
    certificate.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    for key, path in outputs.items():
        records[key] = {
            "path": path.relative_to(root).as_posix(),
            **artifact_digest(path),
        }
    write_json(
        certificate,
        {
            "task_id": task,
            "run_id": run_id,
            "status": status,
            "checkpointed_at_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": records,
        },
    )
    return certificate


def verify_source(root: Path, config: dict[str, object]) -> dict[str, object]:
    from audit.run_manifest import sha256_file
    from representation.b07_recovery_seed import _tree_inventory, inventory_digest

    expected = config["source_b07"]
    source_id = str(expected["run_id"])
    source = root / "results" / source_id
    if sha256_file(source / "manifest.json") != str(expected["manifest_sha256"]):
        raise RuntimeError("B07-complete source manifest hash changed")
    inventory = _tree_inventory(source)
    if len(inventory) != int(expected["file_count"]):
        raise RuntimeError("B07-complete source file count changed")
    if inventory_digest(inventory) != str(expected["tree_inventory_sha256"]):
        raise RuntimeError("B07-complete source tree hash changed")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("task_statuses") != {
        "B-01": "PASS_CONVERGED", "B-07": "PASS_CERTIFIED"
    }:
        raise RuntimeError("B07-complete source manifest status is incompatible")
    anchors = {
        "b01_reuse_certificate_sha256": source / "certificates" / "b01_preserved_reuse.json",
        "b07_certificate_sha256": source / "certificates" / "b07_wedderburn_exact.json",
        "b07_seed_certificate_sha256": source / "certificates" / "b07_character_checkpoint_seed.json",
        "block_spectra_sha256": source / "derived" / "wedderburn_block_spectra.parquet",
        "validation_matrix_sha256": source / "validation_matrix.parquet",
    }
    for key, path in anchors.items():
        if sha256_file(path) != str(expected[key]):
            raise RuntimeError(f"B07-complete source anchor hash changed: {key}")
    b01 = json.loads(anchors["b01_reuse_certificate_sha256"].read_text(encoding="utf-8"))
    if b01.get("status") != "PASS_CONVERGED" or b01.get("execution") != "NOT_RERUN":
        raise RuntimeError("source B-01 reuse certificate is incompatible")
    original_b01_certificate = root / str(b01["source_certificate"])
    original_b01_derived = root / str(b01["source_derived"])
    if sha256_file(original_b01_certificate) != b01["source_certificate_sha256"]:
        raise RuntimeError("original B-01 certificate hash changed")
    if sha256_file(original_b01_derived) != b01["source_derived_sha256"]:
        raise RuntimeError("original B-01 derived hash changed")
    original_b01_raw = root / "results" / str(b01["preserved_source_run_id"]) / "raw" / "finite_cover_spectra"
    actual_raw = {path.name: sha256_file(path) for path in sorted(original_b01_raw.glob("*.npz"))}
    if actual_raw != b01["source_raw_sha256"]:
        raise RuntimeError("original B-01 raw hashes changed")
    b07 = json.loads(anchors["b07_certificate_sha256"].read_text(encoding="utf-8"))
    seed = json.loads(anchors["b07_seed_certificate_sha256"].read_text(encoding="utf-8"))
    if b07.get("status") != "PASS_CERTIFIED" or not b07.get("complete"):
        raise RuntimeError("source B-07 certificate is not complete")
    if seed.get("trusted_block_factor_pair_count") != 154 or not seed.get("all_groups_complete"):
        raise RuntimeError("source B-07 seed does not contain all 154 hash pairs")
    blocks = pd.read_parquet(anchors["block_spectra_sha256"])
    pairs = blocks[["tower_id", "level", "rep_index", "raw_irrep_sha256", "raw_block_sha256"]].drop_duplicates()
    if len(pairs) != 154 or pairs.duplicated(["tower_id", "level", "rep_index"]).any():
        raise RuntimeError("source B-07 block spectrum is not representation-complete")
    trusted = {
        (str(row["group"]), int(row["rep_index"])): row
        for row in seed["trusted_block_factor_pairs"]
    }
    for row in pairs.itertuples(index=False):
        group = f"{row.tower_id}_level_{int(row.level)}"
        record = trusted.get((group, int(row.rep_index)))
        if record is None or record["factor_sha256"] != row.raw_irrep_sha256 or record["block_sha256"] != row.raw_block_sha256:
            raise RuntimeError(f"source B-07 trusted pair mismatch: {group}/{row.rep_index}")
        raw_group = source / "raw" / "representation" / group
        factor = raw_group / "character_factors" / f"factor_{int(row.rep_index):04d}.json"
        block = raw_group / "blocks" / f"block_{int(row.rep_index):04d}.h5"
        if sha256_file(factor) != row.raw_irrep_sha256 or sha256_file(block) != row.raw_block_sha256:
            raise RuntimeError(f"source B-07 raw pair changed: {group}/{row.rep_index}")
    quotient_dimensions = {}
    for quotient in b07["quotients"]:
        factorization = source / str(quotient["full_regular_spectrum_factorization"])
        if sha256_file(factorization) != quotient["full_regular_spectrum_factorization_sha256"]:
            raise RuntimeError("source exact regular factorization hash changed")
        if not quotient["passed"] or quotient["dimension_recombination"] != quotient["order"]:
            raise RuntimeError("source quotient recombination is incomplete")
        quotient_dimensions[(str(quotient["tower_id"]), int(quotient["level"]))] = int(quotient["order"])
    actual_dimensions = blocks.groupby(["tower_id", "level"]).regular_multiplicity.sum().to_dict()
    if actual_dimensions != quotient_dimensions:
        raise RuntimeError("source B-07 regular dimensions do not recombine")
    old_matrix = pd.read_parquet(anchors["validation_matrix_sha256"])
    statuses = old_matrix.set_index("code_id").status.to_dict()
    if statuses.get("B-01") != "PASS_CONVERGED" or statuses.get("B-07") != "PASS_CERTIFIED":
        raise RuntimeError("source B-01/B-07 validation matrix changed")
    if any(statuses.get(task) != "NOT_STARTED" for task in EXECUTION_ORDER):
        raise RuntimeError("source run unexpectedly executed post-B07 tasks")
    return {
        "source_id": source_id,
        "source": source,
        "inventory": inventory,
        "tree_hash": inventory_digest(inventory),
        "manifest_hash": sha256_file(source / "manifest.json"),
        "blocks": blocks,
        "b01": b01,
        "b07": b07,
        "b01_raw": original_b01_raw,
        "b01_derived": original_b01_derived,
    }


def write_reuse_certificates(root: Path, run_dir: Path, run_id: str, verified) -> dict[str, dict[str, Path]]:
    from audit.data_io import write_json

    source = verified["source"]
    b01_certificate = run_dir / "certificates" / "b01_verified_reuse.json"
    write_json(
        b01_certificate,
        {
            "task_id": "B-01", "run_id": run_id, "status": "PASS_CONVERGED",
            "execution": "NOT_RERUN", "source_run_id": verified["source_id"],
            "source_manifest_sha256": verified["manifest_hash"],
            "source_tree_inventory_sha256": verified["tree_hash"],
            "nested_source_reuse_certificate": verified["b01"], "source_modified": False,
        },
    )
    b07_certificate = run_dir / "certificates" / "b07_verified_reuse.json"
    write_json(
        b07_certificate,
        {
            "task_id": "B-07", "run_id": run_id, "status": "PASS_CERTIFIED",
            "execution": "NOT_RERUN", "source_run_id": verified["source_id"],
            "source_manifest_sha256": verified["manifest_hash"],
            "source_tree_inventory_sha256": verified["tree_hash"],
            "source_b07_certificate": (source / "certificates" / "b07_wedderburn_exact.json").relative_to(root).as_posix(),
            "source_block_spectra": (source / "derived" / "wedderburn_block_spectra.parquet").relative_to(root).as_posix(),
            "irrep_block_count": 154, "complete_exact_recombination_preserved": True,
            "source_modified": False,
        },
    )
    return {
        "B-01": {"raw": verified["b01_raw"], "derived": verified["b01_derived"], "certificate": b01_certificate},
        "B-07": {"raw": source / "raw" / "representation", "derived": source / "derived" / "wedderburn_block_spectra.parquet", "certificate": b07_certificate},
    }


def write_phase_checkpoint(root: Path, run_dir: Path, run_id: str, statuses, outputs, errors) -> str:
    blockers = {task: status for task, status in statuses.items() if status in BLOCKING}
    inconclusive = {task: status for task, status in statuses.items() if status == "INCONCLUSIVE"}
    complete = len(statuses) == len(TASKS) and not blockers
    state = "PHASE_B_COMPLETE" if complete else ("PHASE_B_BLOCKED" if blockers else "PHASE_B_PARTIAL")
    lines = [
        "# Phase B checkpoint (post-B07 continuation)", "", f"- Run ID: `{run_id}`",
        f"- State: `{state}`", "- B-01 rerun or modified: `false`",
        "- B-07 rerun or modified: `false`", "- Certified non-Abelian towers only: `true`",
        "- Scientific computation inside plotting scripts: `false`", "",
        "| Task | Status | Certificate |", "|---|---|---|",
    ]
    for task in TASKS:
        certificate = outputs.get(task, {}).get("certificate")
        relative = certificate.relative_to(root).as_posix() if certificate else "-"
        lines.append(f"| {task} | {statuses.get(task, 'NOT_STARTED')} | `{relative}` |")
    lines.extend(
        ["", f"Blockers: `{json.dumps(blockers, sort_keys=True)}`",
         f"Inconclusive: `{json.dumps(inconclusive, sort_keys=True)}`",
         "", "Scientific interpretation: bulk conclusions are restricted to certified retained sectors of the non-Abelian towers.",
         "Failed, expected-failure, and inconclusive tests remain in the immutable run.",
         "", "Next task: `D-01`" if complete else "Next task: resolve the recorded blocker.", ""]
    )
    text = "\n".join(lines)
    (run_dir / "derived" / "checkpoint_B.md").write_text(text, encoding="utf-8")
    (root / "reports" / "checkpoint_B.md").write_text(text, encoding="utf-8")
    project_path = root / "PROJECT_STATE.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project.update(
        {
            "state": state, "current_phase": "B", "latest_run_id": run_id,
            "latest_run_directory": run_dir.relative_to(root).as_posix(),
            "phase_b_task_statuses": statuses,
            "phase_b_b07_source_run_id": "618264e7923cf816783616b6dfc40fee0f8ca17a70a511610028fb0120841d47",
            "next_task": "D-01" if complete else next((task for task in TASKS if task not in statuses or statuses[task] in BLOCKING), "B-02"),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_json(project_path, project)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    add_path(root)

    from audit.data_io import write_json
    from audit.run_manifest import finalize_run, initialize_run
    from bulk.common_space_embedding import run as b10
    from bulk.cross_tower_independence import run as b06
    from bulk.derivative_tiers import run as b13
    from bulk.edge_gap_transport import run as b04
    from bulk.full_shell_balance import run as b11
    from bulk.full_shell_spectral_inheritance import run as b12
    from bulk.injectivity_radius_audit import run as b15
    from bulk.lifted_weyl_no_loss import run as b02
    from bulk.no_pollution_certificate import run as b03
    from bulk.open_patch_control import run as b14
    from bulk.operator_tempered_test import run as b05
    from representation.character_incompleteness import run as b08
    from representation.public_projector_crosscheck import run as b09

    config_path = root / "configs" / "phase_b_post_b07.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    if project.get("state") != "PHASE_B_B07_COMPLETE" or project.get("latest_run_id") != config["source_b07"]["run_id"]:
        raise SystemExit("project is not at the declared B07-complete checkpoint")
    gate_path = root / str(config["tower_gate"]["certificate"])
    levels_path = root / str(config["tower_gate"]["levels"])
    from audit.run_manifest import sha256_file
    if sha256_file(gate_path) != config["tower_gate"]["certificate_sha256"]:
        raise SystemExit("non-Abelian tower gate certificate hash changed")
    if sha256_file(levels_path) != config["tower_gate"]["levels_sha256"]:
        raise SystemExit("non-Abelian tower level hash changed")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_CERTIFIED" or not gate["theorem_certificate"]["r_inj_word_tends_to_infinity"]:
        raise SystemExit("non-Abelian tower gate is not certified")
    verified = verify_source(root, config)
    run_id, run_dir = initialize_run(root)
    statuses = {"B-01": "PASS_CONVERGED", "B-07": "PASS_CERTIFIED"}
    outputs = write_reuse_certificates(root, run_dir, run_id, verified)
    errors: dict[str, str] = {}
    save_task_checkpoint(root, run_dir, run_id, "B-01", statuses["B-01"], outputs["B-01"])
    save_task_checkpoint(root, run_dir, run_id, "B-07", statuses["B-07"], outputs["B-07"])
    context: dict[str, object] = {
        "blocks": verified["blocks"], "wedderburn_diagnostics": verified["b07"]["quotients"],
        "gate": gate, "gate_levels": pd.read_parquet(levels_path),
        "actions": sorted((root / str(config["tower_gate"]["raw_directory"])).glob("*.npz")),
    }
    functions = {
        "B-02": b02, "B-03": b03, "B-04": b04, "B-05": b05, "B-06": b06,
        "B-08": b08, "B-09": b09, "B-10": b10, "B-11": b11, "B-12": b12,
        "B-13": b13, "B-14": b14, "B-15": b15,
    }
    save_progress(run_dir, statuses, outputs, errors)
    for task in EXECUTION_ORDER:
        try:
            status, out = functions[task](config, run_dir, run_id, root, context)
            statuses[task] = status
            outputs[task] = out
            save_task_checkpoint(root, run_dir, run_id, task, status, out)
        except Exception:
            statuses[task] = "FAIL_IMPLEMENTATION"
            errors[task] = traceback.format_exc()
        save_progress(run_dir, statuses, outputs, errors)
        if statuses[task] in BLOCKING:
            break
    source_after = verify_source(root, config)
    if source_after["tree_hash"] != verified["tree_hash"] or source_after["inventory"] != verified["inventory"]:
        raise RuntimeError("frozen B07-complete source changed during Phase-B continuation")
    integrity = run_dir / "certificates" / "frozen_b07_source_integrity.json"
    write_json(
        integrity,
        {
            "status": "PASS_EXACT", "source_run_id": verified["source_id"],
            "before_equals_after": True, "manifest_sha256": verified["manifest_hash"],
            "tree_inventory_sha256": verified["tree_hash"], "file_count": len(verified["inventory"]),
        },
    )
    report = {
        "run_id": run_id, "source_b07_run_id": verified["source_id"],
        "execution_order": EXECUTION_ORDER, "task_statuses": statuses, "errors": errors,
        "b01_rerun": False, "b07_rerun": False, "source_modified": False,
        "certified_nonabelian_towers_only": True,
    }
    atomic_json(run_dir / "logs" / "phase_b_post_b07_execution_report.json", report)
    pd.DataFrame(validation_rows(root, run_id, statuses, outputs)).to_parquet(run_dir / "validation_matrix.parquet", index=False)
    update_manifest(root, run_id, statuses, outputs)
    state = write_phase_checkpoint(root, run_dir, run_id, statuses, outputs, errors)
    finalize_run(run_dir, "COMPLETE" if state == "PHASE_B_COMPLETE" else "INCOMPLETE", statuses)
    print(json.dumps({"run_id": run_id, "state": state, "task_statuses": statuses, "errors": errors}, indent=2))
    return 0 if state == "PHASE_B_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
