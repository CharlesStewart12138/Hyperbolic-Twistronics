from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml


TERMINAL_TASK_STATUSES = {
    "PASS_EXACT",
    "PASS_CERTIFIED",
    "PASS_CONVERGED",
    "PASS_STRONG_BULK",
    "PASS_WEAK_BULK",
    "PASS_LOCAL_DOS",
    "PASS_PIECEWISE_DOS",
    "PASS_RESTRICTED_CLASS",
    "PASS_TWO_PARAMETER_CURVATURE",
    "PASS_THREE_FIELD",
    "INCONCLUSIVE",
    "FAIL_THEORY",
    "FAIL_IMPLEMENTATION",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected mapping in {path}")
    return value


def tree_inventory(directory: Path, *, exclude: Iterable[str] = ()) -> dict[str, str]:
    ignored = set(exclude)
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(part in ignored for part in path.parts)
    }


def inventory_digest(inventory: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json(inventory)).hexdigest()


def software_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "pandas", "pyarrow", "PyYAML", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "MISSING"
    return result


def validation_root(extension_root: Path) -> Path:
    return extension_root.parent


def _resolve_anchor(extension_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (extension_root / path).resolve()


def verify_frozen_anchors(extension_root: Path) -> dict[str, object]:
    config = load_yaml(extension_root / "configs" / "extension.yaml")
    anchors = config["frozen_anchors"]
    checks: dict[str, object] = {}
    for key, value in anchors.items():
        if key.endswith("_sha256"):
            continue
        expected_key = key + "_sha256"
        if expected_key not in anchors:
            continue
        path = _resolve_anchor(extension_root, str(value))
        actual = sha256_file(path)
        expected = str(anchors[expected_key]).lower()
        checks[key] = {"path": str(path), "expected": expected, "actual": actual, "pass": actual == expected}
    parent = config["parent_project"]
    parent_path = _resolve_anchor(extension_root, str(parent["final_status"]))
    parent_hash = sha256_file(parent_path)
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    checks["parent_project"] = {
        "path": str(parent_path),
        "expected": str(parent["final_status_sha256"]),
        "actual": parent_hash,
        "state": parent_payload.get("state"),
        "pass": parent_hash == str(parent["final_status_sha256"]) and parent_payload.get("state") == "PROJECT_COMPLETE",
    }
    prior = config["parent_extension"]
    prior_path = _resolve_anchor(extension_root, str(prior["final_status"]))
    prior_hash = sha256_file(prior_path)
    prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
    checks["parent_extension"] = {
        "path": str(prior_path),
        "expected": str(prior["final_status_sha256"]),
        "actual": prior_hash,
        "state": prior_payload.get("state"),
        "pass": prior_hash == str(prior["final_status_sha256"]) and prior_payload.get("extension_complete") is True,
    }
    failed = [key for key, value in checks.items() if not bool(value["pass"])]
    if failed:
        raise RuntimeError(f"frozen-anchor verification failed: {failed}")
    return checks


def build_identity(extension_root: Path, family: str) -> tuple[str, dict[str, object]]:
    anchor_checks = verify_frozen_anchors(extension_root)
    input_inventory: dict[str, str] = {}
    for directory in ("configs", "src", "workflow", "tests"):
        base = extension_root / directory
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                input_inventory[path.relative_to(extension_root).as_posix()] = sha256_file(path)
    identity = {
        "schema_version": 1,
        "phase": "DEEP_RESOLUTION",
        "family": family,
        "input_inventory": input_inventory,
        "input_inventory_sha256": inventory_digest(input_inventory),
        "anchor_checks": anchor_checks,
        "software": software_versions(),
    }
    run_id = hashlib.sha256(canonical_json(identity)).hexdigest()
    return run_id, identity


def initialize_run(extension_root: Path, family: str) -> tuple[str, Path, dict[str, object]]:
    run_id, identity = build_identity(extension_root, family)
    run_dir = extension_root / "results" / run_id
    if run_dir.exists():
        raise FileExistsError(f"immutable run already exists: {run_dir}")
    for name in ("raw", "derived", "certificates", "logs", "figure_data", "reports", "exact"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "phase": "DEEP_RESOLUTION",
        "family": family,
        "run_id": run_id,
        "status": "RUNNING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "identity": identity,
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_id, run_dir, identity


def finalize_run(run_dir: Path, task_statuses: dict[str, str], scientific_classification: str) -> dict[str, object]:
    invalid = {key: value for key, value in task_statuses.items() if value not in TERMINAL_TASK_STATUSES}
    if invalid:
        raise ValueError(f"nonterminal task statuses: {invalid}")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RUNNING":
        raise RuntimeError("run is already frozen")
    manifest["status"] = "COMPLETE"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["task_statuses"] = task_statuses
    manifest["scientific_classification"] = scientific_classification
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    inventory = tree_inventory(run_dir)
    freeze = {
        "run_id": manifest["run_id"],
        "family": manifest["family"],
        "scientific_classification": scientific_classification,
        "task_statuses": task_statuses,
        "file_count": len(inventory),
        "tree_inventory": inventory,
        "tree_inventory_sha256": inventory_digest(inventory),
    }
    write_json(run_dir / "freeze_certificate.json", freeze)
    return freeze


def record_failure(run_dir: Path, task: str, error: BaseException) -> None:
    import traceback

    write_json(
        run_dir / "certificates" / "failure.json",
        {
            "task": task,
            "status": "FAIL_IMPLEMENTATION",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    )

