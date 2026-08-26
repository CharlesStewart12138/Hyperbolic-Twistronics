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


TERMINAL_STATUSES = {
    "PASS_EXACT",
    "PASS_CERTIFIED",
    "PASS_CONVERGED",
    "FAIL_THEORY",
    "FAIL_IMPLEMENTATION",
    "INCONCLUSIVE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def tree_inventory(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def inventory_digest(inventory: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json(inventory)).hexdigest()


def hashes_under(root: Path, names: Iterable[str]) -> dict[str, str]:
    paths: list[Path] = []
    for name in names:
        base = root / name
        if base.exists():
            paths.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(set(paths))}


def software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "pandas", "pyarrow", "PyYAML", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    return versions


def load_extension_config(extension_root: Path) -> dict[str, object]:
    return yaml.safe_load((extension_root / "configs" / "extension.yaml").read_text(encoding="utf-8"))


def verify_parent(extension_root: Path) -> dict[str, object]:
    project_root = extension_root.parent
    config = load_extension_config(extension_root)
    parent = config["parent_project"]
    final_run = project_root / "results" / str(parent["final_run_id"])
    final_inventory = tree_inventory(final_run)
    final_status = json.loads((project_root / str(parent["final_status"])).read_text(encoding="utf-8"))
    checks = {
        "project_complete": final_status.get("state") == "PROJECT_COMPLETE" and final_status.get("project_complete") is True,
        "final_status_sha256": sha256_file(project_root / str(parent["final_status"])) == str(parent["final_status_sha256"]),
        "task_manifest_sha256": sha256_file(project_root / str(parent["task_manifest"])) == str(parent["task_manifest_sha256"]),
        "final_run_manifest_sha256": sha256_file(final_run / "manifest.json") == str(parent["final_run_manifest_sha256"]),
        "final_run_tree_inventory_sha256": inventory_digest(final_inventory) == str(parent["final_run_tree_inventory_sha256"]),
        "final_run_file_count": len(final_inventory) == int(parent["final_run_file_count"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen parent verification failed: {checks}")
    return {
        "checks": checks,
        "final_run_tree_inventory_sha256": inventory_digest(final_inventory),
        "final_run_file_count": len(final_inventory),
    }


def build_identity(extension_root: Path) -> tuple[str, dict[str, object]]:
    parent_verification = verify_parent(extension_root)
    identity = {
        "phase": "POSTVALIDATION_RESOLUTION",
        "extension_hashes": hashes_under(extension_root, ("configs", "src", "workflow", "tests")),
        "parent_verification": parent_verification,
        "software_versions": software_versions(),
    }
    return hashlib.sha256(canonical_json(identity)).hexdigest(), identity


def initialize_run(extension_root: Path) -> tuple[str, Path, dict[str, object]]:
    run_id, identity = build_identity(extension_root)
    run_dir = extension_root / "results" / run_id
    if run_dir.exists():
        raise FileExistsError(f"immutable extension run already exists: {run_dir}")
    for name in ("raw", "derived", "certificates", "logs", "figures", "figure_data", "reports", "exact"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": "POSTVALIDATION_RESOLUTION",
        "status": "RUNNING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "identity": identity,
        "randomness": "only preregistered deterministic seeds",
    }
    write_json(run_dir / "manifest.json", manifest)
    checksum_lines = [f"# run_id={run_id}"]
    checksum_lines.extend(f"{value}  {key}" for key, value in identity["extension_hashes"].items())
    (run_dir / "input_checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return run_id, run_dir, identity


def finalize_run(run_dir: Path, status: str, task_statuses: dict[str, str]) -> None:
    if status not in {"COMPLETE", "INCOMPLETE"}:
        raise ValueError(status)
    if any(value not in TERMINAL_STATUSES for value in task_statuses.values()):
        raise ValueError("nonterminal extension task status")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RUNNING":
        raise RuntimeError("run is not mutable")
    manifest["status"] = status
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["task_statuses"] = task_statuses
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
