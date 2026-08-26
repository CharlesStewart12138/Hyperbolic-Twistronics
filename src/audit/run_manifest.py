from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HASH_ROOTS = ("configs", "src", "workflow", "environment", "tools", "tests")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes_under(root: Path, directories: tuple[str, ...]) -> dict[str, str]:
    files: list[Path] = []
    for directory in directories:
        base = root / directory
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(set(files))}


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "NO_GIT_COMMIT"


def software_versions() -> dict[str, str]:
    names = ["numpy", "scipy", "pandas", "sympy", "h5py", "pyarrow", "python-flint", "PyYAML"]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "MISSING"
    return versions


def public_hashes(root: Path) -> dict[str, str]:
    provenance = root / "public_data" / "provenance.json"
    if not provenance.exists():
        return {}
    data = json.loads(provenance.read_text(encoding="utf-8"))
    return {
        str(row["name"]): str(row.get("tree_sha256", "NOT_FETCHED"))
        for row in data.get("resources", [])
    }


def canonical_json(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_identity(root: Path) -> tuple[str, dict[str, object]]:
    identity = {
        "git_commit": git_commit(root),
        "config_and_source_hashes": hashes_under(root, HASH_ROOTS),
        "public_data_hashes": public_hashes(root),
        "software_versions": software_versions(),
    }
    run_id = hashlib.sha256(canonical_json(identity)).hexdigest()
    return run_id, identity


def initialize_run(root: Path) -> tuple[str, Path]:
    run_id, identity = build_identity(root)
    run_dir = root / "results" / run_id
    if run_dir.exists():
        raise FileExistsError(f"immutable run already exists: {run_dir}")
    for name in ("raw", "derived", "certificates", "logs", "exact"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": run_id,
        "status": "RUNNING",
        "created_at_utc": now,
        "git_commit": identity["git_commit"],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "identity_payload": identity,
        "random_seed": None,
        "solver_tolerance": None,
        "precision": None,
        "data_provenance": "references/ and public_data/provenance.json",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    frozen = {
        path: sha for path, sha in identity["config_and_source_hashes"].items()
        if str(path).startswith("configs/")
    }
    (run_dir / "config_frozen.yaml").write_text(
        f"# run_id: {run_id}\n" + "\n".join(f"{key}: {value}" for key, value in frozen.items()) + "\n",
        encoding="utf-8",
    )
    checks = [f"# run_id={run_id}"]
    checks.extend(f"{sha}  {path}" for path, sha in identity["config_and_source_hashes"].items())
    checks.extend(f"{sha}  public:{name}" for name, sha in identity["public_data_hashes"].items())
    (run_dir / "input_checksums.sha256").write_text("\n".join(checks) + "\n", encoding="utf-8")
    return run_id, run_dir


def finalize_run(run_dir: Path, status: str, task_statuses: dict[str, str]) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RUNNING":
        raise RuntimeError("only a RUNNING manifest may be finalized")
    manifest["status"] = status
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["task_statuses"] = task_statuses
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    run_id, run_dir = initialize_run(args.project_root.resolve())
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

