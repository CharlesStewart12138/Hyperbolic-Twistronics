from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=600
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def fetch_git(resource: dict[str, object], destination: Path) -> dict[str, object]:
    revision = str(resource["revision"])
    repo = destination / "repo"
    if not repo.exists():
        repo.mkdir(parents=True)
        git("init", "--quiet", cwd=repo)
        git("remote", "add", "origin", str(resource["url"]), cwd=repo)
        git("fetch", "--depth", "1", "origin", revision, cwd=repo)
        git("checkout", "--detach", "--quiet", "FETCH_HEAD", cwd=repo)
    head = git("rev-parse", "HEAD", cwd=repo)
    if head != revision:
        raise RuntimeError(f"{resource['name']}: expected {revision}, found {head}")
    license_files = [
        path for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
        if (path := repo / name).exists()
    ]
    return {
        "name": resource["name"],
        "url": resource["url"],
        "revision": revision,
        "doi": resource.get("doi"),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "tree_sha256": tree_sha256(repo),
        "license_files": [
            {"path": path.name, "sha256": file_sha256(path)} for path in license_files
        ],
        "status": "PASS_EXTERNAL",
        "scope": "provenance_and_integrity_only; scientific agreement not yet tested",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--run-id", default="PRE_RUN_PROVENANCE")
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for resource in registry["resources"]:
        destination = args.public_root / resource["name"] / str(resource["revision"])
        if args.fetch:
            record = fetch_git(resource, destination)
        else:
            record = {
                **resource,
                "status": "INCONCLUSIVE",
                "scope": "registered_only; rerun with --fetch to verify content",
            }
        record["run_id"] = args.run_id
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"run_id": args.run_id, "resources": records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

