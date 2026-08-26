from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_version(command: str, arguments: list[str]) -> dict[str, str | None]:
    executable = shutil.which(command)
    if executable is None:
        return {"executable": None, "version": None}
    try:
        result = subprocess.run(
            [executable, *arguments], capture_output=True, text=True, check=False, timeout=30
        )
        output = (result.stdout or result.stderr).strip().splitlines()
        version = output[0] if output else f"exit={result.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        version = f"unavailable: {exc}"
    return {"executable": executable, "version": version}


def distribution_snapshot() -> list[dict[str, str]]:
    rows = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name") or "UNKNOWN"
        rows.append({"name": name, "version": dist.version})
    return sorted(rows, key=lambda row: row["name"].lower())


def snapshot(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "cpu_count": os.cpu_count(),
        "tools": {
            "git": command_version("git", ["--version"]),
            "sage": command_version("sage", ["--version"]),
            "gap": command_version("gap", ["-q", "-c", "Print(GAPInfo.Version);QUIT;"]),
            "snakemake": command_version("snakemake", ["--version"]),
        },
        "distributions": distribution_snapshot(),
    }


def lock_lines(data: dict[str, object]) -> list[str]:
    distributions = data["distributions"]
    assert isinstance(distributions, list)
    return [f"# run_id={data['run_id']}"] + [
        f"{row['name']}=={row['version']}" for row in distributions
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root-lock", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = snapshot(args.run_id)
    (args.output_dir / "environment.json").write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )
    content = "\n".join(lock_lines(data)) + "\n"
    (args.output_dir / "environment.lock").write_text(content, encoding="utf-8")
    if args.root_lock:
        args.root_lock.write_text(content, encoding="utf-8")
    print(args.output_dir / "environment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

