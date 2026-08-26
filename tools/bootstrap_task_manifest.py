from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TASK_RE = re.compile(r"^(?P<task>[IGSBD]-\d{2})\s+(?P<path>\S+)\s*$")
NC_RE = re.compile(r"^(?P<task>NC-\d{2})\s*$")


def first_objective(lines: list[str], start: int) -> str:
    parts: list[str] = []
    for line in lines[start:]:
        clean = line.strip()
        if not clean:
            if parts:
                break
            continue
        if clean.startswith("---") or clean.startswith("==="):
            break
        if clean.rstrip(":") in {"Store", "Save", "Output", "Success", "Status"}:
            break
        parts.append(clean)
        if clean.endswith("."):
            break
    return " ".join(parts)


def parse_tasks(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    rows: list[dict[str, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        match = TASK_RE.match(stripped)
        if match:
            task_id = match.group("task")
            phase = task_id[0]
            rows.append(
                {
                    "task_id": task_id,
                    "phase": phase,
                    "code_path": match.group("path"),
                    "objective": first_objective(lines, index + 1),
                    "evidence_target": "declared_by_master_instruction",
                    "prerequisites": "execution_order_and_scientific_gates",
                    "status": "NOT_STARTED",
                    "run_id": "",
                    "raw_output": "",
                    "derived_output": "",
                    "certificate": "",
                    "notes": "",
                }
            )
            continue
        match = NC_RE.match(stripped)
        if match:
            rows.append(
                {
                    "task_id": match.group("task"),
                    "phase": "NC",
                    "code_path": "assigned_by_owner_task",
                    "objective": first_objective(lines, index + 1),
                    "evidence_target": "FAIL_EXPECTED",
                    "prerequisites": "owning_scientific_task",
                    "status": "NOT_STARTED",
                    "run_id": "",
                    "raw_output": "",
                    "derived_output": "",
                    "certificate": "",
                    "notes": "Mandatory negative control; never discard a failure.",
                }
            )
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task identifiers in master instruction")
    expected = {
        *(f"I-{n:02d}" for n in range(1, 11)),
        *(f"G-{n:02d}" for n in range(1, 16)),
        *(f"S-{n:02d}" for n in range(1, 25)),
        *(f"B-{n:02d}" for n in range(1, 16)),
        *(f"D-{n:02d}" for n in range(1, 16)),
        *(f"NC-{n:02d}" for n in range(1, 10)),
    }
    missing = expected - set(ids)
    extra = set(ids) - expected
    if missing or extra:
        raise ValueError(f"task manifest mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = parse_tasks(args.instruction.read_text(encoding="utf-8-sig"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

