from pathlib import Path

from tools.bootstrap_task_manifest import parse_tasks


ROOT = Path(__file__).resolve().parents[1]


def test_master_instruction_has_all_atomic_tasks() -> None:
    text = (ROOT / "references" / "MASTER_EXECUTION_INSTRUCTION.txt").read_text(encoding="utf-8-sig")
    rows = parse_tasks(text)
    assert len(rows) == 88
    assert {row["task_id"] for row in rows} >= {"I-01", "G-15", "S-24", "B-15", "D-15", "NC-09"}

