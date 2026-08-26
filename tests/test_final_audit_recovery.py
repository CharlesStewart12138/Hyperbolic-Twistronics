from __future__ import annotations

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_blank_derived_contract_is_exact() -> None:
    recovery = yaml.safe_load((ROOT / "configs" / "final_audit_recovery.yaml").read_text(encoding="utf-8"))
    expected = {f"I-{index:02d}" for index in range(1, 11)}
    assert set(recovery["repair_contract"]["tasks_with_blank_legacy_derived_output"]) == expected
    with (ROOT / "TASK_MANIFEST.csv").open("r", encoding="utf-8", newline="") as handle:
        actual = {row["task_id"] for row in csv.DictReader(handle) if not row["derived_output"]}
    assert actual == expected


def test_recovery_is_audit_only() -> None:
    source = (ROOT / "workflow" / "run_final_audit_recovery.py").read_text(encoding="utf-8")
    assert 'from audit.final_global_audit import run as final_audit' in source
    assert "scientific_tasks_rerun\": False" in source
    forbidden_imports = [
        "from analysis.magic_subsequence_sampling",
        "from analysis.magic_complexity",
        "from geometry.incommensurate_joint_limit",
        "from spectral.operational_magic_metrics",
    ]
    assert not any(item in source for item in forbidden_imports)


def test_final_audit_uses_scientific_source_and_explicit_legacy_rule() -> None:
    source = (ROOT / "src" / "audit" / "final_global_audit.py").read_text(encoding="utf-8")
    assert 'item["derived_output"] or "NOT_APPLICABLE"' in source
    assert "tasks_with_blank_legacy_derived_output" in source
    assert "scientific_source_run_dir" in source
