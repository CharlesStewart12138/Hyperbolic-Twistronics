from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_bounded_preview_ranges_are_frozen_and_complete_workbook_is_unchanged() -> None:
    recovery = yaml.safe_load(
        (ROOT / "configs" / "final_audit_workbook_render_recovery.yaml").read_text(encoding="utf-8")
    )
    contract = recovery["repair_contract"]
    assert contract["workbook_rows_must_remain_complete"] == 88
    assert contract["workbook_formula_scan_must_remain_complete"] is True
    assert contract["workbook_summary_reconciliation_must_remain_complete"] is True
    assert contract["preview_ranges"] == {
        "Summary": "A1:H16",
        "Validation Matrix": "A1:S14",
        "Error Budget": "A1:R14",
        "Provenance": "A1:H8",
    }


def test_workbook_builder_uses_exact_bounded_ranges() -> None:
    source = (ROOT / "src" / "audit" / "theorem_validation_workbook.mjs").read_text(encoding="utf-8")
    for sheet, cell_range in {
        "Summary": "A1:H16",
        "Validation Matrix": "A1:S14",
        "Error Budget": "A1:R14",
        "Provenance": "A1:H8",
    }.items():
        assert f'"{sheet}": "{cell_range}"' in source
    assert "autoCrop: \"all\"" not in source
    assert "rendered_ranges: previewRanges" in source
