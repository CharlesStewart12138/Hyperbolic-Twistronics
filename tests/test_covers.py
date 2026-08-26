from pathlib import Path

from covers.generate_cover_towers import generate


ROOT = Path(__file__).resolve().parents[1]


def test_three_inequivalent_cover_towers(tmp_path: Path) -> None:
    result = generate(ROOT / "configs" / "tower_definitions.yaml", tmp_path, "TEST_RUN")
    assert result["status"] == "PASS_EXACT"
    assert result["towers_declared"] == 3
    assert all(not row["bulk_gate_eligible"] for row in result["records"])

