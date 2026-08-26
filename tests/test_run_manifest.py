from pathlib import Path

from audit.run_manifest import build_identity


def test_run_identity_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.yaml").write_text("x: 1\n", encoding="utf-8")
    first, first_payload = build_identity(tmp_path)
    second, second_payload = build_identity(tmp_path)
    assert first == second
    assert first_payload == second_payload
    assert len(first) == 64

