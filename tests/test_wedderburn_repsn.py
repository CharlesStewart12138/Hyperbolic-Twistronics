from pathlib import Path

from representation import wedderburn_resumable as implementation
from representation.wedderburn_resumable_repsn import install, repsn_irrep_script


def test_repsn_script_requires_full_affording_verification(tmp_path: Path) -> None:
    script = repsn_irrep_script(18, tmp_path / "irrep.part")
    assert 'LoadPackage("repsn")' in script
    assert "IrreducibleAffordingRepresentation(B07_IRR[idx])" in script
    assert "IsAffordingRepresentation(B07_IRR[idx],rep)" in script
    assert "AFFORDING_VERIFIED" in script
    assert "affordingVerified<>true" in script
    assert "GEN_ENTRY" in script


def test_install_selects_repsn_backend() -> None:
    install()
    assert implementation._irrep_script is repsn_irrep_script
