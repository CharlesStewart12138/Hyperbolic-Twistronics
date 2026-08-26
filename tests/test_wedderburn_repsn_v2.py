from pathlib import Path

from representation import wedderburn_resumable as implementation
from representation.wedderburn_resumable import REP_BEGIN_RE
from representation.wedderburn_resumable_repsn_v2 import install, repsn_irrep_script


def test_repsn_v2_header_is_parser_compatible(tmp_path: Path) -> None:
    script = repsn_irrep_script(18, tmp_path / "irrep.part")
    assert "IsAffordingRepresentation(B07_IRR[idx],rep)" in script
    assert "AFFORDING_VERIFIED" in script
    assert REP_BEGIN_RE.fullmatch("REP_BEGIN index=18 degree=24")
    assert "backend=repsn" not in script


def test_install_v2_selects_parser_compatible_backend() -> None:
    install()
    assert implementation._irrep_script is repsn_irrep_script
