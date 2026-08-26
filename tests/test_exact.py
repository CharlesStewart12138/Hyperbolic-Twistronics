from exact.commensurability_238 import exact_certificate as commensurability_certificate
from exact.octagon_group import exact_certificate as octagon_certificate


def test_octagon_relator_is_exact() -> None:
    result = octagon_certificate()
    assert result["status"] == "PASS_EXACT"
    assert result["checks"]["octagon_cycle_relator_identity"]


def test_238_quaternion_and_centered_sequence_are_exact() -> None:
    result = commensurability_certificate()
    assert result["status"] == "PASS_EXACT"
    assert result["centered_sequence"]["pass"]

