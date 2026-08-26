from exact.interval_backend import backend_certificate


def test_interval_backend_is_honest() -> None:
    result = backend_certificate(128)
    assert result["status"] in {"PASS_CERTIFIED", "INCONCLUSIVE"}
    if result["status"] == "PASS_CERTIFIED":
        assert result["contains_exact_one"]
        assert result["positive_lower_bound_verified"]

