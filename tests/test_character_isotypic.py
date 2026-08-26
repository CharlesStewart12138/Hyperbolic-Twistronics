from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from representation.character_isotypic_decomposition import (
    _reused_factor_linked_block,
    _factor_script,
    _recombination_script,
    _roots,
    exact_walk_class_counts,
)


def test_exact_walk_class_counts_on_cyclic_regular_action() -> None:
    forward = np.array([1, 2, 3, 0], dtype=np.int64)
    inverse = np.array([3, 0, 1, 2], dtype=np.int64)
    permutations = np.stack([forward, inverse, forward, inverse, forward, inverse, forward, inverse])
    class_map = np.arange(4, dtype=np.int32)
    counts, identities = exact_walk_class_counts(permutations, class_map, 3)
    assert counts[0] == [0, 4, 0, 4]
    assert sum(counts[1]) == 8**2
    assert sum(counts[2]) == 8**3
    assert identities == [0, 32, 0]


def test_factor_script_uses_characters_and_newton_without_matrices(tmp_path) -> None:
    script = _factor_script(4, [[1, 2], [3, 4]], tmp_path / "factor.part")
    assert "B07_IRR[idx]" in script
    assert "elementary" in script
    assert "COEFFICIENT" in script
    assert "IrreducibleAffordingRepresentation" not in script
    assert "Image(rep" not in script
    assert "Elements(B07_G)" not in script


def test_recombination_script_checks_direct_regular_moments(tmp_path) -> None:
    script = _recombination_script([[1, 2], [3, 4]], [1, 2], 6, tmp_path / "recombine.part")
    assert "combined=direct" in script
    assert "B07_IRR[i][1]" in script
    assert "RECOMBINATION" in script


def test_high_precision_roots_preserve_exact_multiplicities() -> None:
    coefficients = [
        "1", "0", "-52", "-24", "854", "600", "-4628", "-936", "9585", "-5400", "0", "0"
    ]
    roots, imaginary_residual, characteristic_residual = _roots(coefficients)
    expected = np.sort(
        np.asarray([-4.701562118716424, 1.701562118716424, 0, 0, 1, 1, 5, 5, -3, -3, -3])
    )
    assert np.allclose(roots, expected, atol=1.0e-13, rtol=0.0)
    assert imaginary_residual < 1.0e-50
    assert characteristic_residual < 1.0e-10


def test_high_precision_roots_accept_real_cyclotomic_coefficient() -> None:
    roots, imaginary_residual, characteristic_residual = _roots(["1", "-E(5)-E(5)^4"])
    assert np.allclose(roots, [2.0 * np.cos(2.0 * np.pi / 5.0)], atol=1.0e-13, rtol=0.0)
    assert imaginary_residual < 1.0e-50
    assert characteristic_residual < 1.0e-12


def test_reused_character_block_is_accepted_only_by_exact_factor_hash(tmp_path: Path) -> None:
    factor = tmp_path / "factor.json"
    factor.write_text("{}", encoding="utf-8")
    from audit.run_manifest import sha256_file

    block = tmp_path / "block.h5"
    with h5py.File(block, "w") as handle:
        handle.attrs["status"] = "COMPLETE"
        handle.attrs["rep_index"] = 4
        handle.attrs["degree"] = 2
        handle.attrs["action_sha256"] = "action-hash"
        handle.attrs["exact_irrep_sha256"] = "legacy-explicit-irrep-hash"
        handle.attrs["exact_entry_count"] = 0
        handle.attrs["imaginary_residual"] = 0.0
        handle.attrs["characteristic_residual"] = 0.0
        handle.attrs["backend"] = "exact_character_isotypic_newton"
        handle.create_dataset("adjacency_eigenvalues", data=np.asarray([-1.0, 1.0]))
    trusted_pair = {
        "factor_sha256": sha256_file(factor),
        "block_sha256": sha256_file(block),
    }
    record = _reused_factor_linked_block(
        block,
        index=4,
        degree=2,
        action_hash="action-hash",
        factor_path=factor,
        trusted_source_pair=trusted_pair,
    )
    assert record["irrep_sha256"] == sha256_file(factor)
    assert record["degree"] == 2

    factor.write_text('{"changed": true}', encoding="utf-8")
    import pytest

    with pytest.raises(RuntimeError, match="not linked"):
        _reused_factor_linked_block(
            block,
            index=4,
            degree=2,
            action_hash="action-hash",
            factor_path=factor,
            trusted_source_pair=trusted_pair,
        )
