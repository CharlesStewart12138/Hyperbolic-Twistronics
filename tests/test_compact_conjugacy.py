from __future__ import annotations

import numpy as np

from covers.nonabelian_congruence_towers import enumerate_marked_group, matrix_inverse, matrix_multiply
from representation.compact_conjugacy import (
    _parse_alignment,
    compact_conjugacy_classes,
    representative_words,
)


def test_compact_conjugacy_classes_for_sl2_f2() -> None:
    modulus = 2
    identity = (1, 0, 0, 1)
    swap = (0, 1, 1, 0)
    shear = (1, 1, 0, 1)
    generators = (swap, shear, identity, identity)
    elements = enumerate_marked_group(generators, modulus)
    assert len(elements) == 6
    index = {element: number for number, element in enumerate(elements)}
    moves = generators + tuple(matrix_inverse(generator, modulus) for generator in generators)
    permutations = np.stack(
        [
            np.fromiter(
                (index[matrix_multiply(element, move, modulus)] for element in elements),
                dtype=np.int32,
                count=len(elements),
            )
            for move in moves
        ]
    )
    class_map, representatives, sizes = compact_conjugacy_classes(
        np.asarray(elements, dtype=np.int64), permutations, modulus
    )
    assert sorted(int(value) for value in sizes) == [1, 2, 3]
    assert class_map.shape == (6,)
    words = representative_words(permutations, representatives)
    assert len(words) == 3
    for representative, word in zip(representatives, words, strict=True):
        endpoint = 0
        for letter in word:
            move_index = letter - 1 if letter > 0 else 4 + (-letter - 1)
            endpoint = int(permutations[move_index, endpoint])
        assert endpoint == int(representative)


def test_compact_alignment_returns_gap_class_permutation(tmp_path) -> None:
    alignment = tmp_path / "alignment.txt"
    alignment.write_text(
        "\n".join(
            [
                "CLASS_ALIGNMENT compact=1 gap=1 compact_size=1 gap_size=1",
                "CLASS_ALIGNMENT compact=2 gap=3 compact_size=2 gap_size=2",
                "CLASS_ALIGNMENT compact=3 gap=2 compact_size=3 gap_size=3",
                "CHAR_VALUE rep=1 class=1 value=1",
                "CHAR_VALUE rep=1 class=2 value=7",
                "CHAR_VALUE rep=1 class=3 value=5",
            ]
        ),
        encoding="utf-8",
    )
    characters, compact_to_gap = _parse_alignment(
        alignment, np.asarray([0, 1, 1, 2, 2, 2]), np.asarray([1, 2, 3]), [1]
    )
    assert characters == [["1", "7", "5"]]
    assert compact_to_gap.tolist() == [0, 2, 1]
    assert compact_to_gap[np.asarray([0, 1, 1, 2, 2, 2])].tolist() == [0, 2, 2, 1, 1, 1]
