from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from cover_towers import (  # noqa: E402
    RELATOR,
    enumerate_marked_group,
    reduction_map,
)
from dyadic_ring import (  # noqa: E402
    build_ring,
    evaluate_word,
    expected_sl2_order,
    marked_generators,
)
from word_automaton import parse_word_acceptor  # noqa: E402


def test_ramified_ring_sizes_and_defining_relation() -> None:
    for depth in range(1, 7):
        ring = build_ring(depth)
        assert ring.size == 2**depth
        x2 = ring.mul(ring.x, ring.x)
        x4 = ring.mul(x2, x2)
        relation = ring.sub(ring.sub(x4, ring.add(x2, x2)), ring.one)
        assert relation == ring.zero


def test_marked_generators_relator_and_exact_small_group_orders() -> None:
    identity_by_depth = []
    for depth in range(1, 4):
        ring = build_ring(depth)
        generators = marked_generators(ring)
        identity = (ring.one, ring.zero, ring.zero, ring.one)
        assert evaluate_word(ring, generators, RELATOR) == identity
        group = enumerate_marked_group(
            "dyadic_ramified",
            ring,
            with_auxiliary_p7=False,
            maximum_order=10_000,
        )
        assert expected_sl2_order(depth) % group.order == 0
        assert group.order > 1
        identity_by_depth.append(group)
    for parent, child in zip(identity_by_depth, identity_by_depth[1:]):
        parent_index = reduction_map(child, parent)
        counts = np.bincount(parent_index, minlength=parent.order)
        assert np.all(counts == child.order // parent.order)


def test_word_acceptor_parser_reorders_generators(tmp_path: Path) -> None:
    fixture = tmp_path / "acceptor.txt"
    fixture.write_text(
        "\n".join(
            [
                "GAP_VERSION=fixture",
                "AUTOMATIC=true",
                "STATE_COUNT=1",
                "ALPHABET_SIZE=8",
                "INITIAL=1",
                "ACCEPTING=1",
                "ALPHABET=a|a^-1|b|b^-1|c|c^-1|d|d^-1",
                "ROW=1|1,1,1,1,1,1,1,1",
            ]
        ),
        encoding="utf-8",
    )
    acceptor = parse_word_acceptor(fixture)
    assert acceptor.state_count == 1
    assert acceptor.alphabet_letters == (1, 2, 3, 4, -1, -2, -3, -4)
    assert np.array_equal(acceptor.transitions, np.zeros((1, 8), dtype=np.int32))
