from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from word_automaton_v2 import parse_word_acceptor  # noqa: E402


def test_gap_internal_alphabet_and_continuation(tmp_path: Path) -> None:
    path = tmp_path / "gap_acceptor.txt"
    path.write_text(
        "\n".join(
            [
                "GAP_VERSION=4.16.0",
                "AUTOMATIC=true",
                "STATE_COUNT=2",
                "ALPHABET_SIZE=8",
                "INITIAL=1",
                "ACCEPTING=1,\\",
                ",2",
                "ALPHABET=_g1|_g2|_g3|_g4|_g5|_g6|_g7|_g8",
                "ROW=1|1,1,1,1,1,1,1,1",
                "ROW=2|2,2,2,2,2,2,2,2",
            ]
        ),
        encoding="utf-8",
    )
    acceptor = parse_word_acceptor(path)
    assert acceptor.alphabet_letters == (1, 2, 3, 4, -1, -2, -3, -4)
    assert acceptor.accepting == frozenset({0, 1})
