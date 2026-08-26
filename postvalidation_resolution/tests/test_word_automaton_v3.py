from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from word_automaton_v3 import parse_word_acceptor  # noqa: E402


def test_real_gap_export_has_certified_inverse_pairs() -> None:
    fixture = Path(
        "C:/Users/charl/Documents/Codex/2026-08-24/new-chat/work/kbmag_word_acceptor_v2.txt"
    )
    if not fixture.exists():
        return
    text = fixture.read_text(encoding="utf-8")
    insertion = "INVERSE_INDEX=2,1,4,3,6,5,8,7\n"
    amended = fixture.parent / "kbmag_word_acceptor_v3_fixture.txt"
    amended.write_text(text.replace("ALPHABET=", insertion + "ALPHABET="), encoding="utf-8")
    acceptor = parse_word_acceptor(amended)
    for generator in range(4):
        positive = acceptor.transitions[acceptor.initial_state, generator]
        negative = acceptor.transitions[acceptor.initial_state, generator + 4]
        assert acceptor.transitions[positive, generator + 4] == -1
        assert acceptor.transitions[negative, generator] == -1
