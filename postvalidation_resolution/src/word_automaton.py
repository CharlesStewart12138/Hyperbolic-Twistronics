from __future__ import annotations

import os
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import sha256_file


def to_cygwin(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.drive:
        raise ValueError("Windows drive required")
    return "/cygdrive/" + resolved.drive[0].lower() + resolved.as_posix()[2:]


@dataclass(frozen=True)
class WordAcceptor:
    transitions: np.ndarray
    initial_state: int
    accepting: frozenset[int]
    alphabet_letters: tuple[int, ...]
    gap_version: str
    source_sha256: str

    @property
    def state_count(self) -> int:
        return int(self.transitions.shape[0])


def _letter(label: str) -> int:
    compact = label.replace(" ", "")
    match = re.fullmatch(r"([abcd])(?:\^-1)?", compact)
    if match is None:
        raise ValueError(f"unrecognized KBMAG alphabet symbol: {label!r}")
    index = "abcd".index(match.group(1)) + 1
    return -index if "^-1" in compact else index


def export_word_acceptor(
    extension_root: Path,
    output: Path,
    *,
    gap_bash: str,
    gap_binary_cygwin: str,
    stdout_log: Path,
    stderr_log: Path,
) -> WordAcceptor:
    script = extension_root / "src" / "export_word_acceptor.g"
    environment = os.environ.copy()
    environment["POSTVALIDATION_WORD_ACCEPTOR"] = to_cygwin(output)
    command = [
        gap_bash,
        "--login",
        "-c",
        f'"{gap_binary_cygwin}" -b -q "{to_cygwin(script)}"',
    ]
    completed = subprocess.run(
        command,
        cwd=extension_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or "AUTOMATIC=true" not in completed.stdout or not output.exists():
        raise RuntimeError(f"GAP/KBMAG word-acceptor export failed: {completed.stderr[-3000:]}")
    return parse_word_acceptor(output)


def parse_word_acceptor(path: Path) -> WordAcceptor:
    fields: dict[str, str] = {}
    rows: dict[int, list[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ROW="):
            state_text, values_text = line[4:].split("|", 1)
            rows[int(state_text)] = [int(value) for value in values_text.split(",")]
        elif "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    if fields.get("AUTOMATIC") != "true":
        raise RuntimeError("word acceptor is not certified automatic")
    state_count = int(fields["STATE_COUNT"])
    alphabet_size = int(fields["ALPHABET_SIZE"])
    table = np.zeros((state_count, alphabet_size), dtype=np.int32)
    for state in range(1, state_count + 1):
        values = rows[state]
        if len(values) != alphabet_size:
            raise RuntimeError("word-acceptor row width mismatch")
        table[state - 1] = np.asarray(values, dtype=np.int32) - 1
        table[state - 1][np.asarray(values) == 0] = -1
    alphabet = tuple(_letter(value) for value in fields["ALPHABET"].split("|"))
    expected = (1, -1, 2, -2, 3, -3, 4, -4)
    if set(alphabet) != set(expected):
        raise RuntimeError(f"unexpected KBMAG alphabet: {alphabet}")
    order = tuple(alphabet.index(letter) for letter in (1, 2, 3, 4, -1, -2, -3, -4))
    reordered = table[:, order]
    initial = [int(value) for value in fields["INITIAL"].split(",") if value]
    accepting = frozenset(int(value) - 1 for value in fields["ACCEPTING"].split(",") if value)
    if len(initial) != 1:
        raise RuntimeError("word acceptor must have one initial state")
    return WordAcceptor(
        transitions=reordered,
        initial_state=initial[0] - 1,
        accepting=accepting,
        alphabet_letters=(1, 2, 3, 4, -1, -2, -3, -4),
        gap_version=fields["GAP_VERSION"],
        source_sha256=sha256_file(path),
    )


def shortest_kernel_normal_word(
    permutations: np.ndarray,
    acceptor: WordAcceptor,
    *,
    maximum_product_states: int,
) -> tuple[int, tuple[int, ...], int]:
    order = int(permutations.shape[1])
    start_key = acceptor.initial_state * order
    parents: dict[int, tuple[int, int]] = {start_key: (-1, -1)}
    frontier: deque[tuple[int, int, int]] = deque([(acceptor.initial_state, 0, 0)])
    while frontier:
        automaton_state, group_state, depth = frontier.popleft()
        for move_index, letter in enumerate(acceptor.alphabet_letters):
            target_automaton = int(acceptor.transitions[automaton_state, move_index])
            if target_automaton < 0:
                continue
            target_group = int(permutations[move_index, group_state])
            key = target_automaton * order + target_group
            if key in parents:
                continue
            parents[key] = (automaton_state * order + group_state, move_index)
            if len(parents) > maximum_product_states:
                raise MemoryError("preregistered product-automaton state cap exceeded")
            if target_group == 0 and target_automaton in acceptor.accepting:
                word_indices: list[int] = []
                cursor = key
                while parents[cursor][0] >= 0:
                    parent, symbol = parents[cursor]
                    word_indices.append(symbol)
                    cursor = parent
                word = tuple(acceptor.alphabet_letters[index] for index in reversed(word_indices))
                return depth + 1, word, len(parents)
            frontier.append((target_automaton, target_group, depth + 1))
    raise RuntimeError("finite quotient product automaton has no accepted kernel word")
