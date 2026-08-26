from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from common import sha256_file
from word_automaton import WordAcceptor, shortest_kernel_normal_word, to_cygwin


def export_word_acceptor(
    extension_root: Path,
    output: Path,
    *,
    gap_bash: str,
    gap_binary_cygwin: str,
    stdout_log: Path,
    stderr_log: Path,
) -> WordAcceptor:
    script = extension_root / "src" / "export_word_acceptor_v3.g"
    environment = os.environ.copy()
    environment["POSTVALIDATION_WORD_ACCEPTOR"] = to_cygwin(output)
    completed = subprocess.run(
        [gap_bash, "--login", "-c", f'"{gap_binary_cygwin}" -b -q "{to_cygwin(script)}"'],
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
    logical_text = path.read_text(encoding="utf-8").replace("\\\n", "")
    for line in logical_text.splitlines():
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
    if alphabet_size != 8:
        raise RuntimeError("the marked surface group must have eight signed letters")
    table = np.zeros((state_count, alphabet_size), dtype=np.int32)
    for state in range(1, state_count + 1):
        values = rows.get(state)
        if values is None or len(values) != alphabet_size:
            raise RuntimeError(f"invalid word-acceptor row {state}")
        table[state - 1] = np.asarray(values, dtype=np.int32) - 1
        table[state - 1][np.asarray(values) == 0] = -1

    inverse = tuple(int(value) - 1 for value in fields["INVERSE_INDEX"].split(","))
    if len(inverse) != alphabet_size:
        raise RuntimeError("inverse-index width mismatch")
    if any(value < 0 or value >= alphabet_size for value in inverse):
        raise RuntimeError("inverse index outside alphabet")
    if any(inverse[inverse[index]] != index or inverse[index] == index for index in range(alphabet_size)):
        raise RuntimeError("inverse index is not a fixed-point-free involution")

    signed_by_raw: dict[int, int] = {}
    generator_index = 1
    for raw_index in range(alphabet_size):
        if raw_index in signed_by_raw:
            continue
        inverse_index = inverse[raw_index]
        signed_by_raw[raw_index] = generator_index
        signed_by_raw[inverse_index] = -generator_index
        generator_index += 1
    canonical = (1, 2, 3, 4, -1, -2, -3, -4)
    signed_alphabet = tuple(signed_by_raw[index] for index in range(alphabet_size))
    if set(signed_alphabet) != set(canonical):
        raise RuntimeError("derived signed alphabet is incomplete")
    order = tuple(signed_alphabet.index(letter) for letter in canonical)
    reordered = table[:, order]
    initial = [int(value) - 1 for value in fields["INITIAL"].split(",") if value]
    accepting = frozenset(int(value) - 1 for value in fields["ACCEPTING"].split(",") if value)
    if len(initial) != 1 or not accepting:
        raise RuntimeError("invalid initial/accepting states")
    for generator in range(4):
        positive_state = int(reordered[initial[0], generator])
        negative_state = int(reordered[initial[0], generator + 4])
        if positive_state < 0 or negative_state < 0:
            raise RuntimeError("one-letter word was unexpectedly rejected")
        if reordered[positive_state, generator + 4] != -1:
            raise RuntimeError("positive-negative cancellation was accepted")
        if reordered[negative_state, generator] != -1:
            raise RuntimeError("negative-positive cancellation was accepted")
    return WordAcceptor(
        transitions=reordered,
        initial_state=initial[0],
        accepting=accepting,
        alphabet_letters=canonical,
        gap_version=fields["GAP_VERSION"],
        source_sha256=sha256_file(path),
    )
