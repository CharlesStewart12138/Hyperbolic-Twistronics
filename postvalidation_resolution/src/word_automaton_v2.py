from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

from common import sha256_file
from word_automaton import WordAcceptor, shortest_kernel_normal_word, to_cygwin


def _letter(label: str) -> int:
    compact = label.replace(" ", "")
    named = re.fullmatch(r"([abcd])(?:\^-1)?", compact)
    if named is not None:
        index = "abcd".index(named.group(1)) + 1
        return -index if "^-1" in compact else index
    internal = re.fullmatch(r"_g([1-8])", compact)
    if internal is not None:
        index = int(internal.group(1))
        return index if index <= 4 else -(index - 4)
    raise ValueError(f"unrecognized KBMAG alphabet symbol: {label!r}")


def export_word_acceptor(
    extension_root: Path,
    output: Path,
    *,
    gap_bash: str,
    gap_binary_cygwin: str,
    stdout_log: Path,
    stderr_log: Path,
) -> WordAcceptor:
    script = extension_root / "src" / "export_word_acceptor_v2.g"
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
    table = np.zeros((state_count, alphabet_size), dtype=np.int32)
    for state in range(1, state_count + 1):
        if state not in rows:
            raise RuntimeError(f"word-acceptor state {state} is missing")
        values = rows[state]
        if len(values) != alphabet_size:
            raise RuntimeError("word-acceptor row width mismatch")
        table[state - 1] = np.asarray(values, dtype=np.int32) - 1
        table[state - 1][np.asarray(values) == 0] = -1
    alphabet = tuple(_letter(value) for value in fields["ALPHABET"].split("|"))
    canonical = (1, 2, 3, 4, -1, -2, -3, -4)
    if set(alphabet) != set(canonical):
        raise RuntimeError(f"unexpected KBMAG alphabet: {alphabet}")
    order = tuple(alphabet.index(letter) for letter in canonical)
    initial = [int(value) for value in fields["INITIAL"].split(",") if value]
    accepting = frozenset(int(value) - 1 for value in fields["ACCEPTING"].split(",") if value)
    if len(initial) != 1:
        raise RuntimeError("word acceptor must have one initial state")
    if not accepting or min(accepting) < 0 or max(accepting) >= state_count:
        raise RuntimeError("word acceptor has invalid accepting states")
    return WordAcceptor(
        transitions=table[:, order],
        initial_state=initial[0] - 1,
        accepting=accepting,
        alphabet_letters=canonical,
        gap_version=fields["GAP_VERSION"],
        source_sha256=sha256_file(path),
    )
