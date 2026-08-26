from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def reduce_word(word: Iterable[int], generator_count: int = 4) -> tuple[int, ...]:
    reduced: list[int] = []
    for letter in word:
        value = int(letter)
        if value == 0 or abs(value) > generator_count:
            raise ValueError(f"word letter {value} is outside +/-1..+/-{generator_count}")
        if reduced and reduced[-1] == -value:
            reduced.pop()
        else:
            reduced.append(value)
    return tuple(reduced)


class LazyWordRepresentation:
    """Evaluate requested group words from generator matrices with a bounded cache.

    The constructor receives only the four generator matrices and their inverses.
    It never enumerates the finite group and never materializes matrices for all
    group elements. Non-generator matrices are produced only by ``evaluate``.
    """

    def __init__(
        self,
        generators: dict[int, np.ndarray],
        inverses: dict[int, np.ndarray],
        *,
        maximum_cached_words: int = 256,
    ) -> None:
        if maximum_cached_words < 1:
            raise ValueError("maximum_cached_words must be positive")
        if set(generators) != {1, 2, 3, 4} or set(inverses) != {1, 2, 3, 4}:
            raise ValueError("exactly four generators and four inverses are required")
        shapes = {np.asarray(matrix).shape for matrix in [*generators.values(), *inverses.values()]}
        if len(shapes) != 1 or next(iter(shapes))[0] != next(iter(shapes))[1]:
            raise ValueError("generator matrices must have one common square shape")
        self._generators = {index: np.asarray(matrix, dtype=np.complex128) for index, matrix in generators.items()}
        self._inverses = {index: np.asarray(matrix, dtype=np.complex128) for index, matrix in inverses.items()}
        self._maximum_cached_words = maximum_cached_words
        degree = next(iter(shapes))[0]
        self._cache: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()
        self._cache[()] = np.eye(degree, dtype=np.complex128)

    @classmethod
    def from_block(cls, path: Path, *, maximum_cached_words: int = 256) -> "LazyWordRepresentation":
        with h5py.File(path, "r") as handle:
            generators = {index: np.asarray(handle[f"generator_{index}"]) for index in range(1, 5)}
            inverses = {index: np.asarray(handle[f"generator_{index}_inverse"]) for index in range(1, 5)}
        return cls(generators, inverses, maximum_cached_words=maximum_cached_words)

    @property
    def materialized_group_element_count(self) -> int:
        return 8

    @property
    def cached_words(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._cache)

    def evaluate(self, word: Iterable[int]) -> np.ndarray:
        key = reduce_word(word)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = self._cache[()].copy()
        for letter in key:
            value = value @ (self._generators[letter] if letter > 0 else self._inverses[-letter])
        value.setflags(write=False)
        self._cache[key] = value
        while len(self._cache) > self._maximum_cached_words:
            oldest = next(iter(self._cache))
            if oldest == ():
                self._cache.move_to_end(oldest)
                oldest = next(iter(self._cache))
            self._cache.pop(oldest)
        return value
