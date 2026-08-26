from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from r10_dos import *  # noqa: F401,F403


def slq_local_fixed_memmap(
    action,
    depth: int,
    *,
    temp_parent: Path | None = None,
    breakdown_tolerance: float = 1.0e-13,
):
    """Two-pass SLQ with an explicitly closed Windows memory map."""
    if depth < 2:
        raise ValueError("Lanczos depth must be at least two")
    directory = Path(tempfile.mkdtemp(prefix="deep_slq_v2_", dir=temp_parent))
    basis_path = directory / "basis.dat"
    basis = np.memmap(basis_path, dtype="float32", mode="w+", shape=(depth + 1, action.dimension))
    current = np.zeros(action.dimension, dtype=float)
    current[0] = 1.0
    previous = np.zeros_like(current)
    basis[0] = current.astype(np.float32)
    alpha: list[float] = []
    beta: list[float] = []
    maximum_orthogonality = 0.0
    try:
        for step in range(depth):
            image = action.apply(current)
            diagonal = float(np.dot(current, image))
            residual = image - diagonal * current
            if step > 0:
                residual -= beta[-1] * previous
            for _ in range(2):
                stored = basis[: step + 1]
                coefficients = np.asarray(stored @ residual.astype(np.float32), dtype=float)
                residual -= coefficients @ np.asarray(stored, dtype=float)
                del stored, coefficients
            alpha.append(diagonal)
            off_diagonal = float(np.linalg.norm(residual))
            if off_diagonal <= breakdown_tolerance or step == depth - 1:
                break
            beta.append(off_diagonal)
            previous, current = current, residual / off_diagonal
            basis[step + 1] = current.astype(np.float32)
            prior = basis[: step + 1]
            overlaps = np.asarray(prior @ current.astype(np.float32), dtype=float)
            maximum_orthogonality = max(maximum_orthogonality, float(np.max(np.abs(overlaps))))
            del prior, overlaps
        actual_depth = len(alpha)
        tridiagonal = np.diag(np.asarray(alpha, dtype=float))
        if beta:
            off = np.asarray(beta[: actual_depth - 1], dtype=float)
            tridiagonal += np.diag(off, 1) + np.diag(off, -1)
        nodes, vectors = np.linalg.eigh(tridiagonal)
        weights = np.abs(vectors[0]) ** 2
        result = {
            "alpha": np.asarray(alpha, dtype=float),
            "beta": np.asarray(beta[: actual_depth - 1], dtype=float),
            "nodes": nodes,
            "weights": weights,
            "actual_depth": actual_depth,
            "weight_sum_error": float(abs(np.sum(weights) - 1.0)),
            "maximum_orthogonality_residual": maximum_orthogonality,
        }
        basis.flush()
        mmap = getattr(basis, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del basis
        os.remove(basis_path)
        os.rmdir(directory)
        return result
    except Exception:
        basis.flush()
        mmap = getattr(basis, "_mmap", None)
        if mmap is not None:
            mmap.close()
        raise

