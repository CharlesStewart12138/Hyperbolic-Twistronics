from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def task_paths(run_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    raw = run_dir / "raw" / stem
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / f"{stem}.parquet"
    certificate = run_dir / "certificates" / f"{stem}.json"
    return raw, derived, certificate


def retained_group(blocks: pd.DataFrame, tower_id: str, level: int) -> pd.DataFrame:
    return blocks[
        (blocks.tower_id == tower_id)
        & (blocks.level == level)
        & blocks.retained_operator_tempered
    ].copy()


def weighted_cdf_distance(
    values_a: np.ndarray,
    weights_a: np.ndarray,
    values_b: np.ndarray,
    weights_b: np.ndarray,
) -> float:
    va = np.asarray(values_a, dtype=float)
    vb = np.asarray(values_b, dtype=float)
    wa = np.asarray(weights_a, dtype=float)
    wb = np.asarray(weights_b, dtype=float)
    wa = wa / np.sum(wa)
    wb = wb / np.sum(wb)
    oa = np.argsort(va)
    ob = np.argsort(vb)
    va, wa = va[oa], wa[oa]
    vb, wb = vb[ob], wb[ob]
    grid = np.unique(np.concatenate([va, vb]))
    cdf_a = np.concatenate([[0.0], np.cumsum(wa)])[np.searchsorted(va, grid, side="right")]
    cdf_b = np.concatenate([[0.0], np.cumsum(wb)])[np.searchsorted(vb, grid, side="right")]
    return float(np.max(np.abs(cdf_a - cdf_b)))


def gaussian_density(values, weights, grid, eta):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    z = (np.asarray(grid)[:, None] - values[None, :]) / float(eta)
    return np.sum(weights[None, :] * np.exp(-0.5 * z * z), axis=1) / (
        math.sqrt(2.0 * math.pi) * float(eta)
    )


def finish(certificate: Path, payload: dict[str, object], status: str, run_id: str, task: str) -> None:
    write_json(
        certificate,
        {"task_id": task, "run_id": run_id, "status": status, **payload},
    )
