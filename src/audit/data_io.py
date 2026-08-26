from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import zarr


def write_zarr(path: Path, arrays: Mapping[str, object], attrs: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    for name, values in arrays.items():
        data = np.asarray(values)
        group.create_array(name, data=data, overwrite=False)
    group.attrs.update(dict(attrs))


def write_json(path: Path, data: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

