from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from audit.data_io import write_json, write_zarr


def _residual(reference: np.ndarray, candidate: np.ndarray) -> float:
    scale = 1.0 + float(np.max(np.abs(reference)))
    return float(np.max(np.abs(reference - candidate)) / scale)


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["s24_reverse_falsification"]
    source = zarr.open_group(str(run_dir / "raw" / "master_curve_complete_spectra.zarr"), mode="r")
    labels = list(source.attrs["labels"])
    reference_index = labels.index("reference:X=1.00")
    reference = np.asarray(source["normalized_complete_spectra"][reference_index], dtype=float)
    q = np.asarray(source["q"][:], dtype=float)
    q_normalized = q / float(np.max(np.abs(q)))
    scale = 1.0 + float(np.max(np.abs(reference)))

    controls = {
        "correct_control": reference.copy(),
        "wrong_metric": reference * (1.0 + 0.25 * q_normalized[:, None] ** 2),
        "wrong_target_transport": np.roll(reference, shift=1, axis=1),
        "second_shape_variable": reference + 0.20 * scale * q_normalized[:, None] ** 4,
        "incorrect_normalization": 1.30 * reference,
        "incomplete_active_shell": reference.copy(),
    }
    controls["incomplete_active_shell"][:, -1] = 0.0
    rows = []
    threshold = float(task["rejection_residual_minimum"])
    for name, candidate in controls.items():
        residual = _residual(reference, candidate)
        rows.append(
            {
                "control": name,
                "collapse_residual": residual,
                "rejected": name != "correct_control" and residual >= threshold,
                "expected": "ACCEPT" if name == "correct_control" else "REJECT",
            }
        )
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "reverse_falsification_spectra.zarr"
    write_zarr(
        raw,
        {
            "q": q,
            "reference": reference,
            **{name: values for name, values in controls.items() if name != "correct_control"},
        },
        {
            "task_id": "S-24",
            "run_id": run_id,
            "source": "S-18 complete normalized spectra",
            "negative_controls_are_deliberate": True,
        },
    )
    derived = run_dir / "derived" / "reverse_falsification_residuals.parquet"
    frame.to_parquet(derived, index=False)
    correct = float(frame.loc[frame.control == "correct_control", "collapse_residual"].iloc[0])
    required = set(map(str, task["controls"]))
    observed = set(frame.loc[frame.control != "correct_control", "control"])
    rejected = frame.loc[frame.control != "correct_control", "rejected"]
    passed = (
        required == observed
        and bool(rejected.all())
        and correct <= float(task["correct_control_residual_maximum"])
    )
    status = "FAIL_EXPECTED" if passed else "FAIL_IMPLEMENTATION"
    certificate = run_dir / "certificates" / "s24_reverse_falsification.json"
    write_json(
        certificate,
        {
            "task_id": "S-24",
            "run_id": run_id,
            "status": status,
            "correct_control_residual": correct,
            "rejection_threshold": threshold,
            "required_controls": sorted(required),
            "all_incorrect_controls_rejected": bool(rejected.all()),
            "residuals": {row["control"]: row["collapse_residual"] for row in rows},
            "interpretation": "Expected falsification: wrong metric, target transport, shape variable, normalization, and shell descriptions fail the frozen reverse-residual test.",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
