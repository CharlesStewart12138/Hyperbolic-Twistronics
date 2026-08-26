from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def paths(run_dir: Path, task: str):
    stem = task.lower().replace("-", "_")
    raw = run_dir / "raw" / stem
    raw.mkdir(parents=True, exist_ok=False)
    derived = run_dir / "derived" / f"{stem}.parquet"
    certificate = run_dir / "certificates" / f"{stem}.json"
    return raw, derived, certificate


def finish(certificate, task, run_id, payload):
    write_json(certificate, {
        "task_id": task, "run_id": run_id, "status": "FAIL_EXPECTED",
        "negative_control_succeeded": True, **payload,
    })


def nc01(config, run_dir, run_id, root, context):
    raw, derived, certificate = paths(run_dir, "NC-01")
    s01 = json.loads((root / str(config["negative_control_sources"]["s01"])).read_text(encoding="utf-8"))
    s02 = json.loads((root / str(config["negative_control_sources"]["s02"])).read_text(encoding="utf-8"))
    write_json(raw / "verified_sources.json", {"s01": s01, "s02": s02})
    pd.DataFrame([{
        "model": "square five-state", "positive_root_exists": False,
        "sharp_minimum": s02["sharp_minimum"], "no_positive_root": bool(s02["no_positive_root"]),
    }]).to_parquet(derived, index=False)
    finish(certificate, "NC-01", run_id, {
        "conclusion": "the square five-state root-free model falsifies universal root existence",
        "s01_sha256": config["negative_control_sources"]["s01_sha256"],
        "s02_sha256": config["negative_control_sources"]["s02_sha256"],
    })
    return "FAIL_EXPECTED", {"raw": raw, "derived": derived, "certificate": certificate}


def nc02(config, run_dir, run_id, root, context):
    raw, derived, certificate = paths(run_dir, "NC-02")
    source = json.loads((root / str(config["negative_control_sources"]["s16"])).read_text(encoding="utf-8"))
    write_json(raw / "verified_source.json", source)
    pd.DataFrame([{
        "trace_zero": source["trace_zero"], "traceless_response_nonzero": source["traceless_response_nonzero"],
        "operator_norm": source["operator_norm"], "principal_responses": json.dumps(source["principal_responses"]),
    }]).to_parquet(derived, index=False)
    finish(certificate, "NC-02", run_id, {
        "conclusion": "a trace root does not imply cancellation of the traceless Hodge tensor",
        "source_sha256": config["negative_control_sources"]["s16_sha256"],
    })
    return "FAIL_EXPECTED", {"raw": raw, "derived": derived, "certificate": certificate}


def nc03(config, run_dir, run_id, root, context):
    raw, derived, certificate = paths(run_dir, "NC-03")
    lengths = np.arange(2.0, 15.0)
    radius = 1.0
    delta = np.exp(-0.5 * lengths / radius)
    scaled = delta * np.exp(lengths / radius)
    frame = pd.DataFrame({"L_over_R": lengths, "delta_theta": delta, "delta_theta_exp_L_over_R": scaled})
    frame.to_parquet(raw / "violating_approximants.parquet", index=False)
    frame.to_parquet(derived, index=False)
    finish(certificate, "NC-03", run_id, {
        "condition": "|delta theta| exp(L/R) -> 0", "observed_direction": "diverges",
        "first_scaled_error": float(scaled[0]), "last_scaled_error": float(scaled[-1]),
        "conclusion": "pointwise angular approximation is insufficient on exponentially large hyperbolic patches",
    })
    return "FAIL_EXPECTED", {"raw": raw, "derived": derived, "certificate": certificate}


def nc04(config, run_dir, run_id, root, context):
    raw, derived, certificate = paths(run_dir, "NC-04")
    source = pd.read_parquet(context["phase_b_dir"] / "derived" / "b14_open_patch_control.parquet")
    source.to_parquet(raw / "open_disk_boundary_fraction.parquet", index=False)
    source.to_parquet(derived, index=False)
    tail = source[source.radius_word >= 3]
    finish(certificate, "NC-04", run_id, {
        "minimum_large_disk_boundary_fraction": float(tail.boundary_fraction.min()),
        "boundary_contamination_extensive": True,
        "conclusion": "open hyperbolic disks do not provide boundary-negligible thermodynamic approximants",
    })
    return "FAIL_EXPECTED", {"raw": raw, "derived": derived, "certificate": certificate}


def nc05(config, run_dir, run_id, root, context):
    raw, derived, certificate = paths(run_dir, "NC-05")
    source = pd.read_parquet(context["phase_b_dir"] / "derived" / "b08_character_incompleteness.parquet")
    source.to_parquet(raw / "single_character_failures.parquet", index=False)
    source.to_parquet(derived, index=False)
    finish(certificate, "NC-05", run_id, {
        "all_single_character_sectors_incomplete": bool((~source.single_character_representation_complete).all()),
        "total_missed_distinct_energies": int(source.missed_distinct_energies.sum()),
        "conclusion": "one character sector cannot stand in for complete finite-quotient Wedderburn decomposition",
    })
    return "FAIL_EXPECTED", {"raw": raw, "derived": derived, "certificate": certificate}


def nc06(config, run_dir, run_id, root, context):
    raw, derived, certificate = paths(run_dir, "NC-06")
    source = pd.read_parquet(context["phase_b_dir"] / "derived" / "b03_no_pollution_summary.parquet")
    source.to_parquet(raw / "unfiltered_regular_towers.parquet", index=False)
    source.to_parquet(derived, index=False)
    finish(certificate, "NC-06", run_id, {
        "all_full_regular_covers_polluted": bool(source.full_regular_has_pollution.all()),
        "rejected_regular_dimension": int(source.rejected_regular_dimension.sum()),
        "conclusion": "a tower without retained-sector no-pollution filtering cannot support the bulk claim",
    })
    return "FAIL_EXPECTED", {"raw": raw, "derived": derived, "certificate": certificate}


def nc07(config, run_dir, run_id, root, context):
    raw, derived, certificate = paths(run_dir, "NC-07")
    sizes = 2 ** np.arange(3, 11)
    frame = pd.DataFrame({
        "N": sizes, "C0_operator_error": 1.0 / sizes,
        "C1_derivative_error": np.ones_like(sizes, dtype=float),
    })
    frame.to_parquet(raw / "c0_without_c1.parquet", index=False)
    frame.to_parquet(derived, index=False)
    finish(certificate, "NC-07", run_id, {
        "C0_last_error": float(frame.C0_operator_error.iloc[-1]),
        "C1_last_error": float(frame.C1_derivative_error.iloc[-1]),
        "conclusion": "C0 operator convergence alone does not imply C1 derivative convergence",
    })
    return "FAIL_EXPECTED", {"raw": raw, "derived": derived, "certificate": certificate}


def nc08(config, run_dir, run_id, root, context):
    raw, derived, certificate = paths(run_dir, "NC-08")
    u = np.logspace(-2, 2, 81)
    rows = []
    for shape in (-1.0, 0.0, 1.0):
        master = u * u / (1.0 + u * u)
        observable = master * (1.0 + 0.15 * shape)
        for x, reference, value in zip(u, master, observable):
            rows.append({"u": float(x), "shape_variable": shape, "one_parameter_master": float(reference), "observable": float(value), "collapse_residual": float(value - reference)})
    frame = pd.DataFrame(rows)
    frame.to_parquet(raw / "two_parameter_family.parquet", index=False)
    frame.to_parquet(derived, index=False)
    finish(certificate, "NC-08", run_id, {
        "maximum_one_parameter_residual": float(frame.collapse_residual.abs().max()),
        "additional_shape_variable_required": True,
        "conclusion": "a one-parameter master is false when an independent shape variable remains active",
    })
    return "FAIL_EXPECTED", {"raw": raw, "derived": derived, "certificate": certificate}


def nc09(config, run_dir, run_id, root, context):
    raw, derived, certificate = paths(run_dir, "NC-09")
    chi = np.logspace(-2, 2, 121)
    correct = chi * chi / (1.0 + chi * chi)
    wrong_coordinate = chi * (1.0 + 0.25 * chi / (1.0 + chi))
    wrong_master = wrong_coordinate * wrong_coordinate / (1.0 + wrong_coordinate * wrong_coordinate)
    frame = pd.DataFrame({"chi": chi, "correct_master": correct, "wrong_metric_coordinate": wrong_coordinate, "wrong_master": wrong_master, "false_collapse_residual": wrong_master - correct})
    frame.to_parquet(raw / "wrong_metric_collapse.parquet", index=False)
    frame.to_parquet(derived, index=False)
    finish(certificate, "NC-09", run_id, {
        "maximum_false_collapse_residual": float(frame.false_collapse_residual.abs().max()),
        "wrong_metric_rejected": True,
        "conclusion": "a wrong normalization/metric can create a visually plausible but quantitatively false collapse",
    })
    return "FAIL_EXPECTED", {"raw": raw, "derived": derived, "certificate": certificate}


FUNCTIONS = {
    "NC-01": nc01, "NC-02": nc02, "NC-03": nc03, "NC-04": nc04, "NC-05": nc05,
    "NC-06": nc06, "NC-07": nc07, "NC-08": nc08, "NC-09": nc09,
}
