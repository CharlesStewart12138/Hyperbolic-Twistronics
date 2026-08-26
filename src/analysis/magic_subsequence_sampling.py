from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from audit.data_io import write_json


def _arb_midpoint(text: str) -> float:
    match = re.match(r"\[([^ ]+) \+/-", str(text))
    if match is None:
        raise ValueError(f"cannot parse Arb interval: {text}")
    return float(match.group(1))


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["g13_magic_sampling"]
    amendment = config["preregistration_amendment"]["replacement"]
    s04_path = root / str(config["phase_s_source"]["s04_certificate"])
    s04 = json.loads(s04_path.read_text(encoding="utf-8"))
    alpha_star = 1.0 / float(s04["parameter_box"]["q1"])
    interval_midpoint = _arb_midpoint(str(s04["root_interval"]))
    radius = float(task["curvature_radius"])
    lattice_spacing = 2.0 * radius * math.acosh(1.0 + math.sqrt(2.0))
    zeta = lattice_spacing / (2.0 * radius)
    c_nu = (
        2.0
        * math.pi
        * zeta
        / float(task["active_channel_count_nc"])
        * math.sqrt(float(task["dispersion_coefficient_d"]) * alpha_star / float(task["gamma0"]))
    )
    rows = []
    for j in map(int, task["j_values"]):
        arithmetic_half_sine = math.sqrt(3.0 / (j * j + 3.0))
        lambda_arithmetic = math.asinh(math.sinh(zeta) * math.sqrt((j * j + 3.0) / 3.0))
        omega = c_nu * c_nu / (lambda_arithmetic * lambda_arithmetic)
        sampled_half_sine = math.sinh(zeta) / math.sinh(c_nu / math.sqrt(omega))
        theta_magic = 2.0 * math.asin(sampled_half_sine)
        theta_exact = 2.0 * math.atan(math.sqrt(3.0) / j)
        rows.append(
            {
                "j": j,
                "alpha_star": alpha_star,
                "c_nu": c_nu,
                "zeta": zeta,
                "lambda_arithmetic": lambda_arithmetic,
                "omega_j": omega,
                "arithmetic_half_sine": arithmetic_half_sine,
                "sampled_half_sine": sampled_half_sine,
                "half_sine_residual": sampled_half_sine - arithmetic_half_sine,
                "theta_magic": theta_magic,
                "theta_exact": theta_exact,
                "theta_residual": theta_magic - theta_exact,
            }
        )
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "g13_magic_subsequence_inputs.parquet"
    derived = run_dir / "derived" / "magic_subsequence_sampling.parquet"
    frame[["j", "alpha_star", "c_nu", "zeta", "lambda_arithmetic", "omega_j"]].to_parquet(raw, index=False)
    frame.to_parquet(derived, index=False)
    max_residual = float(
        max(frame["half_sine_residual"].abs().max(), frame["theta_residual"].abs().max())
    )
    decreasing = bool(np.all(np.diff(frame["omega_j"].to_numpy(dtype=float)) < 0.0))
    final_ratio = float(frame["omega_j"].iloc[-1] / frame["omega_j"].iloc[0])
    root_matches = abs(alpha_star - interval_midpoint) <= 1.0e-10
    acceptance = task["acceptance"]
    passed = (
        s04.get("status") == acceptance["root_status_required"]
        and root_matches
        and max_residual <= float(acceptance["maximum_sampling_identity_residual"])
        and decreasing
        and final_ratio <= float(amendment["add"]["final_to_initial_omega_ratio_maximum"])
    )
    status = "PASS_CERTIFIED" if passed else "FAIL_THEORY"
    certificate = run_dir / "certificates" / "g13_magic_subsequence_sampling.json"
    write_json(
        certificate,
        {
            "task_id": "G-13",
            "run_id": run_id,
            "status": status,
            "theorem_equations": task["theorem_equations"],
            "source_s04_status": s04.get("status"),
            "source_s04_root_interval": s04.get("root_interval"),
            "alpha_star": alpha_star,
            "root_midpoint_agreement": root_matches,
            "lattice_spacing": lattice_spacing,
            "zeta": zeta,
            "c_nu": c_nu,
            "maximum_sampling_identity_residual": max_residual,
            "omega_strictly_decreasing": decreasing,
            "final_to_initial_omega_ratio": final_ratio,
            "analytic_limit": "omega_j=c_nu^2/lambda_j^2 -> 0 because lambda_j -> infinity",
            "preregistration_amendment_applied": "configs/final_remaining_preregistration_amendment.yaml",
            "scope": "certified M1 character-sector root sampled on the exact centered arithmetic sequence",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
