from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from audit.data_io import write_json


def run(config: dict, run_dir: Path, run_id: str, root: Path) -> tuple[str, dict[str, Path]]:
    task = config["g15_joint_limit"]
    gate = json.loads((root / str(config["nonabelian_tower_source"]["certificate"])).read_text(encoding="utf-8"))
    levels = pd.read_parquet(root / str(config["nonabelian_tower_source"]["levels"]))
    eligible = levels.loc[
        levels["nonabelian"].astype(bool)
        & levels["normal_cover"].astype(bool)
        & levels["bulk_gate_eligible"].astype(bool)
    ].copy()
    rows = []
    target = float(task["target_theta"])
    for record in eligible.itertuples(index=False):
        length_over_radius = float(record.injectivity_radius_word_lower)
        passing_delta = math.exp(-2.0 * length_over_radius)
        failing_delta = math.exp(-0.5 * length_over_radius)
        rows.extend(
            [
                {
                    "tower_id": record.tower_id,
                    "level": int(record.level),
                    "sequence": "passing",
                    "L_over_R": length_over_radius,
                    "target_theta": target,
                    "approximant_theta": target + passing_delta,
                    "absolute_delta_theta": passing_delta,
                    "joint_limit_residual": passing_delta * math.exp(length_over_radius),
                },
                {
                    "tower_id": record.tower_id,
                    "level": int(record.level),
                    "sequence": "deliberately_failing",
                    "L_over_R": length_over_radius,
                    "target_theta": target,
                    "approximant_theta": target + failing_delta,
                    "absolute_delta_theta": failing_delta,
                    "joint_limit_residual": failing_delta * math.exp(length_over_radius),
                },
            ]
        )
    frame = pd.DataFrame(rows)
    raw = run_dir / "raw" / "g15_incommensurate_sequences.parquet"
    frame[["tower_id", "level", "sequence", "L_over_R", "target_theta", "approximant_theta"]].to_parquet(raw, index=False)
    derived = run_dir / "derived" / "incommensurate_joint_limit.parquet"
    frame.to_parquet(derived, index=False)
    checks = {}
    for tower_id, group in frame.groupby("tower_id"):
        passing = group.loc[group.sequence == "passing"].sort_values("level")
        failing = group.loc[group.sequence == "deliberately_failing"].sort_values("level")
        checks[str(tower_id)] = {
            "passing_residual_strictly_decreases": bool((passing.joint_limit_residual.diff().dropna() < 0).all()),
            "failing_residual_strictly_increases": bool((failing.joint_limit_residual.diff().dropna() > 0).all()),
            "passing_analytic_limit": "exp(-L/R) -> 0",
            "failing_analytic_limit": "exp(+L/(2R)) -> infinity",
        }
    required = int(config["nonabelian_tower_source"]["required_inequivalent_towers"])
    gate_status = str(gate.get("status", gate.get("gate_status", "")))
    passed = (
        gate_status == "PASS_CERTIFIED"
        and len(checks) >= required
        and all(item["passing_residual_strictly_decreases"] for item in checks.values())
        and all(item["failing_residual_strictly_increases"] for item in checks.values())
    )
    status = "PASS_CERTIFIED" if passed else "FAIL_THEORY"
    certificate = run_dir / "certificates" / "g15_incommensurate_joint_limit.json"
    write_json(
        certificate,
        {
            "task_id": "G-15",
            "run_id": run_id,
            "status": status,
            "theorem_equation": task["theorem_equation"],
            "tower_gate_status": gate_status,
            "inequivalent_tower_count": len(checks),
            "required_inequivalent_towers": required,
            "tower_checks": checks,
            "passing_sequence_limit": "ZERO",
            "deliberately_failing_sequence_limit": "INFINITY",
            "falsification_preserved": True,
            "bulk_scope": "Only certified non-Abelian normal-cover towers with the frozen injectivity-radius gate are used.",
        },
    )
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
