from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from diffraction.arithmetic_complexity_recovery import run


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_spec_is_preregistered_and_scale_separated():
    config = yaml.safe_load((ROOT / "configs" / "phase_d08_recovery.yaml").read_text(encoding="utf-8"))
    spec = config["d08_preregistered_estimator"]
    assert spec["frozen_before_outcome"] is True
    assert spec["no_posthoc_changes"] is True
    assert spec["adjacent_point_regression_forbidden"] is True
    assert spec["fixed_ratio_c"] == 2
    assert spec["minimum_log_scale_separation"] >= 0.68
    assert spec["pair_block_starts"] == [64, 128, 256, 512, 1024]
    assert spec["envelope_block_starts"] == [64, 128, 256, 512, 1024, 2048]
    assert config["execution_order"][:2] == ["D-08", "D-09"]
    assert config["execution_order"][7:9] == ["D-15", "NC-01"]


def test_exact_power_four_synthetic_control_passes(tmp_path: Path):
    data = tmp_path / "inputs"
    data.mkdir()
    indices = np.arange(1, 5001, dtype=np.int64)
    q = (indices * indices + 3) ** 2
    pd.DataFrame({"j": indices, "coincidence_degree_maximal_order": q}).to_parquet(data / "growth.parquet", index=False)
    g10 = {"status": "PASS_EXACT", "formula": "(j^2+3)^2"}
    g11 = {
        "status": "PASS_CONVERGED", "compatible_with_4": True, "incompatible_with_1": True,
        "numerical_extrapolate_to_inverse_log_zero": 4.061663815456154,
    }
    (data / "g10.json").write_text(json.dumps(g10), encoding="utf-8")
    (data / "g11.json").write_text(json.dumps(g11), encoding="utf-8")
    base = {
        "arithmetic_sources": {
            "growth": "inputs/growth.parquet", "g10_certificate": "inputs/g10.json",
            "g11_certificate": "inputs/g11.json",
        }
    }
    recovery = yaml.safe_load((ROOT / "configs" / "phase_d08_recovery.yaml").read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    status, outputs = run(base, recovery, run_dir, "synthetic", tmp_path, {})
    certificate = json.loads(outputs["certificate"].read_text(encoding="utf-8"))
    assert status == "PASS_CERTIFIED"
    assert certificate["adjacent_point_regression_used"] is False
    assert certificate["posthoc_window_selection_used"] is False
    assert certificate["minimum_log_scale_separation"] >= 0.68
    assert abs(certificate["primary_extrapolated_exponent"] - 4.0) < 1.0e-10
    assert all(certificate["acceptance_checks"].values())


def test_frozen_pathological_estimator_is_not_called_by_recovery_workflow():
    workflow = (ROOT / "workflow" / "run_phase_d08_recovery.py").read_text(encoding="utf-8")
    assert "from diffraction.arithmetic_complexity_recovery import run as d08_recovery" in workflow
    assert "from diffraction.arithmetic_complexity import" not in workflow
    original = (ROOT / "src" / "diffraction" / "arithmetic_complexity.py").read_text(encoding="utf-8")
    assert "growth.tail(min(16, len(growth)))" in original
    assert "np.polyfit" in original
