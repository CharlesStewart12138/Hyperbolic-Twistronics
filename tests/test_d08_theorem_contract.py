from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from audit.run_manifest import sha256_file
from diffraction.arithmetic_complexity_theorem_contract import run


ROOT = Path(__file__).resolve().parents[1]


def test_contract_is_branch_a_and_does_not_infer_regular_variation():
    recovery = yaml.safe_load((ROOT / "configs" / "phase_d08_theorem_contract_recovery.yaml").read_text(encoding="utf-8"))
    contract = recovery["theorem_contract"]
    spec = recovery["d08_preregistered_theorem_matched_test"]
    assert contract["branch"] == "A_LOG_ASYMPTOTIC_EXPONENT"
    assert contract["stronger_regular_variation_claim_registered"] is False
    assert "q_(c*j)/q_j" in contract["explicitly_not_inferred"]
    assert spec["theorem_target"] == "rho_j->0"
    assert spec["fixed_ratio_diagnostic"]["acceptance_role_for_theorem_A"] == "NONE"
    assert spec["dyadic_block_starts"] == [64, 128, 256, 512, 1024, 2048]
    assert spec["no_posthoc_changes"] is True


def test_theorem_matched_synthetic_control_passes_while_fixed_ratio_diagnostic_is_excluded(tmp_path: Path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    indices = np.arange(1, 4096, dtype=np.int64)
    q = (indices * indices + 3) ** 2
    growth = inputs / "growth.parquet"
    pd.DataFrame({"j": indices, "coincidence_degree_maximal_order": q}).to_parquet(growth, index=False)
    g10, g11, frozen = inputs / "g10.json", inputs / "g11.json", inputs / "frozen_d08.json"
    g10.write_text(json.dumps({"status": "PASS_EXACT", "all_local_products_equal_global_degree": True}), encoding="utf-8")
    g11.write_text(json.dumps({"status": "PASS_CONVERGED", "compatible_with_4": True}), encoding="utf-8")
    frozen.write_text(json.dumps({
        "status": "INCONCLUSIVE",
        "primary_extrapolated_exponent": 6.0170367374085725,
        "primary_tail_center": 6.012160883390238,
        "upper_envelope_extrapolated_exponent": 4.038050869764985,
        "lower_envelope_extrapolated_exponent": 3.9977920184187563,
        "acceptance_checks": {
            "upper_envelope_extrapolation": True,
            "lower_envelope_extrapolation": True,
            "envelope_extrapolation_gap": True,
        },
    }), encoding="utf-8")
    config = {"arithmetic_sources": {"growth": "inputs/growth.parquet"}}
    recovery = yaml.safe_load((ROOT / "configs" / "phase_d08_theorem_contract_recovery.yaml").read_text(encoding="utf-8"))
    recovery["theorem_contract"]["g10"] = {"path": "inputs/g10.json", "sha256": sha256_file(g10)}
    recovery["theorem_contract"]["g11"] = {"path": "inputs/g11.json", "sha256": sha256_file(g11)}
    recovery["frozen_predecessor"]["d08_certificate"] = "inputs/frozen_d08.json"
    recovery["frozen_predecessor"]["d08_certificate_sha256"] = sha256_file(frozen)
    status, outputs = run(config, recovery, tmp_path / "run", "synthetic", tmp_path, {})
    certificate = json.loads(outputs["certificate"].read_text(encoding="utf-8"))
    assert status == "PASS_CERTIFIED"
    assert certificate["data_support_log_asymptotic_exponent_four"] is True
    assert certificate["data_support_strong_regular_variation_under_doubling"] is False
    assert certificate["fixed_ratio_acceptance_role_for_theorem_A"] == "NONE"
    assert certificate["fixed_ratio_result_recalculated"] is False
    assert all(certificate["acceptance_checks"].values())
    assert abs(certificate["normalized_residual_extrapolation"]) < 0.12


def test_theorem_workflow_does_not_call_either_historical_estimator():
    workflow = (ROOT / "workflow" / "run_phase_d08_theorem_contract.py").read_text(encoding="utf-8")
    assert "arithmetic_complexity_theorem_contract" in workflow
    assert "arithmetic_complexity_recovery import run" not in workflow
    assert "arithmetic_complexity import run" not in workflow
