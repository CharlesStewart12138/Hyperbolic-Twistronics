from __future__ import annotations

import ast
from pathlib import Path

import yaml


EXTENSION_ROOT = Path(__file__).resolve().parents[1]


def test_recovery_preregistration_is_fixed_and_parseable() -> None:
    config = yaml.safe_load(
        (EXTENSION_ROOT / "configs" / "r8_cover_recovery_preregistration.yaml").read_text(encoding="utf-8")
    )
    assert config["recovery_cover_rule"]["candidate_dyadic_depths"] == list(range(2, 11))
    assert config["recovery_cover_rule"]["no_posthoc_depth_subsets"] is True
    assert config["retained_sector"]["definition"] == "kernel_of_conditional_expectation_to_immediate_depth"


def test_r8_01_workflow_has_no_plotting_dependency() -> None:
    source = (EXTENSION_ROOT / "workflow" / "run_r8_01_recovery.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "matplotlib" not in imported
    assert "plotly" not in imported
