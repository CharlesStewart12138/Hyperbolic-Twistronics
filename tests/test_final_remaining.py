from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import yaml

from spectral.magic_active_shell import moire_length


ROOT = Path(__file__).resolve().parents[1]


def test_final_preregistration_and_execution_order() -> None:
    config = yaml.safe_load((ROOT / "configs" / "final_remaining.yaml").read_text(encoding="utf-8"))
    assert config["preregistered"] is True
    assert config["posthoc_changes_permitted"] is False
    assert config["execution_order"] == [
        "G-13", "G-14", "G-15", "S-17", "S-18", "S-19", "S-20", "S-21", "S-22", "S-23", "S-24"
    ]
    amendment = yaml.safe_load(
        (ROOT / "configs" / "final_remaining_preregistration_amendment.yaml").read_text(encoding="utf-8")
    )
    assert amendment["posthoc_outcome_inspected"] is False
    assert amendment["all_other_windows_ratios_tolerances_subsets_unchanged"] is True


def test_exact_hyperbolic_length_has_euclidean_limit() -> None:
    lattice_spacing = 1.7
    theta = 0.03
    euclidean = lattice_spacing / (2.0 * math.sin(theta / 2.0))
    curved = moire_length(1.0e7, theta, lattice_spacing)
    assert abs(curved / euclidean - 1.0) < 1.0e-10


def test_g13_sampling_identity_algebraically() -> None:
    zeta = 0.7
    c_nu = 1.3
    for j in (1, 4, 17, 100):
        lambda_j = math.asinh(math.sinh(zeta) * math.sqrt((j * j + 3.0) / 3.0))
        omega_j = c_nu * c_nu / (lambda_j * lambda_j)
        sampled = math.sinh(zeta) / math.sinh(c_nu / math.sqrt(omega_j))
        exact = math.sqrt(3.0 / (j * j + 3.0))
        assert abs(sampled - exact) < 1.0e-13


def test_joint_limit_positive_and_negative_controls() -> None:
    lengths = np.asarray([2.5, 3.5, 5.0])
    passing = np.exp(-2.0 * lengths) * np.exp(lengths)
    failing = np.exp(-0.5 * lengths) * np.exp(lengths)
    assert np.all(np.diff(passing) < 0.0)
    assert np.all(np.diff(failing) > 0.0)


def test_remaining_modules_import_and_files_exist() -> None:
    modules = [
        "analysis.magic_subsequence_sampling",
        "analysis.magic_complexity",
        "geometry.incommensurate_joint_limit",
        "spectral.operational_magic_metrics",
        "spectral.master_curve_collapse",
        "spectral.geometry_spectrum_factorization",
        "spectral.magic_landscape",
        "spectral.bifurcation_certificates",
        "spectral.curvature_born_branch",
        "spectral.symmetry_vs_flatness",
        "spectral.reverse_falsification",
        "audit.final_global_audit",
    ]
    for name in modules:
        assert callable(importlib.import_module(name).run)
    assert (ROOT / "src" / "spectral" / "bifurcation_certificates.sage").exists()


def test_final_plotter_is_read_only_scientific_renderer() -> None:
    text = (ROOT / "src" / "plots" / "render_final_publication_figures.py").read_text(encoding="utf-8")
    assert "write_zarr" not in text
    assert "to_parquet" not in text
    assert "scientific_calculation_in_renderer" in text
