from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_final_figures as base
from plot_final_figures import COLORS, panel, save


def figure10(data: Path, output: Path) -> list[Path]:
    """Figure-10 renderer with literal math-text labels (no escape controls)."""
    errors = pd.read_parquet(data / "figure_10_error_budget.parquet")
    cdf = pd.read_parquet(data / "figure_10_cdf_methods.parquet")
    fixed = pd.read_parquet(data / "figure_10_fixed_broadening_density.parquet")
    density = pd.read_parquet(data / "figure_10_vanishing_coherence_density.parquet")
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.0), constrained_layout=True)
    ax = axes[0, 0]
    for tower, group in errors.groupby("tower_id"):
        group = group.sort_values("dyadic_depth")
        ax.semilogy(group.dyadic_depth, group.kappa_N, "o-", color=COLORS[tower], label=tower)
    panel(ax, "A", "Retained-sector CDF local-law error")
    ax.set(xlabel="dyadic depth", ylabel=r"$\kappa_N$")
    ax.grid(True, which="both")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for tower, group in errors.groupby("tower_id"):
        group = group.sort_values("dyadic_depth")
        ax.plot(group.dyadic_depth, group.kappa_over_eta, "o-", color=COLORS[tower], label=tower + r": $\kappa/\eta$")
        ax.plot(group.dyadic_depth, group.eta_to_alpha, "s--", color=COLORS[tower], alpha=0.7, label=tower + r": $\eta^\alpha$")
    panel(ax, "B", "Frozen vanishing-broadening terms")
    ax.set(xlabel="dyadic depth", ylabel="term value")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 0]
    for tower, group in fixed[fixed.dyadic_depth == 10].groupby("tower_id"):
        ax.plot(group.energy, group.density_mean, color=COLORS[tower], label=f"{tower} d10")
    panel(ax, "C", "Fixed-broadening KPM density")
    ax.set(xlabel="energy", ylabel="density")
    ax.grid(True)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    group = density[(density.tower_id == "dyadic_x_p7") & (density.dyadic_depth == 10)]
    mean = group.coherence_weighted_density.to_numpy(dtype=float)
    budget = group.complete_display_budget.to_numpy(dtype=float)
    energy = group.energy.to_numpy(dtype=float)
    ax.plot(energy, mean, color=COLORS["dyadic_x_p7"], label="coherence-weighted SLQ")
    ax.fill_between(energy, np.maximum(0.0, mean - budget), mean + budget, color=COLORS["dyadic_x_p7"], alpha=0.18, label="complete display budget")
    panel(ax, "D", "Coherence-weighted density; unresolved error band")
    ax.set(xlabel="energy", ylabel="weighted density")
    ax.grid(True)
    ax.legend(frameon=False)

    ax = axes[2, 0]
    group = cdf[(cdf.tower_id == "dyadic_x_p7") & (cdf.dyadic_depth == 10)]
    ax.plot(group.energy, group.KPM_CDF_mean, color="#264653", label="KPM")
    ax.plot(group.energy, group.SLQ_CDF_mean, "--", color="#E76F51", label="SLQ")
    panel(ax, "E", "Independent KPM/SLQ CDF cross-check")
    ax.set(xlabel="energy", ylabel="CDF")
    ax.grid(True)
    ax.legend(frameon=False)

    ax = axes[2, 1]
    ax.axis("off")
    panel(ax, "F", "Theorem-matched decision")
    ax.text(
        0.02,
        0.82,
        "Weak/CDF evidence: PASS_CERTIFIED\n\n"
        "Two-tower holdout: PASS_CERTIFIED\n\n"
        "Uniform Hölder certificate: absent\n"
        "eta_N -> 0 at available depths: not shown\n\n"
        "UNSMOOTHED / COHERENCE LIMIT\nINCONCLUSIVE",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFF3CD", "edgecolor": "#B08900"},
    )
    fig.suptitle("Revised Figure 10 — weak convergence is not an unsmoothed density limit", fontweight="bold")
    return save(fig, output, "figure_10_resolution")


def install() -> None:
    base.figure10 = figure10
