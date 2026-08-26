from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "dyadic_ramified": "#16697A",
    "dyadic_x_p7": "#DB6400",
    "curvature_radius": "#A23B72",
    "twist_angle": "#2A9D8F",
    "tunneling_decay": "#457B9D",
    "shell_cutoff": "#E9C46A",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def panel(ax, label: str, title: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.set_title(title, loc="left", fontweight="semibold")


def save(fig, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    svg = directory / f"{stem}.svg"
    png = directory / f"{stem}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [svg, png]


def figure8(data: Path, output: Path) -> list[Path]:
    cover = pd.read_parquet(data / "figure_8_cover_levels.parquet")
    levels = pd.read_parquet(data / "figure_8_level_metrics.parquet")
    cross = pd.read_parquet(data / "figure_8_cross_tower.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    for tower, group in cover.groupby("tower_id"):
        group = group.sort_values("dyadic_depth")
        ax.plot(group.dyadic_depth, group.injectivity_radius_integer, "o-", lw=1.7, color=COLORS[tower], label=tower)
    panel(ax, "A", "Certified injectivity radius by tower")
    ax.set(xlabel="dyadic depth", ylabel=r"integer $r_{inj}$")
    ax.grid(True)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for tower, group in levels.groupby("tower_id"):
        group = group.sort_values("dyadic_depth")
        ax.plot(group.dyadic_depth, group.bandwidth, "o-", color=COLORS[tower], label=f"{tower}: bandwidth")
        ax.plot(group.dyadic_depth, group.external_gap, "s--", color=COLORS[tower], alpha=0.7, label=f"{tower}: gap")
    panel(ax, "B", "Within-tower bandwidth and external gap")
    ax.set(xlabel="dyadic depth", ylabel="normalized energy")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 0]
    for tower, group in levels.groupby("tower_id"):
        group = group.sort_values("dyadic_depth")
        ax.semilogy(group.dyadic_depth, group.eta_N, "o-", color=COLORS[tower], label=f"{tower}: $\eta_N$")
        ax.semilogy(group.dyadic_depth, group.actual_edge_error, "x--", color=COLORS[tower], label=f"{tower}: actual edge")
    panel(ax, "C", "Certificate versus observed edge residual")
    ax.set(xlabel="dyadic depth", ylabel="error (log scale)")
    ax.grid(True, which="both")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 1]
    ax.plot(cross.dyadic_depth, cross.residual_fixed_broadening_DOS_L1, "o-", color="#7B2CBF", label="matched DOS $L^1$")
    ax.plot(cross.dyadic_depth, cross.combined_eta, "s--", color="#4D908E", label="combined $\eta$")
    failed = ~cross.all_residuals_within_combined_eta.astype(bool)
    ax.scatter(cross.loc[failed, "dyadic_depth"], cross.loc[failed, "residual_fixed_broadening_DOS_L1"], marker="x", s=70, color="#C1121F", label="budget failure")
    panel(ax, "D", "Matched cross-tower consistency")
    ax.set(xlabel="matched dyadic depth", ylabel="residual / budget")
    ax.grid(True)
    ax.legend(frameon=False)
    fig.suptitle("Revised Figure 8 — within-tower convergence is distinct from cross-tower consistency", fontweight="bold")
    return save(fig, output, "figure_8_resolution")


def figure9(data: Path, output: Path) -> list[Path]:
    frame = pd.read_parquet(data / "figure_9_balanced_errors.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    components = ["epsilon_core", "epsilon_physical_tail", "epsilon_master_tail", "epsilon_cover", "epsilon_solver", "epsilon_transport"]
    ax = axes[0, 0]
    styles = ["o-", "s-", "^-", "d-", "x-", "+-"]
    mixed = frame[frame.tower_id == "dyadic_x_p7"].sort_values("dyadic_depth")
    for component, style in zip(components, styles):
        ax.semilogy(mixed.dyadic_depth, mixed[component], style, ms=4, label=component.replace("epsilon_", ""))
    panel(ax, "A", "Six errors on one declared diagonal (mixed tower)")
    ax.set(xlabel="dyadic depth", ylabel="component error")
    ax.grid(True, which="both")
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    for tower, group in frame.groupby("tower_id"):
        group = group.sort_values("dyadic_depth")
        ax.plot(group.dyadic_depth, group.epsilon_total, "o-", color=COLORS[tower], label=tower)
    panel(ax, "B", r"Total error; $L_j=1$ at every level")
    ax.set(xlabel="dyadic depth", ylabel=r"$\epsilon_{total}$")
    ax.grid(True)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for tower, group in frame.groupby("tower_id"):
        group = group.sort_values("dyadic_depth")
        ax.semilogy(group.dyadic_depth, group.operator_surrogate_error, "o-", color=COLORS[tower], label=f"{tower}: operator")
        ax.semilogy(group.dyadic_depth, group.fixed_multiple_bound, "--", color=COLORS[tower], alpha=0.65, label=f"{tower}: bound")
    panel(ax, "C", "Operator surrogate and certified multiple")
    ax.set(xlabel="dyadic depth", ylabel="error / bound")
    ax.grid(True, which="both")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 1]
    for tower, group in frame.groupby("tower_id"):
        group = group.sort_values("dyadic_depth")
        ax.plot(group.dyadic_depth, group.bandwidth_error, "o-", color=COLORS[tower], label=f"{tower}: bandwidth")
        ax.plot(group.dyadic_depth, group.gap_error, "s--", color=COLORS[tower], alpha=0.7, label=f"{tower}: gap")
        ax.plot(group.dyadic_depth, group.riesz_projector_error, ":", color=COLORS[tower], label=f"{tower}: projector")
    panel(ax, "D", "Spectral inheritance on the same sequence")
    ax.set(xlabel="dyadic depth", ylabel="absolute error")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    fig.suptitle("Revised Figure 9 — certified finite-level inheritance, but no balanced asymptotic diagonal", fontweight="bold")
    return save(fig, output, "figure_9_resolution")


def figure10(data: Path, output: Path) -> list[Path]:
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
        ax.plot(group.dyadic_depth, group.kappa_over_eta, "o-", color=COLORS[tower], label=f"{tower}: $\kappa/\eta$")
        ax.plot(group.dyadic_depth, group.eta_to_alpha, "s--", color=COLORS[tower], alpha=0.7, label=f"{tower}: $\eta^\alpha$")
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
        "η_N→0 at available depths: not shown\n\n"
        "UNSMOOTHED / COHERENCE LIMIT\nINCONCLUSIVE",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFF3CD", "edgecolor": "#B08900"},
    )
    fig.suptitle("Revised Figure 10 — weak convergence is not an unsmoothed density limit", fontweight="bold")
    return save(fig, output, "figure_10_resolution")


def figure16(data: Path, output: Path) -> list[Path]:
    spectra = pd.read_parquet(data / "figure_16_selected_spectra.parquet")
    maxima = pd.read_parquet(data / "figure_16_channel_maxima.parquet")
    fields = pd.read_parquet(data / "figure_16_field_amplitudes.parquet")
    corrected = pd.read_parquet(data / "figure_16_corrected_holdout.parquet")
    fig, axes = plt.subplots(3, 2, figsize=(10.7, 10.2), constrained_layout=True)
    ax = axes[0, 0]
    label_colors = {
        "reference:X=1.00": "#111111",
        "radius_holdout_6:X=1.00": COLORS["curvature_radius"],
        "theta_holdout_0.1:X=1.00": COLORS["twist_angle"],
        "lambda_holdout_0.24:X=1.00": COLORS["tunneling_decay"],
        "cutoff_4:X=1.00": COLORS["shell_cutoff"],
    }
    for label, group in spectra.groupby("label"):
        for band, band_group in group.groupby("band"):
            ax.plot(band_group.q, band_group.energy, color=label_colors[label], lw=1.0, alpha=0.8, label=label.replace(":X=1.00", "") if band == 0 else None)
    panel(ax, "A", "Matched-X complete spectra (X=1)")
    ax.set(xlabel="normalized q", ylabel="normalized energy")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    x = np.arange(len(maxima))
    width = 0.24
    for offset, column, label in ((-width, "epsilon_C0", "C0"), (0.0, "epsilon_C1", "C1"), (width, "epsilon_C2", "C2")):
        ax.bar(x + offset, maxima[column], width=width, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, [value.replace("_", "\n") for value in maxima.channel], rotation=0)
    panel(ax, "B", "Worst one-parameter operator residual by channel")
    ax.set(ylabel="sup operator norm")
    ax.grid(True, axis="y", which="both")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for channel, group in fields.groupby("channel"):
        ax.loglog(group.asymptotic_scale, group.epsilon_C0, "o", color=COLORS[channel], label=channel)
    panel(ax, "C", "Residual versus preregistered asymptotic field")
    ax.set(xlabel="absolute theory-field amplitude", ylabel="C0 residual")
    ax.grid(True, which="both")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for channel, group in fields.groupby("channel"):
        ordered = group.sort_values("asymptotic_scale")
        ax.semilogy(np.arange(len(ordered)), ordered.asymptotic_scale, "o-", color=COLORS[channel], label=channel)
    panel(ax, "D", "Theory-derived correction-field amplitudes")
    ax.set(xlabel="cases ordered within channel", ylabel="absolute field amplitude")
    ax.grid(True, which="both")
    ax.legend(frameon=False)

    ax = axes[2, 0]
    for channel, group in corrected.groupby("channel"):
        ax.loglog(group.baseline_epsilon_C0, group.corrected_epsilon_C0, "o", color=COLORS[channel], label=channel)
    limits = [max(1e-3, min(corrected.baseline_epsilon_C0.min(), corrected.corrected_epsilon_C0.min()) * 0.7), max(corrected.baseline_epsilon_C0.max(), corrected.corrected_epsilon_C0.max()) * 1.3]
    ax.plot(limits, limits, "k--", lw=1, label="no improvement")
    ax.set(xlim=limits, ylim=limits)
    panel(ax, "E", "Independent corrected-master holdouts")
    ax.set(xlabel="baseline C0", ylabel="corrected C0")
    ax.grid(True, which="both")
    ax.legend(frameon=False)

    ax = axes[2, 1]
    ax.axis("off")
    panel(ax, "F", "Universality classification")
    ax.text(
        0.02,
        0.82,
        "H0 unrestricted one-parameter: FAILS\n"
        "  radius max C0 = 10.596\n"
        "  angle max C0 = 4.886\n\n"
        "H2 correction-field master: INCONCLUSIVE\n"
        "  angle holdout max C0 = 0.0966\n"
        "  radius holdout max C0 = 3.057\n\n"
        "FINAL: PASS_RESTRICTED_CLASS\n"
        "No FAIL_THEORY",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E8F5E9", "edgecolor": "#2D6A4F"},
    )
    fig.suptitle("Revised Figure 16 — fixed-class collapse survives; radius remains a relevant field", fontweight="bold")
    return save(fig, output, "figure_16_resolution")


def render_all(run_dir: Path) -> dict[int, list[Path]]:
    setup_style()
    data = run_dir / "figure_data"
    output = run_dir / "figures"
    return {
        8: figure8(data, output),
        9: figure9(data, output),
        10: figure10(data, output),
        16: figure16(data, output),
    }
