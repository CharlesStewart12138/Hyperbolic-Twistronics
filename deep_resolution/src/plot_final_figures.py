from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#2C7FB8", "#D95F0E", "#31A354", "#756BB1", "#636363"]


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(figure: plt.Figure, directory: Path, stem: str) -> None:
    figure.savefig(directory / f"{stem}.png", bbox_inches="tight")
    figure.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def figure8(data: Path, figures: Path) -> None:
    levels = pd.read_parquet(data / "figure_8_deep_cover_levels.parquet")
    matched = pd.read_parquet(data / "figure_8_matched_levels.parquet")
    schedule = pd.read_parquet(data / "figure_10_vanishing_schedule.parquet")
    holdout = schedule[schedule.split == "holdout"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for color, (tower, group) in zip(COLORS, levels.groupby("tower_id")):
        axes[0, 0].plot(group.level, group.injectivity_radius_lower, "o-", color=color, label=tower.replace("congruence_", ""))
        axes[0, 1].plot(group.injectivity_radius_lower, group.quotient_order_digits, "o-", color=color, label=tower.replace("congruence_", ""))
    axes[0, 0].set(xlabel="congruence level n", ylabel="certified $r_{inj}$ lower", title="A  Growing non-Abelian towers")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(xlabel="certified $r_{inj}$ lower", ylabel="decimal digits of quotient order", title="B  Exact quotient growth")
    labels = [value.replace("congruence_", "") for value in holdout.tower_id]
    x = np.arange(len(labels))
    axes[1, 0].bar(x - 0.18, holdout.kappa_N, 0.36, label="$\kappa_N$", color=COLORS[0])
    axes[1, 0].bar(x + 0.18, holdout.eta_N, 0.36, label="$\eta_N=\sqrt{\kappa_N}$", color=COLORS[1])
    axes[1, 0].set_xticks(x, labels, rotation=18)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="C  Weak spectral-measure holdout", ylabel="error / broadening")
    axes[1, 0].legend(frameon=False)
    mismatch = matched.groupby("threshold", as_index=False).relative_radius_mismatch.max()
    axes[1, 1].plot(mismatch.threshold, mismatch.relative_radius_mismatch, "o-", color=COLORS[3])
    axes[1, 1].axhline(0.0, color="black", lw=0.7)
    axes[1, 1].set(xlabel="matched $r_{inj}$ target", ylabel="max relative mismatch", title="D  Matched cross-tower design")
    axes[1, 1].text(0.03, 0.96, "weak retained-sector closure: PASS\nstrong edge/gap no-pollution: INCONCLUSIVE", transform=axes[1, 1].transAxes, va="top", fontsize=7.5)
    fig.suptitle("Figure 8 — Deep finite-cover closure", fontsize=11)
    fig.tight_layout()
    save(fig, figures, "figure08_deep_finite_cover_closure")


def figure9(data: Path, figures: Path) -> None:
    frame = pd.read_parquet(data / "figure_9_balanced_diagonal.parquet")
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.5))
    for color, (tower, group) in zip(COLORS, frame.groupby("tower_id")):
        axes[0, 0].plot(group.L_j, group.injectivity_radius_lower, "o-", color=color, label=tower.replace("congruence_", ""))
        axes[0, 1].plot(group.L_j, group.epsilon_total_C0, "o-", color=color)
        axes[0, 2].plot(group.L_j, group.epsilon_total_C1, "o-", color=color)
        axes[1, 0].plot(group.L_j, group.epsilon_total_C2, "o-", color=color)
        axes[1, 1].plot(group.L_j, group.L_over_rinj, "o-", color=color)
    axes[0, 0].set(xlabel="$L_j$", ylabel="$r_{inj,j}$", title="A  Genuine balanced diagonal")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(xlabel="$L_j$", ylabel="$\epsilon_{C0}$", title="B  C0 bound")
    axes[0, 2].set(xlabel="$L_j$", ylabel="$\epsilon_{C1}$", title="C  C1 bound")
    axes[1, 0].set(xlabel="$L_j$", ylabel="$\epsilon_{C2}$", title="D  C2 bound")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set(xlabel="$L_j$", ylabel="$L_j/r_{inj,j}$", title="E  Scale separation")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.95, "CERTIFIED\n$L_j=1,2,3,4,5$\n$L_j\to\infty$\n$L_j/r_{inj,j}\to0$\n\nINCONCLUSIVE\nphysical/Hodge packing bounds\ndo not close with word shell", va="top", fontsize=9)
    fig.suptitle("Figure 9 — True balanced full-shell sequence", fontsize=11)
    fig.tight_layout()
    save(fig, figures, "figure09_balanced_full_shell")


def figure10(data: Path, figures: Path) -> None:
    methods = pd.read_parquet(data / "figure_10_KPM_SLQ_CDF.parquet")
    schedule = pd.read_parquet(data / "figure_10_vanishing_schedule.parquet")
    density = pd.read_parquet(data / "figure_10_holdout_coherence_density.parquet")
    holdout = schedule[schedule.split == "holdout"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    for color, row in zip(COLORS, holdout.itertuples()):
        subset = methods[(methods.tower_id == row.tower_id) & (methods.level == row.level)]
        label = row.tower_id.replace("congruence_", "")
        axes[0, 0].plot(subset.energy, subset.KPM_CDF, color=color, lw=1.2, label=f"{label} KPM")
        axes[0, 0].plot(subset.energy, subset.SLQ_CDF, color=color, lw=0.8, ls="--")
    axes[0, 0].set(xlabel="energy", ylabel="CDF", title="A  Independent KPM / SLQ")
    axes[0, 0].legend(frameon=False, ncol=1)
    x = np.arange(len(holdout))
    axes[0, 1].bar(x - 0.2, holdout.kappa_N, 0.4, color=COLORS[0], label="$\kappa_N$")
    axes[0, 1].bar(x + 0.2, holdout.eta_N, 0.4, color=COLORS[1], label="$\eta_N$")
    axes[0, 1].set_xticks(x, [value.replace("congruence_", "") for value in holdout.tower_id], rotation=18)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="B  Frozen vanishing schedule", ylabel="scale")
    axes[0, 1].legend(frameon=False)
    for color, (tower, group) in zip(COLORS, density.groupby("tower_id")):
        axes[1, 0].plot(group.energy, group.coherence_weighted_density, color=color, label=tower.replace("congruence_", ""))
    axes[1, 0].set(xlabel="energy", ylabel="$\rho_{coh,\eta_N}$", title="C  Coherence-weighted density")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(x, holdout.KPM_SLQ_disagreement, color=COLORS[3])
    axes[1, 1].axhline(0.08, color="black", ls="--", lw=0.8, label="frozen gate")
    axes[1, 1].set_xticks(x, [value.replace("congruence_", "") for value in holdout.tower_id], rotation=18)
    axes[1, 1].set(title="D  Method disagreement", ylabel="$\|F_{KPM}-F_{SLQ}\|_\infty$")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].text(0.02, 0.96, "weak CDF + schedule: PASS\nuniform regularity / local DOS: INCONCLUSIVE", transform=axes[1, 1].transAxes, va="top", fontsize=7.4)
    fig.suptitle("Figure 10 — Vanishing-broadening DOS hierarchy", fontsize=11)
    fig.tight_layout()
    save(fig, figures, "figure10_vanishing_broadening_DOS")


def figure16(data: Path, figures: Path) -> None:
    summary = pd.read_parquet(data / "figure_16_channel_summary.parquet")
    shape = pd.read_parquet(data / "figure_16_old_radius_shape_field.parquet")
    holdout = pd.read_parquet(data / "figure_16_hypothesis_holdout.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    channels = list(summary.channel.unique())
    hypotheses = ["H0", "H2", "H3", "H4"]
    x = np.arange(len(channels))
    for index, hypothesis in enumerate(hypotheses):
        values = [float(summary[(summary.channel == channel) & (summary.hypothesis == hypothesis)].max_C0.iloc[0]) for channel in channels]
        axes[0, 0].bar(x + (index - 1.5) * 0.18, values, 0.18, label=hypothesis, color=COLORS[index])
    axes[0, 0].set_xticks(x, channels, rotation=18)
    axes[0, 0].set_yscale("symlog", linthresh=0.1)
    axes[0, 0].set(title="A  Holdout operator residual", ylabel="maximum C0")
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[0, 1].plot(shape.radius, shape.baseline_C0, "o-", color=COLORS[1], label="X-only")
    axes[0, 1].plot(shape.radius, shape.shape_corrected_C0, "o-", color=COLORS[2], label="$\lambda_\perp/a$ shape field")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(xlabel="fixed-octagon radius R", ylabel="C0", title="B  Old radius path is a shape path")
    axes[0, 1].legend(frameon=False)
    combined = holdout[holdout.channel == "combined"]
    labels = combined.case_id.tolist()
    z = np.arange(len(labels))
    axes[1, 0].plot(z, combined.H0_C0, "o-", label="H0", color=COLORS[0])
    axes[1, 0].plot(z, combined.H3_C0, "o-", label="H3", color=COLORS[2])
    axes[1, 0].plot(z, combined.H4_C0, "o-", label="H4", color=COLORS[3])
    axes[1, 0].axhline(0.20, color="black", ls="--", lw=0.8, label="gate")
    axes[1, 0].set_xticks(z, labels, rotation=18)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="C  Joint independent holdout", ylabel="C0")
    axes[1, 0].legend(frameon=False, ncol=2)
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.95, "FINAL CLASSIFICATION\nPASS_RESTRICTED_CLASS\n\nLocal fixed-a operator tangent: rank 2\nScalar observable identifiability: not stable\nH3 two-parameter holdout: fail\nH4 three-field holdout: fail\n\nNo FAIL_THEORY", va="top", fontsize=9)
    fig.suptitle("Figure 16 — Curvature-relevant universality audit", fontsize=11)
    fig.tight_layout()
    save(fig, figures, "figure16_curvature_relevant_universality")


def figure18(data: Path, figures: Path) -> None:
    singular = pd.read_parquet(data / "figure_18_singular_values.parquet")
    summary = pd.read_parquet(data / "figure_18_hypothesis_summary.parquet")
    curvature = pd.read_parquet(data / "figure_18_curvature_scaling.parquet")
    blocks = pd.read_parquet(data / "r16_block_rank.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for color, (family, group) in zip(COLORS, singular.groupby("family")):
        axes[0, 0].plot(group["index"], group.singular_value, "o-", color=color, label=family.replace("_", " "))
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="singular-value index", ylabel="singular value", title="A  Operator vs observable rank")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(blocks.block, blocks.PhiX_PhiG_s2_over_s1, "o-", color=COLORS[2])
    axes[0, 1].axhline(0.05, color="black", ls="--", lw=0.8)
    axes[0, 1].set(xlabel="momentum block", ylabel="$s_2/s_1$", title="B  Local rank stability")
    axes[1, 0].loglog(curvature.coordinate, curvature.actual_C0, "o", color=COLORS[1], label="holdout")
    axes[1, 0].loglog(curvature.coordinate, curvature.predicted_C0, "--", color=COLORS[0], label="frozen training fit")
    axes[1, 0].set(xlabel="$|g-g_0|$", ylabel="X-only C0", title="C  Curvature scaling falsification")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(summary.hypothesis, summary.median_C0, color=COLORS[: len(summary)])
    axes[1, 1].axhline(0.20, color="black", ls="--", lw=0.8)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(title="D  Median joint holdout error", ylabel="median C0")
    axes[1, 1].text(0.03, 0.96, "local operator evidence only\nglobal curvature relevance: INCONCLUSIVE", transform=axes[1, 1].transAxes, va="top", fontsize=7.5)
    fig.suptitle("Figure 18 — Is curvature an independent relevant coordinate?", fontsize=11)
    fig.tight_layout()
    save(fig, figures, "figure18_curvature_coordinate_rank")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    data = run_dir / "figure_data"
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()
    figure8(data, figures)
    figure9(data, figures)
    figure10(data, figures)
    figure16(data, figures)
    figure18(data, figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

