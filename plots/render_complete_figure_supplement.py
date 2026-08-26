from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


TERRACOTTA = "#A64B2A"
WARM_RED = "#8F2D1C"
ORANGE = "#D97706"
AMBER = "#E9A23B"
MUTED_GOLD = "#B8860B"
WARM_BROWN = "#6B3E26"
ROSE = "#C45A4A"
SAND = "#D8A45B"
CREAM = "#FFF8EF"
PAPER = "#FFFDFC"
GRID = "#D8C5B4"
DARK = "#3B2416"
MUTED = "#705447"
PALETTE = [TERRACOTTA, ORANGE, MUTED_GOLD, WARM_RED, ROSE, SAND, WARM_BROWN]
LINESTYLES = ["-", "-.", "--", ":"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "axes.linewidth": 0.8,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.3,
            "legend.frameon": False,
            "text.color": DARK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "svg.fonttype": "none",
            "lines.linewidth": 1.15,
            "lines.markersize": 4.0,
        }
    )


def style_axis(axis: plt.Axes, panel: str | None = None) -> None:
    axis.grid(True, color=GRID, linewidth=0.45, linestyle="--", alpha=0.65)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(DARK)
    axis.tick_params(width=0.65, length=3.0)
    if panel:
        axis.text(
            0.015,
            0.975,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=DARK,
        )


def make_figure(title: str, subtitle: str, panels: tuple[int, int], size: tuple[float, float]):
    figure, axes = plt.subplots(*panels, figsize=size, squeeze=False)
    figure.suptitle(title, x=0.5, y=0.982, fontsize=15, fontweight="bold", color=DARK)
    figure.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=9.2, color=MUTED)
    figure.subplots_adjust(left=0.075, right=0.975, bottom=0.13, top=0.84, wspace=0.28, hspace=0.42)
    frame = Rectangle(
        (0.006, 0.009),
        0.988,
        0.982,
        transform=figure.transFigure,
        fill=False,
        edgecolor=DARK,
        linewidth=0.8,
        zorder=1000,
    )
    frame.set_gid("figure-frame")
    figure.add_artist(frame)
    return figure, axes


def save_figure(figure: plt.Figure, svg: Path, png: Path) -> None:
    figure.savefig(svg, format="svg", dpi=240, metadata={"Creator": "Frozen-data figure supplement renderer"})
    figure.savefig(png, format="png", dpi=240, metadata={"Creator": "Frozen-data figure supplement renderer"})
    plt.close(figure)


def load_npz(extracted: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(extracted / f"{name}.npz", allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def load_frame(extracted: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(extracted / f"{name}.csv")


def short_tower(value: str) -> str:
    return (
        str(value)
        .replace("congruence_", "")
        .replace("_r", "/r")
        .replace("_L", " L")
        .replace("_level_", " L")
    )


def figure01(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    data = load_npz(extracted, "figure01_moire_length")
    errors = load_frame(extracted, "figure01_errors")
    figure, axes = make_figure(
        "Figure 1. Hyperbolic moire-length validation",
        "Frozen G-02 analytic/numeric agreement and saved residuals; no geometry was recomputed",
        (1, 2),
        (11.2, 4.7),
    )
    ax = axes[0, 0]
    positive = (data["xi_analytic"] > 0) & (data["xi_numeric"] > 0)
    ax.loglog(data["xi_analytic"][positive], data["xi_numeric"][positive], "o", color=TERRACOTTA, alpha=0.8, label="45 frozen samples")
    low = float(min(data["xi_analytic"][positive].min(), data["xi_numeric"][positive].min()))
    high = float(max(data["xi_analytic"][positive].max(), data["xi_numeric"][positive].max()))
    ax.plot([low, high], [low, high], linestyle="-.", color=WARM_BROWN, label="analytic = numeric")
    ax.set_xlabel("Analytic moire length ξ")
    ax.set_ylabel("Numeric moire length ξ")
    ax.legend(loc="upper left")
    style_axis(ax, "a")

    ax = axes[0, 1]
    for index, (radius, group) in enumerate(errors.groupby("R")):
        for style_index, (_, subset) in enumerate(group.groupby("threshold")):
            ordered = subset.sort_values("theta")
            label = f"R={radius:g}" if style_index == 0 else None
            ax.plot(
                ordered["theta"],
                ordered["absolute_error"],
                color=PALETTE[index % len(PALETTE)],
                linestyle=LINESTYLES[style_index % len(LINESTYLES)],
                marker="o" if style_index == 0 else None,
                label=label,
            )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0e-18)
    ax.set_xlabel("Twist angle θ")
    ax.set_ylabel("Saved absolute error")
    ax.legend(loc="best", ncol=2)
    style_axis(ax, "b")
    save_figure(figure, svg, png)


def figure02(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    crossover = load_npz(extracted, "figure02_crossover")
    double = load_npz(extracted, "figure02_double_scaling")
    dimensional = load_npz(extracted, "figure02_dimensional_extension")
    figure, axes = make_figure(
        "Figure 2. Area, crossover, and exponent scaling collapse",
        "Frozen G-03--G-06 quantities across curvature, double-scaling, and dimensional endpoint tests",
        (1, 3),
        (14.4, 4.65),
    )
    ax = axes[0, 0]
    for index, radius in enumerate(np.unique(crossover["R"])):
        mask = crossover["R"] == radius
        order = np.argsort(crossover["chi"][mask])
        ax.semilogx(crossover["chi"][mask][order], crossover["normalized_area"][mask][order], color=PALETTE[index], linestyle=LINESTYLES[index], marker="o", label=f"R={radius:g}")
    ax.set_xlabel("Crossover variable χ")
    ax.set_ylabel("Normalized effective area")
    ax.legend(loc="best")
    style_axis(ax, "a")

    ax = axes[0, 1]
    for index, value in enumerate(np.unique(double["u"])):
        mask = double["u"] == value
        order = np.argsort(double["alpha"][mask])
        ax.loglog(double["alpha"][mask][order], double["absolute_error"][mask][order], color=PALETTE[index], linestyle=LINESTYLES[index % 3], marker="o", label=f"u={value:g}")
    ax.set_xlabel("Double-scaling parameter α")
    ax.set_ylabel("Saved absolute convergence error")
    ax.legend(loc="best", ncol=2)
    style_axis(ax, "b")

    ax = axes[0, 2]
    keys = np.column_stack((dimensional["ambient_dimension"], dimensional["active_dimension"]))
    for index, key in enumerate(np.unique(keys, axis=0)):
        mask = (keys[:, 0] == key[0]) & (keys[:, 1] == key[1])
        order = np.argsort(dimensional["y"][mask])
        ax.semilogx(dimensional["y"][mask][order], dimensional["beta"][mask][order], color=PALETTE[index], linestyle=LINESTYLES[index % 3], label=f"D={key[0]}, active={key[1]}")
    ax.set_xlabel("Dimensionless radius y")
    ax.set_ylabel("Saved effective exponent β")
    ax.legend(loc="best", fontsize=7.3)
    style_axis(ax, "c")
    save_figure(figure, svg, png)


def figure03(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    exponent = load_frame(extracted, "figure03_exponent").sort_values("j")
    locking = load_frame(extracted, "figure03_locking")
    figure, axes = make_figure(
        "Figure 3. Arithmetic exponent and radial locking",
        "Frozen G-11 exponent sequence and G-12 endpoint extrapolations",
        (1, 2),
        (11.2, 4.7),
    )
    ax = axes[0, 0]
    ax.semilogx(exponent["j"], exponent["log_q_over_log_s_inverse"], color=TERRACOTTA, marker="o", label="pointwise exponent")
    ax.axhline(4.0, color=WARM_RED, linestyle="-.", label="theorem exponent 4")
    ax.set_xlabel("Arithmetic index j")
    ax.set_ylabel("log qj / log |sj|⁻¹")
    ax.legend(loc="best")
    style_axis(ax, "a")

    ax = axes[0, 1]
    x = np.arange(len(locking))
    width = 0.23
    ax.bar(x - width, locking["target"], width, color=MUTED_GOLD, label="target")
    ax.bar(x, locking["last"], width, color=ORANGE, label="last saved scale")
    ax.bar(x + width, locking["extrapolated"], width, color=TERRACOTTA, label="frozen extrapolation")
    ax.set_xticks(x, [str(value).replace("_", " ") for value in locking["observable"]])
    ax.set_ylabel("Dimensionless locking ratio")
    ax.legend(loc="best")
    style_axis(ax, "b")
    save_figure(figure, svg, png)


def figure04(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    square = load_frame(extracted, "figure04_square")
    root_scan = load_frame(extracted, "figure04_root_scan")
    root_summary = load_frame(extracted, "figure04_root_summary").iloc[0]
    square_certificate = json.loads((root / "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/certificates/s02_square_no_go.json").read_text(encoding="utf-8"))
    square_minimum = float(str(square_certificate["arb_positive_ball"]).split()[0].strip("["))
    figure, axes = make_figure(
        "Figure 4. Square no-go versus hyperbolic positive root",
        "Exact square response remains positive, while the hyperbolic character response crosses a certified simple root",
        (1, 2),
        (11.2, 4.7),
    )
    ax = axes[0, 0]
    ax.plot(np.arange(len(square)), square["curvature_float"], color=ORANGE, marker="o", markevery=8, label="exact square samples")
    ax.axhline(square_minimum, color=WARM_RED, linestyle="-.", label="certified positive minimum")
    ax.set_xlabel("Ordered nonnegative α sample")
    ax.set_ylabel("Square curvature response")
    ax.legend(loc="best")
    style_axis(ax, "a")

    ax = axes[0, 1]
    ax.plot(root_scan["w_over_t"], root_scan["F1"], color=TERRACOTTA, label="hyperbolic F1")
    ax.axhline(0.0, color=WARM_BROWN, linestyle="--")
    ax.axvline(float(root_summary["root_w"]), color=WARM_RED, linestyle="-.", label=f"root = {root_summary['root_w']:.6f}")
    ax.set_xlabel("Coupling w/t")
    ax.set_ylabel("Character response F1")
    ax.legend(loc="best")
    style_axis(ax, "b")
    save_figure(figure, svg, png)


def figure05(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    shells = load_frame(extracted, "figure05_shells")
    displacement = load_frame(extracted, "figure05_displacement").iloc[0]
    figure, axes = make_figure(
        "Figure 5. Full-kernel shell and root-gap validation",
        "Frozen S-05 shell contributions and S-06 certified root displacement",
        (1, 2),
        (11.2, 4.7),
    )
    ax = axes[0, 0]
    ax.plot(shells["word_length"], shells["weight_sum"], color=TERRACOTTA, marker="o", label="shell weight sum")
    ax.plot(shells["word_length"], shells["hodge_trace_sum"], color=ORANGE, linestyle="-.", marker="s", label="Hodge trace sum")
    ax.set_xlabel("Surface-group word length")
    ax.set_ylabel("Saved shell contribution")
    ax.legend(loc="best")
    style_axis(ax, "a")

    ax = axes[0, 1]
    labels = ["first-shell root", "finite full root", "measured displacement", "theoretical bound"]
    values = [
        displacement["first_shell_root"],
        displacement["finite_full_root"],
        displacement["measured_displacement"],
        displacement["sign_aware_theoretical_bound"],
    ]
    ax.bar(np.arange(4), values, color=[MUTED_GOLD, TERRACOTTA, ORANGE, WARM_BROWN])
    ax.vlines(
        1,
        displacement["certified_full_root_lower"],
        displacement["certified_full_root_upper"],
        color=WARM_RED,
        linewidth=1.4,
        linestyles="-.",
        label="certified full-root interval",
    )
    ax.set_xticks(np.arange(4), labels, rotation=18, ha="right")
    ax.set_ylabel("Coupling scale")
    ax.legend(loc="best")
    style_axis(ax, "b")
    save_figure(figure, svg, png)


def figure06(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    tensor = load_npz(extracted, "figure06_hodge_tensor")
    s09 = json.loads((root / "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/certificates/s09_hodge_basis.json").read_text(encoding="utf-8"))
    s11 = json.loads((root / "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/certificates/s11_hodge_tensor.json").read_text(encoding="utf-8"))
    s12 = json.loads((root / "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/certificates/s12_complete_cancellation.json").read_text(encoding="utf-8"))
    figure, axes = make_figure(
        "Figure 6. Hodge response and complete cancellation",
        "Frozen four-dimensional response tensor and exact cancellation certificates",
        (1, 2),
        (11.2, 4.7),
    )
    ax = axes[0, 0]
    for index in range(4):
        ax.plot(tensor["w"], tensor["K"][:, index, index], color=PALETTE[index], linestyle=LINESTYLES[index], label=f"K{index + 1}{index + 1}")
    ax.axhline(0.0, color=WARM_BROWN, linestyle="--")
    ax.set_xlabel("Coupling w")
    ax.set_ylabel("Saved Hodge-response component")
    ax.legend(loc="best", ncol=2)
    style_axis(ax, "a")

    ax = axes[0, 1]
    labels = ["basis coordinate", "finite symmetry", "exact cancellation"]
    values = [
        float(s09["generalized_eigenvalue_coordinate_residual"]),
        float(s11["finite_symmetry_residual"]),
        float(s12["operator_norm_at_exact_parameter_dependent_root"]),
    ]
    ax.bar(np.arange(3), values, color=[MUTED_GOLD, ORANGE, TERRACOTTA])
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.2e}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(np.arange(3), labels, rotation=17, ha="right")
    ax.set_ylabel("Certified residual")
    ax.set_ylim(0, max(values + [1.0e-18]) * 1.35)
    style_axis(ax, "b")
    save_figure(figure, svg, png)


def figure07(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    symmetry = load_frame(extracted, "figure07_symmetry").sort_values("bond_anisotropy")
    robust = load_frame(extracted, "figure07_robustness")
    compare = load_frame(extracted, "figure07_comparison")
    figure, axes = make_figure(
        "Figure 7. Symmetry breaking and multiorbital robustness",
        "Frozen S-14/S-15 scans and S-23 separation of degeneracy from kinetic flattening",
        (1, 3),
        (14.4, 4.7),
    )
    ax = axes[0, 0]
    ax.plot(symmetry["bond_anisotropy"], symmetry["hessian_spread"], color=TERRACOTTA, label="Hessian spread")
    ax.plot(symmetry["bond_anisotropy"], symmetry["traceless_response"], color=ORANGE, linestyle="-.", label="traceless response")
    ax.set_xlabel("Bond anisotropy")
    ax.set_ylabel("Saved response amplitude")
    ax.legend(loc="best")
    style_axis(ax, "a")

    ax = axes[0, 1]
    x = np.arange(len(robust))
    ax.plot(x, robust["root_w"], color=TERRACOTTA, marker="o", label="root w")
    ax.plot(x, robust["gap"], color=MUTED_GOLD, linestyle="-.", marker="s", label="gap")
    ax.set_xticks(x, robust["name"], rotation=70, ha="right", fontsize=7.1)
    ax.set_ylabel("Frozen root / gap value")
    ax.legend(loc="best")
    style_axis(ax, "b")

    ax = axes[0, 2]
    for index, (name, group) in enumerate(compare.groupby("perturbation_class")):
        ordered = group.sort_values("epsilon")
        ax.plot(ordered["epsilon"], ordered["symmetry_degeneracy_spread"], color=PALETTE[index], marker="o", label=f"{name.replace('_', ' ')}: degeneracy")
        ax.plot(ordered["epsilon"], ordered["kinetic_flattening_operator_norm"], color=PALETTE[index], linestyle="-.", marker="s", label=f"{name.replace('_', ' ')}: flattening")
    ax.set_xlabel("Perturbation ε")
    ax.set_ylabel("Saved spectral response")
    ax.legend(loc="best", fontsize=7.0)
    style_axis(ax, "c")
    save_figure(figure, svg, png)


def figure08(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    edges = load_frame(extracted, "figure08_edges")
    cross = load_frame(extracted, "figure08_cross_tower")
    radius = load_frame(extracted, "figure08_radius")
    figure, axes = make_figure(
        "Figure 8. Certified non-Abelian cover convergence",
        "Only the three admissible non-Abelian congruence towers are shown",
        (1, 3),
        (14.4, 4.7),
    )
    ax = axes[0, 0]
    for index, (tower, group) in enumerate(radius.groupby("tower_id")):
        ordered = group.sort_values("level")
        ax.plot(ordered["level"], ordered["injectivity_radius_word_lower"], color=PALETTE[index], linestyle=LINESTYLES[index], marker="o", label=short_tower(tower))
        nonmaterial = ~ordered["materialized"].astype(bool)
        ax.scatter(ordered.loc[nonmaterial, "level"], ordered.loc[nonmaterial, "injectivity_radius_word_lower"], facecolors=PAPER, edgecolors=PALETTE[index], zorder=3)
    ax.set_xlabel("Tower level")
    ax.set_ylabel("Certified word injectivity lower bound")
    ax.set_xticks([1, 2])
    ax.legend(loc="best")
    style_axis(ax, "a")

    ax = axes[0, 1]
    labels = [f"{short_tower(row.tower_id)} L{row.level}" for row in edges.itertuples()]
    x = np.arange(len(edges))
    ax.plot(x, edges["bandwidth"], color=TERRACOTTA, marker="o", label="bandwidth")
    ax.plot(x, edges["external_gap"], color=ORANGE, linestyle="-.", marker="s", label="external gap")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Transported spectral scale")
    ax.legend(loc="best")
    style_axis(ax, "b")

    ax = axes[0, 2]
    labels = [f"{short_tower(a)}–{short_tower(b)}" for a, b in zip(cross["cover_a"], cross["cover_b"])]
    x = np.arange(len(cross))
    for index, column in enumerate(["bandwidth_residual", "heat_trace_residual", "fixed_broadening_dos_sup_residual"]):
        ax.plot(x, cross[column], color=PALETTE[index], linestyle=LINESTYLES[index], marker=["o", "s", "^"][index], label=column.replace("_", " "))
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Cross-tower residual")
    ax.legend(loc="best", fontsize=7.1)
    style_axis(ax, "c")
    save_figure(figure, svg, png)


def figure09(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    balance = load_frame(extracted, "figure09_balance")
    spectral = load_frame(extracted, "figure09_spectral")
    labels = [f"{short_tower(row.tower_id)} L{row.level}" for row in balance.itertuples()]
    x = np.arange(len(balance))
    figure, axes = make_figure(
        "Figure 9. Full-shell and spectrum convergence",
        "Frozen B-11 balanced shell errors and B-12 inherited spectral bounds",
        (1, 2),
        (11.2, 4.7),
    )
    ax = axes[0, 0]
    for index, column in enumerate(["core_error", "physical_tail", "master_tail", "balanced_error_sum"]):
        ax.plot(x, balance[column], color=PALETTE[index], linestyle=LINESTYLES[index], marker=["o", "s", "^", "D"][index], label=column.replace("_", " "))
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Saved full-shell error component")
    ax.legend(loc="best")
    style_axis(ax, "a")

    ax = axes[0, 1]
    for index, column in enumerate(["spectral_hausdorff_error_upper", "bandwidth_error_upper", "gap_error_upper", "riesz_projection_norm_error_upper"]):
        ax.plot(x, spectral[column], color=PALETTE[index], linestyle=LINESTYLES[index], marker=["o", "s", "^", "D"][index], label=column.replace("_upper", "").replace("_", " "))
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Certified spectral error upper bound")
    ax.legend(loc="best")
    style_axis(ax, "b")
    save_figure(figure, svg, png)


def figure10(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    cdf = load_frame(extracted, "figure10_cdf").sort_values("quotient_order")
    dos = load_frame(extracted, "figure10_dos")
    figure, axes = make_figure(
        "Figure 10. Bulk DOS and local spectral observables",
        "D-02 PASS_CONVERGED; D-05 remains INCONCLUSIVE and is shown without relabelling",
        (1, 2),
        (11.2, 4.7),
    )
    ax = axes[0, 0]
    ax.loglog(cdf["quotient_order"], cdf["kappa_N"], color=TERRACOTTA, marker="o")
    for row in cdf.itertuples():
        ax.annotate(short_tower(row.tower_id), (row.quotient_order, row.kappa_N), xytext=(3, 4), textcoords="offset points", fontsize=7.0)
    ax.set_xlabel("Certified quotient order")
    ax.set_ylabel("Retained-sector CDF error κN")
    style_axis(ax, "a")

    ax = axes[0, 1]
    ax.plot(dos["energy"], dos["layer_even_coherence_weighted_density"], color=ORANGE)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Coherence-weighted density")
    ax.text(0.98, 0.94, "INCONCLUSIVE", transform=ax.transAxes, ha="right", va="top", color=WARM_RED, fontweight="bold")
    style_axis(ax, "b")
    save_figure(figure, svg, png)


def figure11(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    dos = load_frame(extracted, "figure11_dos")
    circuit = load_frame(extracted, "figure11_circuit")
    graph = load_frame(extracted, "figure11_graph")
    figure, axes = make_figure(
        "Figure 11. External HyperBloch and circuit reproduction",
        "Frozen parser-based public graph/DOS comparison and circuit-Laplacian reconstruction",
        (1, 2),
        (11.2, 4.7),
    )
    ax = axes[0, 0]
    for index, (benchmark, group) in enumerate(dos.groupby("benchmark")):
        ax.plot(group["energy"], group["density"], color=PALETTE[index], linestyle=LINESTYLES[index], label=str(benchmark))
    ax.set_xlabel("Energy")
    ax.set_ylabel("Broadened public DOS")
    ax.legend(loc="best", fontsize=7.2)
    graph_residuals = ", ".join(f"{value:.1e}" for value in graph["spectral_sup_residual"])
    ax.text(0.02, 0.03, f"D-10 saved spectral residuals: {graph_residuals}", transform=ax.transAxes, fontsize=8.0, color=MUTED)
    style_axis(ax, "a")

    ax = axes[0, 1]
    ax.plot(circuit["eigen_index"], circuit["absolute_residual"], color=TERRACOTTA)
    ax.set_yscale("symlog", linthresh=1.0e-18)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Absolute reconstruction residual")
    style_axis(ax, "b")
    save_figure(figure, svg, png)


def figure12(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    reverse = load_frame(extracted, "figure12_reverse")
    nc03 = load_frame(extracted, "figure12_nc03")
    nc07 = load_frame(extracted, "figure12_nc07")
    matrix = load_frame(extracted, "figure12_matrix")
    expected = matrix.loc[matrix["status"] == "FAIL_EXPECTED", "code_id"].astype(str).tolist()
    figure, axes = make_figure(
        "Figure 12. Negative controls and expected falsification",
        "All FAIL_EXPECTED outcomes remain explicit; correct controls are retained alongside rejected alternatives",
        (2, 2),
        (11.4, 8.0),
    )
    ax = axes[0, 0]
    x = np.arange(len(reverse))
    colors = [MUTED_GOLD if value == "correct_control" else WARM_RED for value in reverse["control"]]
    ax.bar(x, reverse["collapse_residual"], color=colors)
    ax.set_xticks(x, [str(value).replace("_", " ") for value in reverse["control"]], rotation=35, ha="right")
    ax.set_ylabel("Frozen collapse residual")
    style_axis(ax, "a")

    ax = axes[0, 1]
    ax.semilogy(nc03["L_over_R"], nc03["delta_theta_exp_L_over_R"], color=WARM_RED, marker="o")
    ax.set_xlabel("L/R")
    ax.set_ylabel("Wrong joint-limit residual")
    style_axis(ax, "b")

    ax = axes[1, 0]
    ax.loglog(nc07["N"], nc07["C0_operator_error"], color=TERRACOTTA, marker="o", label="C0 operator error")
    ax.loglog(nc07["N"], nc07["C1_derivative_error"], color=ORANGE, linestyle="-.", marker="s", label="C1 derivative error")
    ax.set_xlabel("N")
    ax.set_ylabel("Negative-control error")
    ax.legend(loc="best")
    style_axis(ax, "c")

    ax = axes[1, 1]
    y = np.arange(len(expected))
    ax.barh(y, np.ones(len(expected)), color=WARM_RED)
    ax.set_yticks(y, expected)
    ax.set_xlim(0, 1.03)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_xlabel("Expected falsification recorded")
    style_axis(ax, "d")
    save_figure(figure, svg, png)


def figure13(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    frame = load_frame(extracted, "figure13_complexity").sort_values("j")
    figure, axes = make_figure(
        "Figure 13. Exact sampled magic complexity",
        "Style-only rerender from frozen G-13/G-14 data",
        (1, 1),
        (8.0, 5.0),
    )
    ax = axes[0, 0]
    x = 1.0 / np.sqrt(frame["omega_j"].to_numpy(float))
    ax.plot(x, frame["sqrt_omega_log_q"], color=TERRACOTTA, marker="o", label="saved sequence")
    ax.plot(x, frame["target_4c"], color=WARM_RED, linestyle="-.", label="frozen 4cν target")
    ax.set_xlabel("1 / √ωj")
    ax.set_ylabel("√ωj log qj")
    ax.legend(loc="best")
    style_axis(ax, "a")
    save_figure(figure, svg, png)


def figure14(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    frame = load_frame(extracted, "figure14_joint_limit")
    figure, axes = make_figure(
        "Figure 14. Joint incommensurate-limit falsification",
        "Style-only rerender; certified non-Abelian towers and both frozen control sequences are preserved",
        (1, 1),
        (8.0, 5.0),
    )
    ax = axes[0, 0]
    for index, ((sequence, tower), group) in enumerate(frame.groupby(["sequence", "tower_id"])):
        ordered = group.sort_values("L_over_R")
        ax.semilogy(ordered["L_over_R"], ordered["joint_limit_residual"], color=PALETTE[index % len(PALETTE)], linestyle="-" if sequence == "passing" else "-.", marker="o", label=f"{sequence.replace('_', ' ')} · {short_tower(tower)}")
    ax.set_xlabel("Certified L/R lower bound")
    ax.set_ylabel("|δθ| exp(L/R)")
    ax.legend(loc="best", fontsize=7.2, ncol=2)
    style_axis(ax, "a")
    save_figure(figure, svg, png)


def figure15(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    data = load_npz(extracted, "figure15_operational_magic")
    figure, axes = make_figure(
        "Figure 15. Operational magic target spectrum",
        "Style-only rerender from the frozen S-17 finite active-fiber spectrum",
        (1, 1),
        (8.0, 5.0),
    )
    ax = axes[0, 0]
    for band in range(data["complete_eigenvalues"].shape[1]):
        ax.plot(data["q"], data["complete_eigenvalues"][:, band], color=[SAND, ORANGE, ROSE][band], linestyle=LINESTYLES[band], alpha=0.85, label=f"complete band {band + 1}")
    ax.plot(data["q"], data["target_energy"], color=WARM_RED, linewidth=1.4, label="transported target")
    ax.set_xlabel("Transported momentum q")
    ax.set_ylabel("Energy / t")
    ax.legend(loc="best")
    style_axis(ax, "a")
    save_figure(figure, svg, png)


def figure16(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    data = load_npz(extracted, "figure16_master_collapse")
    metadata = json.loads((extracted / "extraction_metadata.json").read_text(encoding="utf-8"))
    labels = metadata["datasets"]["figure16_master_collapse"]["source_attrs"]["labels"]
    selected = [index for index, label in enumerate(labels) if str(label).endswith("X=1.00")]
    figure, axes = make_figure(
        "Figure 16. Complete-spectrum master-collapse audit",
        "Style-only rerender; S-18 remains INCONCLUSIVE and is restricted to the frozen finite active fiber",
        (1, 1),
        (8.0, 5.0),
    )
    ax = axes[0, 0]
    for palette_index, index in enumerate(selected):
        label = str(labels[index]).split(":")[0]
        ax.plot(data["q"], data["normalized_target_bands"][index], color=PALETTE[palette_index], linestyle=LINESTYLES[palette_index % 3], label=label)
    ax.set_xlabel("Transported momentum q")
    ax.set_ylabel("Normalized target energy")
    ax.legend(loc="best", ncol=2)
    ax.text(0.98, 0.04, "INCONCLUSIVE", transform=ax.transAxes, ha="right", va="bottom", color=WARM_RED, fontweight="bold")
    style_axis(ax, "a")
    save_figure(figure, svg, png)


def figure17(root: Path, extracted: Path, svg: Path, png: Path) -> None:
    data = load_npz(extracted, "figure17_magic_landscape")
    figure, axes = make_figure(
        "Figure 17. Operational magic landscape",
        "Style-only rerender of all three frozen curvature slices; finite active-fiber scope",
        (1, 3),
        (14.4, 4.65),
    )
    minimum = float(data["score_M"].min())
    maximum = float(data["score_M"].max())
    image = None
    for index, axis in enumerate(axes[0]):
        image = axis.imshow(
            data["score_M"][index],
            origin="lower",
            aspect="auto",
            extent=[float(data["w_over_t"].min()), float(data["w_over_t"].max()), float(data["theta"].min()), float(data["theta"].max())],
            cmap="YlOrRd",
            vmin=minimum,
            vmax=maximum,
            interpolation="nearest",
        )
        axis.set_title(f"K = {data['K'][index]:.6f}")
        axis.set_xlabel("w/t")
        if index == 0:
            axis.set_ylabel("Twist angle θ")
        style_axis(axis, chr(ord("a") + index))
    colorbar = figure.colorbar(image, ax=list(axes[0]), fraction=0.02, pad=0.02)
    colorbar.set_label("Frozen magic score M")
    colorbar.outline.set_edgecolor(WARM_RED)
    colorbar.outline.set_linewidth(0.8)
    save_figure(figure, svg, png)


BUILDERS = [
    figure01,
    figure02,
    figure03,
    figure04,
    figure05,
    figure06,
    figure07,
    figure08,
    figure09,
    figure10,
    figure11,
    figure12,
    figure13,
    figure14,
    figure15,
    figure16,
    figure17,
]

SLUGS = [
    "hyperbolic_moire_length",
    "geometric_scaling_collapse",
    "arithmetic_exponent_radial_locking",
    "square_nogo_hyperbolic_root",
    "full_kernel_gap_validation",
    "hodge_cancellation",
    "symmetry_robustness",
    "nonabelian_cover_convergence",
    "full_shell_spectrum_convergence",
    "bulk_dos_local_observables",
    "external_reproduction",
    "negative_controls",
    "magic_complexity",
    "joint_limit",
    "operational_spectrum",
    "master_collapse",
    "magic_landscape",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--extracted-inputs", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    extracted = args.extracted_inputs.resolve()
    svg_output = args.svg_output.resolve()
    png_output = args.png_output.resolve()
    svg_output.mkdir(parents=True, exist_ok=False)
    png_output.mkdir(parents=True, exist_ok=False)
    configure_style()
    for number, (slug, builder) in enumerate(zip(SLUGS, BUILDERS), start=1):
        builder(
            root,
            extracted,
            svg_output / f"figure{number:02d}_{slug}.svg",
            png_output / f"figure{number:02d}_{slug}.png",
        )
    print(json.dumps({"figure_count": len(BUILDERS), "scientific_results_computed": False}))


if __name__ == "__main__":
    main()

