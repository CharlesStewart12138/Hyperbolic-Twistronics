from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import zarr


ORIGINAL_BUILDERS = {
    1: "figure01",
    2: "figure02",
    3: "figure03",
    4: "figure04",
    5: "figure05",
    6: "figure06",
    7: "figure07",
    11: "figure11",
    12: "figure12",
    13: "figure13",
    14: "figure14",
    15: "figure15",
    17: "figure17",
}

OUTPUT_STEMS = {
    1: "figure01_hyperbolic_moire_length_beautified",
    2: "figure02_geometric_scaling_collapse_beautified",
    3: "figure03_arithmetic_exponent_radial_locking_beautified",
    4: "figure04_square_nogo_hyperbolic_root_beautified",
    5: "figure05_full_kernel_gap_validation_beautified",
    6: "figure06_hodge_cancellation_beautified",
    7: "figure07_symmetry_robustness_beautified",
    8: "figure08_deep_finite_cover_closure_beautified",
    9: "figure09_balanced_full_shell_beautified",
    10: "figure10_vanishing_broadening_DOS_beautified",
    11: "figure11_external_reproduction_beautified",
    12: "figure12_negative_controls_beautified",
    13: "figure13_magic_complexity_beautified",
    14: "figure14_joint_limit_beautified",
    15: "figure15_operational_spectrum_beautified",
    16: "figure16_curvature_relevant_universality_beautified",
    17: "figure17_magic_landscape_beautified",
    18: "figure18_curvature_coordinate_rank_beautified",
}

TITLE_MAP = {
    1: "Hyperbolic Moire-Length Consistency",
    2: "Geometric and Dimensional Scaling Collapse",
    3: "Arithmetic Exponent and Radial Locking",
    4: "Square-Lattice No-Go and Hyperbolic Positive Root",
    5: "Full-Kernel Shell Closure and Root-Gap Certification",
    6: "Hodge-Response Structure and Exact Cancellation",
    7: "Symmetry Breaking and Multiorbital Robustness",
    8: "Weak Bulk Spectral Consistency Across Certified Non-Abelian Towers",
    9: "Balanced Full-Shell Error Hierarchy",
    10: "Vanishing-Broadening Spectral-Measure Hierarchy",
    11: "External HyperBloch and Circuit Reproduction",
    12: "Negative Controls and Expected Falsification",
    13: "Exact Sampled Magic Complexity",
    14: "Joint Incommensurate-Limit Falsification",
    15: "Operational Magic-Target Spectrum",
    16: "Restricted-Class Operator Collapse and Profile-Coordinate Correction",
    17: "Operational Magic Landscape",
    18: "Operator-Tangent Rank Structure of the Curvature-Related Sector",
}

PAPER = "#FFFCF7"
INK = "#34261F"
MUTED_INK = "#6B584C"
GRID = "#D8C9BC"
PALETTE = ["#8E4B32", "#B66A3C", "#9A6B3E", "#7A4E42", "#B07C64", "#6F5A48"]
LINESTYLES = ["-.", "--", ":", "-"]


TEXT_REPLACEMENTS = {
    "Analytic moire length xi": r"Analytic moire length, $\xi$",
    "Numeric moire length xi": r"Numeric moire length, $\xi$",
    "Twist angle theta": r"Twist angle, $\theta$",
    "Crossover variable chi": r"Crossover variable, $\chi$",
    "Double-scaling parameter alpha": r"Double-scaling parameter, $\alpha$",
    "Dimensionless radius y": r"Dimensionless radius, $y$",
    "Saved effective exponent beta": r"Saved effective exponent, $\beta$",
    "Arithmetic index j": r"Arithmetic index, $j$",
    "log qj / log |sj|^-1": r"$\log q_j/\log |s_j|^{-1}$",
    "Ordered nonnegative alpha sample": r"Ordered nonnegative $\alpha$ sample",
    "hyperbolic F1": r"hyperbolic $F_1$",
    "Coupling w/t": r"Coupling, $w/t$",
    "Character response F1": r"Character response, $F_1$",
    "Coupling w": r"Coupling, $w$",
    "Perturbation epsilon": r"Perturbation, $\varepsilon$",
    "root w": r"root $w$",
    "L/R": r"$L/R$",
    "N": r"$N$",
    "C0 operator error": r"$C^0$ operator error",
    "C1 derivative error": r"$C^1$ derivative error",
    "1 / sqrt(omega_j)": r"$1/\sqrt{\omega_j}$",
    "sqrt(omega_j) log qj": r"$\sqrt{\omega_j}\,\log q_j$",
    "frozen 4cnu target": r"frozen $4c\nu$ target",
    "Certified L/R lower bound": r"Certified $L/R$ lower bound",
    "|delta theta| exp(L/R)": r"$|\delta\theta|\exp(L/R)$",
    "Transported momentum q": r"Transported momentum, $q$",
    "Energy / t": r"Energy, $E/t$",
    "w/t": r"$w/t$",
    "Twist angle theta": r"Twist angle, $\theta$",
    "Frozen magic score M": r"Frozen magic score, $M$",
    "Retained-sector CDF error kappa_N": r"Retained-sector CDF error, $\kappa_N$",
    "analytic = numeric": r"$\xi_{\mathrm{analytic}}=\xi_{\mathrm{numeric}}$",
    "theorem exponent 4": r"theorem exponent $\beta=4$",
    "shell weight sum": r"shell weight sum",
    "Hodge trace sum": r"Hodge trace sum",
    "certified full-root interval": r"certified full-root interval",
    "Frozen root / gap value": r"Frozen root/gap value",
    "Wrong joint-limit residual": r"Wrong joint-limit residual, $|\delta\theta|\exp(L/R)$",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def configure_style() -> None:
    for path in (
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/timesi.ttf",
        "C:/Windows/Fonts/timesbi.ttf",
    ):
        font_manager.fontManager.addfont(path)
    resolved = Path(font_manager.findfont("Times New Roman", fallback_to_default=False)).resolve()
    if resolved.name.lower() != "times.ttf":
        raise RuntimeError(f"Times New Roman regular font did not resolve to times.ttf: {resolved}")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 9.2,
            "axes.titlesize": 9.8,
            "axes.titleweight": "normal",
            "axes.labelsize": 9.2,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.72,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.framealpha": 0.90,
            "legend.edgecolor": MUTED_INK,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.dpi": 450,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.cal": "Times New Roman:italic",
            "axes.unicode_minus": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 5.2,
        }
    )


def normalize_key(value: str) -> str:
    return (
        value.replace("ξ", "xi")
        .replace("θ", "theta")
        .replace("χ", "chi")
        .replace("α", "alpha")
        .replace("β", "beta")
        .replace("ε", "epsilon")
        .replace("κN", "kappa_N")
        .replace("√ωj", "sqrt(omega_j)")
        .replace("ωj", "omega_j")
        .replace("qj", "qj")
        .replace("sj", "sj")
        .replace("⁻¹", "^-1")
        .replace("δθ", "delta theta")
        .replace("ν", "nu")
        .strip()
    )


def math_text(value: str) -> str:
    if not value or "$" in value:
        return value
    key = normalize_key(value)
    if key in TEXT_REPLACEMENTS:
        return TEXT_REPLACEMENTS[key]
    match = re.fullmatch(r"R=([0-9.eE+\-]+)", value)
    if match:
        return rf"$R={match.group(1)}$"
    match = re.fullmatch(r"u=([0-9.eE+\-]+)", value)
    if match:
        return rf"$u={match.group(1)}$"
    match = re.fullmatch(r"D=([0-9]+), active=([0-9]+)", value)
    if match:
        return rf"$D={match.group(1)},\ d_{{\mathrm{{active}}}}={match.group(2)}$"
    match = re.fullmatch(r"root = ([0-9.eE+\-]+)", value)
    if match:
        return rf"root $w/t={match.group(1)}$"
    match = re.fullmatch(r"K([1-4])([1-4])", value)
    if match:
        return rf"$K_{{{match.group(1)}{match.group(2)}}}$"
    match = re.fullmatch(r"K = ([0-9.eE+\-]+)", value)
    if match:
        return rf"$K={match.group(1)}$"
    return value.replace(" · ", ": ")


def style_axis(axis: plt.Axes, panel: str | None = None) -> None:
    axis.set_axis_on()
    axis.patch.set_visible(True)
    axis.grid(True, color=GRID, linewidth=0.42, linestyle=":", alpha=0.72)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.72)
        spine.set_color(INK)
    axis.tick_params(width=0.58, length=2.8, direction="out", pad=2.5)
    if panel:
        axis.text(
            0.018,
            0.978,
            f"({panel})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.8,
            fontweight="bold",
            color=INK,
            bbox={"facecolor": PAPER, "edgecolor": "none", "alpha": 0.88, "pad": 0.8},
            zorder=20,
        )
    legend = axis.get_legend()
    if legend is not None:
        frame = legend.get_frame()
        frame.set_linewidth(0.55)
        frame.set_edgecolor(MUTED_INK)
        frame.set_facecolor(PAPER)


def finalize_figure(figure: plt.Figure) -> None:
    for axis in figure.axes:
        is_colorbar = axis.get_label() == "<colorbar>"
        if not axis.axison and not is_colorbar:
            axis.set_axis_on()
            axis.set_xticks([])
            axis.set_yticks([])
        if not is_colorbar:
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.72)
                spine.set_color(INK)
        data_lines = [line for line in axis.lines if len(np.atleast_1d(line.get_xdata())) > 1]
        for index, line in enumerate(data_lines):
            line.set_linewidth(1.0 if index == 0 else 0.82)
            if len(data_lines) > 1:
                line.set_linestyle(LINESTYLES[index % len(LINESTYLES)])
            marker = line.get_marker()
            if marker not in (None, "None", "", " "):
                line.set_marker("*")
                line.set_markersize(5.2)
                line.set_markeredgewidth(0.42)
        legend = axis.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_linewidth(0.55)
            frame.set_edgecolor(MUTED_INK)
            frame.set_facecolor(PAPER)
        for text_object in axis.findobj(mpl.text.Text):
            text_object.set_fontfamily("Times New Roman")
            text_object.set_text(math_text(text_object.get_text()))
    for text_object in figure.texts:
        text_object.set_fontfamily("Times New Roman")
        text_object.set_text(math_text(text_object.get_text()))


def save_triplet(figure: plt.Figure, run_root: Path, stem: str, title: str) -> dict[str, dict]:
    finalize_figure(figure)
    outputs = {
        "png": run_root / "rendered_png" / f"{stem}.png",
        "svg": run_root / "rendered_svg" / f"{stem}.svg",
        "pdf": run_root / "rendered_pdf" / f"{stem}.pdf",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"Refusing to overwrite an existing rendered figure: {stem}")
    figure.savefig(
        outputs["png"],
        format="png",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.055,
        metadata={"Software": "Frozen-data academic figure renderer", "Title": title},
    )
    figure.savefig(
        outputs["svg"],
        format="svg",
        bbox_inches="tight",
        pad_inches=0.055,
        metadata={"Creator": "Frozen-data academic figure renderer", "Title": title, "Description": "Style-only rerender from verified frozen data"},
    )
    figure.savefig(
        outputs["pdf"],
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.055,
        metadata={"Creator": "Frozen-data academic figure renderer", "Title": title, "Subject": "Style-only rerender from verified frozen data"},
    )
    plt.close(figure)
    return {
        kind: {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for kind, path in outputs.items()
    }


def tower_label(value: str) -> str:
    match = re.search(r"p([0-9]+)_r([0-9]+)", str(value))
    if match:
        return rf"$\mathrm{{p}}_{{{match.group(1)}}}/\mathrm{{r}}_{{{match.group(2)}}}$"
    return str(value).replace("congruence_", "").replace("_", "-")


def original_make_figure_factory(current_number: list[int]):
    def make_figure(_title: str, _subtitle: str, panels: tuple[int, int], size: tuple[float, float]):
        number = current_number[0]
        figure, axes = plt.subplots(*panels, figsize=size, squeeze=False)
        figure.suptitle(TITLE_MAP[number], x=0.5, y=0.982, fontsize=12.6, fontweight="normal", color=INK)
        top = 0.895 if panels[0] == 1 else 0.91
        figure.subplots_adjust(left=0.075, right=0.975, bottom=0.14, top=top, wspace=0.29, hspace=0.38)
        return figure, axes

    return make_figure


def render_original_figures(validation_root: Path, run_root: Path, records: dict[int, dict]) -> None:
    renderer = load_module("frozen_original_renderer", validation_root / "src/plots/render_complete_figure_supplement.py")
    extractor = load_module("frozen_original_extractor", validation_root / "src/plots/extract_figure_supplement_inputs.py")
    current_number = [0]

    def load_npz_direct(_unused: Path, alias: str) -> dict[str, np.ndarray]:
        relative, array_names = extractor.ZARR_SOURCES[alias]
        group = zarr.open_group(str(validation_root / relative), mode="r")
        return {name: np.asarray(group[name][:]) for name in array_names}

    def load_frame_direct(_unused: Path, alias: str) -> pd.DataFrame:
        return pd.read_parquet(validation_root / extractor.PARQUET_SOURCES[alias])

    renderer.load_npz = load_npz_direct
    renderer.load_frame = load_frame_direct
    renderer.make_figure = original_make_figure_factory(current_number)
    renderer.style_axis = style_axis
    renderer.configure_style = configure_style
    renderer.TERRACOTTA = PALETTE[0]
    renderer.ORANGE = PALETTE[1]
    renderer.MUTED_GOLD = PALETTE[2]
    renderer.WARM_RED = PALETTE[3]
    renderer.ROSE = PALETTE[4]
    renderer.SAND = "#C89A61"
    renderer.WARM_BROWN = PALETTE[5]
    renderer.PAPER = PAPER
    renderer.DARK = INK
    renderer.MUTED = MUTED_INK
    renderer.GRID = GRID
    renderer.PALETTE = PALETTE
    renderer.LINESTYLES = LINESTYLES

    def short_tower(value: str) -> str:
        return tower_label(value)

    renderer.short_tower = short_tower

    def save_intercept(figure: plt.Figure, svg: Path, _png: Path) -> None:
        number = current_number[0]
        records[number] = save_triplet(figure, run_root, svg.stem, TITLE_MAP[number])

    renderer.save_figure = save_intercept
    for number, builder_name in ORIGINAL_BUILDERS.items():
        current_number[0] = number
        stem = OUTPUT_STEMS[number]
        builder = getattr(renderer, builder_name)
        builder(validation_root, validation_root, Path(f"{stem}.svg"), Path(f"{stem}.png"))


def make_deep_figure(number: int, panels: tuple[int, int], size: tuple[float, float]):
    figure, axes = plt.subplots(*panels, figsize=size, squeeze=False)
    figure.suptitle(TITLE_MAP[number], x=0.5, y=0.985, fontsize=12.6, fontweight="normal", color=INK)
    figure.subplots_adjust(left=0.09, right=0.975, bottom=0.105, top=0.90, wspace=0.32, hspace=0.42)
    return figure, axes


def text_panel(axis: plt.Axes, panel: str, text: str) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    style_axis(axis, panel)
    axis.text(0.08, 0.86, text, transform=axis.transAxes, va="top", ha="left", fontsize=9.2, linespacing=1.4)


def render_figure08(data: Path, run_root: Path) -> dict:
    levels = pd.read_parquet(data / "figure_8_deep_cover_levels.parquet")
    matched = pd.read_parquet(data / "figure_8_matched_levels.parquet")
    schedule = pd.read_parquet(data / "figure_10_vanishing_schedule.parquet")
    holdout = schedule[schedule["split"] == "holdout"]
    figure, axes = make_deep_figure(8, (2, 2), (8.0, 6.0))
    for index, (tower, group) in enumerate(levels.groupby("tower_id")):
        ordered = group.sort_values("level")
        axes[0, 0].plot(ordered["level"], ordered["injectivity_radius_lower"], color=PALETTE[index], marker="*", label=tower_label(tower))
        axes[0, 1].plot(ordered["injectivity_radius_lower"], ordered["quotient_order_digits"], color=PALETTE[index], marker="*", label=tower_label(tower))
    axes[0, 0].set(xlabel=r"Congruence level, $n$", ylabel=r"Certified lower bound, $r_{\mathrm{inj}}$", title="Certified Injectivity-Radius Growth")
    axes[0, 0].legend(loc="best")
    axes[0, 1].set(xlabel=r"Certified lower bound, $r_{\mathrm{inj}}$", ylabel=r"Decimal digits of $|Q_n|$", title="Exact Quotient-Order Growth")
    axes[0, 1].legend(loc="best")
    labels = [tower_label(value) for value in holdout["tower_id"]]
    x = np.arange(len(labels))
    axes[1, 0].bar(x - 0.18, holdout["kappa_N"], 0.36, label=r"$\kappa_N$", color=PALETTE[0])
    axes[1, 0].bar(x + 0.18, holdout["eta_N"], 0.36, label=r"$\eta_N=\sqrt{\kappa_N}$", color=PALETTE[1])
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(ylabel=r"Error or broadening scale", title="Weak Spectral-Measure Holdout")
    axes[1, 0].legend(loc="best")
    mismatch = matched.groupby("threshold", as_index=False)["relative_radius_mismatch"].max()
    axes[1, 1].plot(mismatch["threshold"], mismatch["relative_radius_mismatch"], color=PALETTE[3], marker="*")
    axes[1, 1].set(xlabel=r"Matched $r_{\mathrm{inj}}$ target", ylabel="Maximum relative mismatch", title="Matched Cross-Tower Design")
    axes[1, 1].text(0.05, 0.92, "PASS_WEAK_BULK\nStrong edge/gap no-pollution:\nINCONCLUSIVE", transform=axes[1, 1].transAxes, va="top", fontsize=8.5, color=MUTED_INK)
    for panel, axis in zip("abcd", axes.flat):
        style_axis(axis, panel)
    return save_triplet(figure, run_root, OUTPUT_STEMS[8], TITLE_MAP[8])


def render_figure09(data: Path, run_root: Path) -> dict:
    frame = pd.read_parquet(data / "figure_9_balanced_diagonal.parquet")
    figure, axes = make_deep_figure(9, (2, 3), (10.3, 6.1))
    for index, (tower, group) in enumerate(frame.groupby("tower_id")):
        ordered = group.sort_values("L_j")
        label = tower_label(tower)
        axes[0, 0].plot(ordered["L_j"], ordered["injectivity_radius_lower"], color=PALETTE[index], marker="*", label=label)
        axes[0, 1].plot(ordered["L_j"], ordered["epsilon_total_C0"], color=PALETTE[index], marker="*")
        axes[0, 2].plot(ordered["L_j"], ordered["epsilon_total_C1"], color=PALETTE[index], marker="*")
        axes[1, 0].plot(ordered["L_j"], ordered["epsilon_total_C2"], color=PALETTE[index], marker="*")
        axes[1, 1].plot(ordered["L_j"], ordered["L_over_rinj"], color=PALETTE[index], marker="*")
    axes[0, 0].set(xlabel=r"Shell scale, $L_j$", ylabel=r"$r_{\mathrm{inj},j}$", title="Certified Balanced Diagonal")
    axes[0, 0].legend(loc="best")
    axes[0, 1].set(xlabel=r"$L_j$", ylabel=r"$\varepsilon_{C^0}$", title=r"$C^0$ Error Bound")
    axes[0, 2].set(xlabel=r"$L_j$", ylabel=r"$\varepsilon_{C^1}$", title=r"$C^1$ Error Bound")
    axes[1, 0].set(xlabel=r"$L_j$", ylabel=r"$\varepsilon_{C^2}$", title=r"$C^2$ Error Bound")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set(xlabel=r"$L_j$", ylabel=r"$L_j/r_{\mathrm{inj},j}$", title="Scale Separation")
    text_panel(
        axes[1, 2],
        "f",
        "PASS_CERTIFIED\n$L_j=1,2,3,4,5$\n$L_j\\to\\infty$ and $L_j/r_{\\mathrm{inj},j}\\to0$\n\nPhysical/Hodge tail closure:\nINCONCLUSIVE",
    )
    for panel, axis in zip("abcde", axes.flat[:5]):
        style_axis(axis, panel)
    return save_triplet(figure, run_root, OUTPUT_STEMS[9], TITLE_MAP[9])


def render_figure10(data: Path, run_root: Path) -> dict:
    methods = pd.read_parquet(data / "figure_10_KPM_SLQ_CDF.parquet")
    schedule = pd.read_parquet(data / "figure_10_vanishing_schedule.parquet")
    density = pd.read_parquet(data / "figure_10_holdout_coherence_density.parquet")
    holdout = schedule[schedule["split"] == "holdout"]
    figure, axes = make_deep_figure(10, (2, 2), (8.0, 6.1))
    for index, row in enumerate(holdout.itertuples()):
        subset = methods[(methods["tower_id"] == row.tower_id) & (methods["level"] == row.level)]
        label = tower_label(row.tower_id)
        axes[0, 0].plot(subset["energy"], subset["KPM_CDF"], color=PALETTE[index], label=f"{label} KPM")
        axes[0, 0].plot(subset["energy"], subset["SLQ_CDF"], color=PALETTE[index], linestyle="--", linewidth=0.78, label=f"{label} SLQ")
    axes[0, 0].set(xlabel=r"Energy, $E$", ylabel=r"Cumulative measure, $F(E)$", title="Independent KPM-SLQ Cumulative Measures")
    axes[0, 0].legend(loc="best", ncol=1)
    x = np.arange(len(holdout))
    labels = [tower_label(value) for value in holdout["tower_id"]]
    axes[0, 1].bar(x - 0.20, holdout["kappa_N"], 0.40, color=PALETTE[0], label=r"$\kappa_N$")
    axes[0, 1].bar(x + 0.20, holdout["eta_N"], 0.40, color=PALETTE[1], label=r"$\eta_N=\sqrt{\kappa_N}$")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(ylabel="Scale", title="Frozen Vanishing-Broadening Schedule")
    axes[0, 1].legend(loc="best")
    for index, (tower, group) in enumerate(density.groupby("tower_id")):
        axes[1, 0].plot(group["energy"], group["coherence_weighted_density"], color=PALETTE[index], label=tower_label(tower))
    axes[1, 0].set(xlabel=r"Energy, $E$", ylabel=r"$\rho_{\mathrm{coh},N,\eta}(E)$", title="Coherence-Weighted Density")
    axes[1, 0].legend(loc="best")
    axes[1, 1].bar(x, holdout["KPM_SLQ_disagreement"], color=PALETTE[3])
    axes[1, 1].axhline(0.08, color=INK, linestyle="--", linewidth=0.78, label="frozen gate")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set(ylabel=r"$\|F_{\mathrm{KPM}}-F_{\mathrm{SLQ}}\|_{\infty}$", title="Method-Disagreement Bound")
    axes[1, 1].legend(loc="best")
    axes[1, 1].text(0.05, 0.91, "Weak CDF and schedule: PASS\nLocal unsmoothed/coherence DOS:\nINCONCLUSIVE", transform=axes[1, 1].transAxes, va="top", fontsize=8.3, color=MUTED_INK)
    for panel, axis in zip("abcd", axes.flat):
        style_axis(axis, panel)
    return save_triplet(figure, run_root, OUTPUT_STEMS[10], TITLE_MAP[10])


def render_figure16(data: Path, run_root: Path) -> dict:
    summary = pd.read_parquet(data / "figure_16_channel_summary.parquet")
    shape = pd.read_parquet(data / "figure_16_old_radius_shape_field.parquet")
    holdout = pd.read_parquet(data / "figure_16_hypothesis_holdout.parquet")
    figure, axes = make_deep_figure(16, (2, 2), (8.2, 6.2))
    channels = list(summary["channel"].unique())
    hypotheses = ["H0", "H2", "H3", "H4"]
    x = np.arange(len(channels))
    for index, hypothesis in enumerate(hypotheses):
        values = [float(summary[(summary["channel"] == channel) & (summary["hypothesis"] == hypothesis)]["max_C0"].iloc[0]) for channel in channels]
        axes[0, 0].bar(x + (index - 1.5) * 0.18, values, 0.18, label=rf"$H_{{{hypothesis[1:]}}}$", color=PALETTE[index])
    axes[0, 0].set_xticks(x, channels)
    axes[0, 0].set_yscale("symlog", linthresh=0.1)
    axes[0, 0].set(ylabel=r"Maximum $C^0$ residual", title="Holdout Operator Residuals")
    axes[0, 0].legend(loc="best", ncol=2)
    axes[0, 1].plot(shape["radius"], shape["baseline_C0"], marker="*", color=PALETTE[1], label=r"$X$ only")
    axes[0, 1].plot(shape["radius"], shape["shape_corrected_C0"], marker="*", color=PALETTE[2], label=r"$\lambda_{\perp}/a$ shape field")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(xlabel=r"Fixed-octagon radius, $R$", ylabel=r"$C^0$ residual", title="Profile-Coordinate Correction")
    axes[0, 1].legend(loc="best")
    combined = holdout[holdout["channel"] == "combined"]
    z = np.arange(len(combined))
    for index, hypothesis in enumerate(("H0", "H3", "H4")):
        axes[1, 0].plot(z, combined[f"{hypothesis}_C0"], marker="*", color=PALETTE[index], label=rf"$H_{{{hypothesis[1:]}}}$")
    axes[1, 0].axhline(0.20, color=INK, linestyle="--", linewidth=0.78, label="frozen gate")
    axes[1, 0].set_xticks(z, combined["case_id"], rotation=18, ha="right")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(ylabel=r"$C^0$ residual", title="Independent Joint Holdout")
    axes[1, 0].legend(loc="best", ncol=2)
    text_panel(
        axes[1, 1],
        "d",
        "PASS_RESTRICTED_CLASS\n\nLocal fixed-$a$ operator tangent: rank 2\nScalar identifiability: unstable\n$H_3$ two-parameter holdout: fail\n$H_4$ three-field holdout: fail\n\nNo FAIL_THEORY",
    )
    for panel, axis in zip("abc", axes.flat[:3]):
        style_axis(axis, panel)
    return save_triplet(figure, run_root, OUTPUT_STEMS[16], TITLE_MAP[16])


def render_figure18(data: Path, run_root: Path) -> dict:
    singular = pd.read_parquet(data / "figure_18_singular_values.parquet")
    summary = pd.read_parquet(data / "figure_18_hypothesis_summary.parquet")
    curvature = pd.read_parquet(data / "figure_18_curvature_scaling.parquet")
    blocks = pd.read_parquet(data / "r16_block_rank.parquet")
    figure, axes = make_deep_figure(18, (2, 2), (8.0, 6.1))
    for index, (family, group) in enumerate(singular.groupby("family")):
        axes[0, 0].plot(group["index"], group["singular_value"], marker="*", color=PALETTE[index], label=str(family).replace("_", " "))
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="Singular-value index", ylabel="Singular value", title="Operator and Observable Rank")
    axes[0, 0].legend(loc="best")
    axes[0, 1].plot(blocks["block"], blocks["PhiX_PhiG_s2_over_s1"], marker="*", color=PALETTE[2])
    axes[0, 1].axhline(0.05, color=INK, linestyle="--", linewidth=0.78, label="frozen rank gate")
    axes[0, 1].set(xlabel="Momentum block", ylabel=r"$s_2/s_1$", title="Local Rank Stability")
    axes[0, 1].legend(loc="best")
    axes[1, 0].loglog(curvature["coordinate"], curvature["actual_C0"], marker="*", linestyle="None", color=PALETTE[1], label="holdout")
    axes[1, 0].loglog(curvature["coordinate"], curvature["predicted_C0"], linestyle="-.", color=PALETTE[0], label="frozen training fit")
    axes[1, 0].set(xlabel=r"$|g-g_0|$", ylabel=r"$X$-only $C^0$ residual", title="Curvature-Scaling Falsification")
    axes[1, 0].legend(loc="best")
    axes[1, 1].bar(summary["hypothesis"], summary["median_C0"], color=PALETTE[: len(summary)])
    axes[1, 1].axhline(0.20, color=INK, linestyle="--", linewidth=0.78, label="frozen gate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(ylabel=r"Median $C^0$ residual", title="Median Joint-Holdout Error")
    axes[1, 1].legend(loc="best")
    axes[1, 1].text(0.05, 0.91, "Local operator evidence only\nGlobal curvature relevance:\nINCONCLUSIVE", transform=axes[1, 1].transAxes, va="top", fontsize=8.3, color=MUTED_INK)
    for panel, axis in zip("abcd", axes.flat):
        style_axis(axis, panel)
    return save_triplet(figure, run_root, OUTPUT_STEMS[18], TITLE_MAP[18])


def verify_registry_sources(validation_root: Path, registry: dict) -> None:
    for figure in registry["selected_figures"]:
        for source in figure["source_data"]:
            path = validation_root / source["path"]
            if source["kind"] == "file" and sha256_file(path) != source["sha256"]:
                raise RuntimeError(f"Frozen source changed before rendering: {path}")


def publish_outputs(run_root: Path, output_root: Path, records: dict[int, dict]) -> None:
    for number in sorted(records):
        for kind in ("png", "svg", "pdf"):
            source = Path(records[number][kind]["path"])
            destination = output_root / f"rendered_{kind}" / source.name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(source, destination)
            if sha256_file(destination) != records[number][kind]["sha256"]:
                raise RuntimeError(f"Published copy hash mismatch: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--beautified-root", type=Path, required=True)
    args = parser.parse_args()
    validation_root = args.validation_root.resolve()
    output_root = args.beautified_root.resolve()
    pre = json.loads((output_root / "manifests/pre_render_manifest.json").read_text(encoding="utf-8"))
    if pre["state"] != "PRE_RENDER_REGISTRY_AND_STYLE_FROZEN":
        raise RuntimeError("The pre-render contract is not frozen")
    registry_path = output_root / "source_registry/valid_figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if sha256_file(registry_path) != pre["source_registry_sha256"]:
        raise RuntimeError("Source registry changed after pre-registration")
    verify_registry_sources(validation_root, registry)
    configure_style()
    run_id = pre["beautified_run_id"]
    run_root = output_root / "results" / run_id
    records: dict[int, dict] = {}
    render_original_figures(validation_root, run_root, records)
    data = validation_root / "deep_resolution/figure_data"
    records[8] = render_figure08(data, run_root)
    records[9] = render_figure09(data, run_root)
    records[10] = render_figure10(data, run_root)
    records[16] = render_figure16(data, run_root)
    records[18] = render_figure18(data, run_root)
    if sorted(records) != list(range(1, 19)):
        raise RuntimeError(f"Expected figures 1-18, got {sorted(records)}")
    publish_outputs(run_root, output_root, records)
    manifest = {
        "schema_version": 1,
        "beautified_run_id": run_id,
        "state": "RENDER_COMPLETE_PENDING_QA",
        "scientific_results_computed": False,
        "scientific_data_modified": False,
        "figure_count": len(records),
        "font_resolution": font_manager.findfont("Times New Roman", fallback_to_default=False),
        "math_engine": "matplotlib mathtext with Times New Roman custom font set",
        "figures": {
            str(number): {
                "title": TITLE_MAP[number],
                "outputs": records[number],
                "source_data": next(item["source_data"] for item in registry["selected_figures"] if item["figure_number"] == number),
            }
            for number in sorted(records)
        },
    }
    manifest_path = run_root / "manifests/render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copy2(manifest_path, output_root / "manifests/render_manifest.json")
    print(json.dumps({"run_id": run_id, "figure_count": len(records), "state": manifest["state"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
