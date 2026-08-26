from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE / "render_beautified_figures.py"
SPEC = importlib.util.spec_from_file_location("publication_figure_core", BASE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(BASE)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


renderer.PAPER = "#FFFFFF"
renderer.OUTPUT_STEMS = {
    number: stem.removesuffix("_beautified")
    for number, stem in renderer.OUTPUT_STEMS.items()
}
renderer.TEXT_REPLACEMENTS.update(
    {
        "Delta over xi": r"$\Delta/\xi$",
        "r sc over xi": r"$r_{\mathrm{sc}}/\xi$",
    }
)

base_configure_style = renderer.configure_style
base_finalize = renderer.finalize_figure
base_save_triplet = renderer.save_triplet


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_digest(path: Path) -> tuple[int, str]:
    inventory = {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(inventory), hashlib.sha256(canonical).hexdigest()


def verify_sources(project_root: Path, registry: dict) -> None:
    for figure in registry["selected_figures"]:
        for source in figure["source_data"]:
            path = project_root / source["path"]
            if source["kind"] == "file":
                if sha256_file(path) != source["sha256"]:
                    raise RuntimeError(f"Frozen source hash mismatch: {path}")
            else:
                count, digest = directory_digest(path)
                if count != source["file_count"] or digest != source["tree_inventory_sha256"]:
                    raise RuntimeError(f"Frozen source tree mismatch: {path}")


def configure_style() -> None:
    base_configure_style()
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )


def finalize_figure(figure) -> None:
    base_finalize(figure)
    for axis in figure.axes:
        legend = axis.get_legend()
        if legend is None:
            continue
        for handle in getattr(legend, "legend_handles", []):
            if isinstance(handle, mpl.lines.Line2D):
                marker = handle.get_marker()
                if marker not in (None, "None", "", " "):
                    handle.set_marker("*")
                    handle.set_markersize(5.2)
                    handle.set_markeredgewidth(0.42)


def set_white_surfaces(figure) -> None:
    figure.patch.set_facecolor("#FFFFFF")
    for axis in figure.axes:
        axis.patch.set_facecolor("#FFFFFF")
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#FFFFFF")


def remove_figure09_panel_f(figure) -> None:
    axes = list(figure.axes)
    if len(axes) != 6:
        raise RuntimeError(f"Figure 9 expected six pre-layout axes, found {len(axes)}")
    top_left = axes[0].get_position()
    top_right = axes[2].get_position()
    lower_left = axes[3].get_position()
    lower_right = axes[4].get_position()
    figure.delaxes(axes[5])
    full_left = top_left.x0
    full_right = top_right.x1
    full_width = full_right - full_left
    gap = full_width * 0.085
    width = (full_width - gap) / 2.0
    axes[3].set_position([full_left, lower_left.y0, width, lower_left.height])
    axes[4].set_position([full_left + width + gap, lower_right.y0, width, lower_right.height])


def remove_figure16_panel_d(figure) -> None:
    axes = list(figure.axes)
    if len(axes) != 4:
        raise RuntimeError(f"Figure 16 expected four pre-layout axes, found {len(axes)}")
    top_left = axes[0].get_position()
    top_right = axes[1].get_position()
    lower = axes[2].get_position()
    figure.delaxes(axes[3])
    full_left = top_left.x0
    full_right = top_right.x1
    full_width = full_right - full_left
    width = full_width * 0.64
    left = full_left + (full_width - width) / 2.0
    axes[2].set_position([left, lower.y0, width, lower.height])


def remove_figure11_annotation(figure) -> None:
    matches = [
        text
        for text in figure.findobj(mpl.text.Text)
        if text.get_text().startswith("D-10 saved spectral residuals")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Figure 11 expected one removable annotation, found {len(matches)}")
    matches[0].set_text("")


def save_triplet(figure, run_root: Path, stem: str, title: str):
    if stem == renderer.OUTPUT_STEMS[3]:
        figure.axes[1].set_xticks(np.arange(2), [r"$\Delta/\xi$", r"$r_{\mathrm{sc}}/\xi$"])
    if stem == renderer.OUTPUT_STEMS[8]:
        for text in figure.findobj(mpl.text.Text):
            if text.get_text().startswith("PASS_WEAK_BULK"):
                text.set_position((0.96, 0.92))
                text.set_ha("right")
    if stem == renderer.OUTPUT_STEMS[9]:
        remove_figure09_panel_f(figure)
    if stem == renderer.OUTPUT_STEMS[11]:
        remove_figure11_annotation(figure)
    if stem == renderer.OUTPUT_STEMS[16]:
        remove_figure16_panel_d(figure)
    if stem == renderer.OUTPUT_STEMS[18]:
        for text in list(figure.findobj(mpl.text.Text)):
            if text.get_text().startswith("Local operator evidence only"):
                text.set_text("")
        figure.text(
            0.5,
            0.935,
            "Local operator-rank evidence; global curvature relevance remains INCONCLUSIVE",
            ha="center",
            va="center",
            fontsize=8.8,
            color=renderer.MUTED_INK,
        )
        figure.subplots_adjust(top=0.865)
        figure.axes[3].set_xticks(
            np.arange(5),
            [r"$H_0$", r"$H_2$", r"$H_3$", r"$H_4$", r"wrong $K$"],
        )
    set_white_surfaces(figure)
    return base_save_triplet(figure, run_root, stem, title)


renderer.configure_style = configure_style
renderer.finalize_figure = finalize_figure
renderer.save_triplet = save_triplet


def relocate_outputs(output_root: Path, records: dict[int, dict]) -> None:
    for kind in ("png", "svg", "pdf"):
        source_dir = output_root / f"rendered_{kind}"
        target_dir = output_root / kind
        source_dir.rename(target_dir)
        for record in records.values():
            old = Path(record[kind]["path"])
            new = target_dir / old.name
            record[kind]["path"] = new.relative_to(output_root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the current publication figure release from frozen inputs.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    for kind in ("png", "svg", "pdf"):
        (output_root / f"rendered_{kind}").mkdir()

    registry_path = project_root / "manifests/figure_release/valid_figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    verify_sources(project_root, registry)
    configure_style()
    records: dict[int, dict] = {}
    renderer.render_original_figures(project_root, output_root, records)
    data = project_root / "deep_resolution/figure_data"
    records[8] = renderer.render_figure08(data, output_root)
    records[9] = renderer.render_figure09(data, output_root)
    records[10] = renderer.render_figure10(data, output_root)
    records[16] = renderer.render_figure16(data, output_root)
    records[18] = renderer.render_figure18(data, output_root)
    if sorted(records) != list(range(1, 19)):
        raise RuntimeError(f"Expected figures 1-18, got {sorted(records)}")
    relocate_outputs(output_root, records)
    manifest = {
        "schema_version": 1,
        "state": "PUBLICATION_FIGURES_RENDERED",
        "scientific_results_computed": False,
        "scientific_data_modified": False,
        "figure_count": 18,
        "source_registry": registry_path.relative_to(project_root).as_posix(),
        "figures": {str(number): records[number] for number in sorted(records)},
    }
    (output_root / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"state": manifest["state"], "figure_count": 18}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
