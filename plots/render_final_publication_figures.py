from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


WIDTH, HEIGHT = 1100, 680
NAVY, BLUE, TEAL, ORANGE, RED, PURPLE, GRID = (
    "#183153", "#2F6BFF", "#0F8B8D", "#F39C12", "#C44536", "#7D5FFF", "#D7E0E8"
)


def _scale(values, low, high, reverse=False):
    data = np.asarray(values, dtype=float)
    minimum, maximum = float(np.nanmin(data)), float(np.nanmax(data))
    if math.isclose(minimum, maximum):
        minimum -= 0.5
        maximum += 0.5
    normalized = (data - minimum) / (maximum - minimum)
    if reverse:
        normalized = 1.0 - normalized
    return low + normalized * (high - low)


def _header(title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="70" y="55" font-family="Arial" font-size="25" font-weight="700" fill="{NAVY}">{html.escape(title)}</text>',
        f'<text x="70" y="82" font-family="Arial" font-size="13" fill="#4B6175">{html.escape(subtitle)}</text>',
    ]


def _axes(parts, x0, y0, x1, y1, xlabel, ylabel):
    parts.extend(
        [
            f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{NAVY}" stroke-width="1.5"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{NAVY}" stroke-width="1.5"/>',
            f'<text x="{(x0+x1)/2}" y="{y1+48}" text-anchor="middle" font-family="Arial" font-size="14" fill="{NAVY}">{html.escape(xlabel)}</text>',
            f'<text x="{x0-52}" y="{(y0+y1)/2}" text-anchor="middle" transform="rotate(-90 {x0-52} {(y0+y1)/2})" font-family="Arial" font-size="14" fill="{NAVY}">{html.escape(ylabel)}</text>',
        ]
    )
    for fraction in np.linspace(0, 1, 6):
        y = y0 + fraction * (y1 - y0)
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="{GRID}" stroke-width="0.7"/>')


def _polyline(parts, xs, ys, color, width=2.5, dash=None):
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>')


def magic_complexity(run_dir: Path, output: Path):
    frame = pd.read_parquet(run_dir / "derived" / "magic_complexity.parquet").sort_values("j")
    parts = _header("Exact sampled magic complexity", "Saved G-13/G-14 data; the horizontal dashed line is the frozen coefficient 4c_nu")
    _axes(parts, 100, 120, 1020, 590, "1/sqrt(omega_j)", "sqrt(omega_j) log q_j")
    xs = _scale(1.0 / np.sqrt(frame.omega_j), 115, 1005)
    all_y = np.concatenate([frame.sqrt_omega_log_q.to_numpy(), frame.target_4c.to_numpy()])
    ys = _scale(frame.sqrt_omega_log_q, 560, 145)
    target_y = float(_scale([all_y.min(), float(frame.target_4c.iloc[0]), all_y.max()], 560, 145)[1])
    _polyline(parts, xs, ys, BLUE)
    _polyline(parts, [115, 1005], [target_y, target_y], RED, dash="8 6")
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def joint_limit(run_dir: Path, output: Path):
    frame = pd.read_parquet(run_dir / "derived" / "incommensurate_joint_limit.parquet")
    parts = _header("Joint incommensurate-limit falsification", "Certified non-Abelian towers only; passing and deliberately failing residuals are both preserved")
    _axes(parts, 100, 120, 1020, 590, "certified L/R lower bound", "|delta theta| exp(L/R) (log scale)")
    colors = {"passing": TEAL, "deliberately_failing": RED}
    for sequence, group in frame.groupby("sequence"):
        ordered = group.sort_values(["tower_id", "level"])
        xs = _scale(ordered.L_over_R, 115, 1005)
        ys = _scale(np.log10(ordered.joint_limit_residual), 560, 145)
        _polyline(parts, xs, ys, colors[str(sequence)], width=2.2)
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def operational_spectrum(run_dir: Path, output: Path):
    group = zarr.open_group(str(run_dir / "raw" / "operational_magic.zarr"), mode="r")
    q = np.asarray(group["q"][:], dtype=float)
    values = np.asarray(group["complete_eigenvalues"][:], dtype=float)
    target = np.asarray(group["target_energy"][:], dtype=float)
    parts = _header("Operational magic target spectrum", "One transported target supplies W, Delta, Omega_max, rho_coh, and C_coh")
    _axes(parts, 100, 120, 1020, 590, "transported momentum q", "energy / t")
    xs = _scale(q, 115, 1005)
    all_values = np.concatenate([values.ravel(), target])
    low, high = float(all_values.min()), float(all_values.max())
    def ymap(data):
        return 560 - (np.asarray(data, dtype=float) - low) / max(high - low, 1.0e-12) * 415
    for band in range(values.shape[1]):
        _polyline(parts, xs, ymap(values[:, band]), GRID, width=1.5)
    _polyline(parts, xs, ymap(target), BLUE, width=3.2)
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def master_collapse(run_dir: Path, output: Path):
    group = zarr.open_group(str(run_dir / "raw" / "master_curve_complete_spectra.zarr"), mode="r")
    labels = list(group.attrs["labels"])
    q = np.asarray(group["q"][:], dtype=float)
    target = np.asarray(group["normalized_target_bands"][:], dtype=float)
    indices = [index for index, label in enumerate(labels) if label.endswith("X=1.00")]
    parts = _header("Complete-spectrum master-collapse audit", "Saved normalized target bands at matched X=1; complete three-band spectra remain in the Zarr dataset")
    _axes(parts, 100, 120, 1020, 590, "transported momentum q", "normalized target energy")
    xs = _scale(q, 115, 1005)
    selected = target[indices]
    low, high = float(selected.min()), float(selected.max())
    palette = [BLUE, TEAL, ORANGE, RED, PURPLE]
    for color, index in zip(palette, indices):
        ys = 560 - (target[index] - low) / max(high - low, 1.0e-12) * 415
        _polyline(parts, xs, ys, color, width=2.0)
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def landscape(run_dir: Path, output: Path):
    group = zarr.open_group(str(run_dir / "raw" / "magic_landscape.zarr"), mode="r")
    k_values = np.asarray(group["K"][:], dtype=float)
    theta = np.asarray(group["theta"][:], dtype=float)
    w_values = np.asarray(group["w_over_t"][:], dtype=float)
    score = np.asarray(group["score_M"][:], dtype=float)
    index = int(np.argmin(np.abs(k_values + 0.015625)))
    plane = score[index]
    parts = _header("Operational magic landscape", f"Saved M(theta,K,w/t) slice at K={k_values[index]:.6f}; no values are calculated in this renderer")
    x0, y0, x1, y1 = 125, 125, 1015, 580
    minimum, maximum = float(plane.min()), float(plane.max())
    for it in range(len(theta)):
        for iw in range(len(w_values)):
            fraction = (float(plane[it, iw]) - minimum) / max(maximum - minimum, 1.0e-12)
            red = int(245 - 150 * fraction)
            green = int(248 - 80 * fraction)
            blue = int(255 - 5 * fraction)
            xx = x0 + iw * (x1 - x0) / len(w_values)
            yy = y1 - (it + 1) * (y1 - y0) / len(theta)
            width = (x1 - x0) / len(w_values) + 0.5
            height = (y1 - y0) / len(theta) + 0.5
            parts.append(f'<rect x="{xx:.2f}" y="{yy:.2f}" width="{width:.2f}" height="{height:.2f}" fill="rgb({red},{green},{blue})"/>')
    _axes(parts, x0, y0, x1, y1, "w/t", "theta")
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    builders = {
        "figure13_magic_complexity.svg": magic_complexity,
        "figure14_joint_limit.svg": joint_limit,
        "figure15_operational_spectrum.svg": operational_spectrum,
        "figure16_master_collapse.svg": master_collapse,
        "figure17_magic_landscape.svg": landscape,
    }
    for filename, builder in builders.items():
        builder(args.run_dir, args.output / filename)
    (args.output / "render_manifest.json").write_text(
        json.dumps({"renderer": __file__, "figures": list(builders), "scientific_calculation_in_renderer": False}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
