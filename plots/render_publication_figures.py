from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd


WIDTH, HEIGHT = 1100, 680
NAVY, BLUE, TEAL, ORANGE, RED, GRID = "#183153", "#2F6BFF", "#0F8B8D", "#F39C12", "#C44536", "#D7E0E8"


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
    parts.extend([
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{NAVY}" stroke-width="1.5"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{NAVY}" stroke-width="1.5"/>',
        f'<text x="{(x0+x1)/2}" y="{y1+48}" text-anchor="middle" font-family="Arial" font-size="14" fill="{NAVY}">{html.escape(xlabel)}</text>',
        f'<text x="{x0-52}" y="{(y0+y1)/2}" text-anchor="middle" transform="rotate(-90 {x0-52} {(y0+y1)/2})" font-family="Arial" font-size="14" fill="{NAVY}">{html.escape(ylabel)}</text>',
    ])
    for fraction in np.linspace(0, 1, 6):
        y = y0 + fraction * (y1 - y0)
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="{GRID}" stroke-width="0.7"/>')


def _polyline(parts, xs, ys, color, width=2.5, dash=None):
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>' )


def _legend(parts, entries, x=760, y=104):
    for index, (label, color, dash) in enumerate(entries):
        yy = y + 23 * index
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x}" y1="{yy}" x2="{x+32}" y2="{yy}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        parts.append(f'<text x="{x+42}" y="{yy+5}" font-family="Arial" font-size="12" fill="{NAVY}">{html.escape(label)}</text>')


def figure10(data_dir: Path, output: Path):
    cdf = pd.read_parquet(data_dir / "figure10_cdf_convergence.parquet").sort_values("quotient_order")
    dos = pd.read_parquet(data_dir / "figure10_coherence_weighted_dos.parquet")
    parts = _header("Figure 10 — certified bulk spectral diagnostics", "Kernel-independent retained-sector CDF errors and the saved fixed-broadening coherence-weighted measure")
    _axes(parts, 90, 120, 510, 590, "quotient order (log scale)", "CDF error κ_N")
    xs = _scale(np.log10(cdf.quotient_order), 100, 500)
    ys = _scale(cdf.kappa_N, 560, 145, reverse=False)
    _polyline(parts, xs, ys, BLUE)
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{BLUE}"/>')
    _axes(parts, 620, 120, 1040, 590, "energy", "coherence-weighted density")
    xs2 = _scale(dos.energy, 630, 1030)
    ys2 = _scale(dos.layer_even_coherence_weighted_density, 560, 145)
    _polyline(parts, xs2, ys2, TEAL)
    _legend(parts, [("retained-sector CDF", BLUE, None), ("coherence-weighted DOS", TEAL, None)], x=760, y=105)
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def figure11(data_dir: Path, output: Path):
    pairs = pd.read_parquet(data_dir / "figure11_scale_separated_exponents.parquet").sort_values("block_start")
    env = pd.read_parquet(data_dir / "figure11_dyadic_envelopes.parquet").sort_values("block_start")
    parts = _header("Figure 11 — theorem-matched logarithmic exponent", "Dyadic pointwise log-exponent envelopes; the separate c=2 regular-variation diagnostic is excluded from theorem-A acceptance")
    _axes(parts, 100, 120, 1020, 590, "dyadic block start J", "effective exponent β")
    all_x = np.log2(np.concatenate([pairs.block_start.to_numpy(), env.block_start.to_numpy()]))
    xmin, xmax = float(all_x.min()), float(all_x.max())
    def xmap(values):
        return 115 + (np.log2(np.asarray(values, dtype=float)) - xmin) / (xmax - xmin) * 890
    all_y = np.concatenate([pairs.median_beta.to_numpy(), env.upper_pointwise_exponent.to_numpy(), env.lower_pointwise_exponent.to_numpy(), np.asarray([4.0, 4.061663815456154])])
    ymin, ymax = float(all_y.min()) - 0.2, float(all_y.max()) + 0.2
    def ymap(values):
        return 565 - (np.asarray(values, dtype=float) - ymin) / (ymax - ymin) * 420
    _polyline(parts, xmap(pairs.block_start), ymap(pairs.median_beta), BLUE)
    _polyline(parts, xmap(env.block_start), ymap(env.upper_pointwise_exponent), ORANGE)
    _polyline(parts, xmap(env.block_start), ymap(env.lower_pointwise_exponent), TEAL)
    _polyline(parts, [115, 1005], ymap([4.0, 4.0]), NAVY, dash="8 6")
    _polyline(parts, [115, 1005], ymap([4.061663815456154, 4.061663815456154]), RED, dash="3 5")
    _legend(parts, [("median log exponent", BLUE, None), ("upper envelope", ORANGE, None), ("lower envelope", TEAL, None), ("theorem β=4", NAVY, "8 6"), ("G-11: 4.06166", RED, "3 5")], x=750, y=105)
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def figure12(data_dir: Path, output: Path):
    dos = pd.read_parquet(data_dir / "figure12_public_hyperbloch_dos.parquet")
    circuit = pd.read_parquet(data_dir / "figure12_circuit_reconstruction.parquet")
    parts = _header("Figure 12 — external graph and circuit reproduction", "Public HyperBloch graph spectra and saved circuit-Laplacian reconstruction residuals")
    _axes(parts, 90, 120, 570, 590, "energy", "broadened public DOS")
    colors = [BLUE, TEAL, ORANGE, RED]
    legend = []
    density_min = float(dos.density.min())
    density_max = float(dos.density.max())
    density_span = density_max - density_min if density_max > density_min else 1.0
    for index, (name, subset) in enumerate(dos.groupby("benchmark")):
        xs = _scale(subset.energy, 100, 560)
        ys = 560.0 - (np.asarray(subset.density, dtype=float) - density_min) / density_span * 415.0
        color = colors[index % len(colors)]
        _polyline(parts, xs, ys, color, width=2.0)
        legend.append((str(name), color, None))
    _axes(parts, 650, 120, 1040, 590, "eigenvalue index", "absolute reconstruction residual")
    xs2 = _scale(circuit.eigen_index, 660, 1030)
    residual = np.maximum(np.asarray(circuit.absolute_residual, dtype=float), 1.0e-18)
    ys2 = _scale(np.log10(residual), 560, 145)
    _polyline(parts, xs2, ys2, RED, width=2.0)
    legend.append(("circuit residual", RED, None))
    _legend(parts, legend, x=760, y=105)
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("figure_data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figure10(args.figure_data, args.output / "figure10_bulk_dos.svg")
    figure11(args.figure_data, args.output / "figure11_arithmetic_exponent.svg")
    figure12(args.figure_data, args.output / "figure12_external_reproduction.svg")


if __name__ == "__main__":
    main()
