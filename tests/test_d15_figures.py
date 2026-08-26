from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plots.render_publication_figures import figure10, figure11, figure12


def test_saved_data_only_svg_renderer(tmp_path: Path):
    data = tmp_path / "data"
    output = tmp_path / "figures"
    data.mkdir()
    output.mkdir()
    pd.DataFrame({
        "tower_id": ["a", "a", "b"], "level": [1, 2, 1],
        "quotient_order": [100, 200, 400], "retained_dimension": [40, 90, 170],
        "kappa_N": [0.08, 0.05, 0.04], "kernel": ["NONE"] * 3, "reference": ["r"] * 3,
    }).to_parquet(data / "figure10_cdf_convergence.parquet", index=False)
    energy = np.linspace(-2, 2, 20)
    pd.DataFrame({
        "energy": energy, "layer_even_coherence_weighted_density": np.exp(-energy * energy),
        "broadening": 0.08, "tower_id": "a", "level": 1,
    }).to_parquet(data / "figure10_coherence_weighted_dos.parquet", index=False)
    pd.DataFrame({
        "block_start": [64, 128, 256, 512, 1024], "median_beta": [4.3, 4.2, 4.1, 4.08, 4.05],
    }).to_parquet(data / "figure11_scale_separated_exponents.parquet", index=False)
    pd.DataFrame({
        "block_start": [64, 128, 256, 512, 1024, 2048],
        "upper_pointwise_exponent": [4.8, 4.6, 4.5, 4.4, 4.3, 4.2],
        "lower_pointwise_exponent": [3.2, 3.4, 3.5, 3.6, 3.7, 3.8],
    }).to_parquet(data / "figure11_dyadic_envelopes.parquet", index=False)
    rows = []
    for benchmark, shift in (("a", 0.0), ("b", 0.4)):
        for value in energy:
            rows.append({"benchmark": benchmark, "energy": value, "density": float(np.exp(-(value - shift) ** 2)), "broadening": 0.08})
    pd.DataFrame(rows).to_parquet(data / "figure12_public_hyperbloch_dos.parquet", index=False)
    pd.DataFrame({
        "eigen_index": np.arange(20), "hamiltonian_eigenvalue": energy,
        "recovered_circuit_eigenvalue": energy, "absolute_residual": np.full(20, 1.0e-14),
    }).to_parquet(data / "figure12_circuit_reconstruction.parquet", index=False)
    figure10(data, output / "figure10_bulk_dos.svg")
    figure11(data, output / "figure11_arithmetic_exponent.svg")
    figure12(data, output / "figure12_external_reproduction.svg")
    for path in output.glob("*.svg"):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert text.endswith("</svg>")
        assert path.stat().st_size > 1000
    assert len(list(output.glob("*.svg"))) == 3
