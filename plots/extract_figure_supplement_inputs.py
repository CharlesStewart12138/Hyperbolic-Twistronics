from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


ZARR_SOURCES = {
    "figure01_moire_length": (
        "data/figure_sources/runs/5f785708371af9f1335a4e084146aa48e37b495c8f88737b551055c5cd8b3de3/raw/moire_length.zarr",
        ["R", "threshold", "theta", "xi_numeric", "xi_analytic", "absolute_error", "registry_residual"],
    ),
    "figure02_crossover": (
        "data/figure_sources/runs/5f785708371af9f1335a4e084146aa48e37b495c8f88737b551055c5cd8b3de3/raw/crossover.zarr",
        ["R", "theta", "chi", "normalized_area", "collapse_residual", "beta_exact", "beta_numeric"],
    ),
    "figure02_double_scaling": (
        "data/figure_sources/runs/5f785708371af9f1335a4e084146aa48e37b495c8f88737b551055c5cd8b3de3/raw/double_scaling.zarr",
        ["u", "alpha", "normalized_length_G", "limit_G", "absolute_error"],
    ),
    "figure02_dimensional_extension": (
        "data/figure_sources/runs/5f785708371af9f1335a4e084146aa48e37b495c8f88737b551055c5cd8b3de3/raw/dimensional_extension.zarr",
        ["ambient_dimension", "active_dimension", "fixed_axis_count", "full_D_law_admissible", "y", "beta"],
    ),
    "figure06_hodge_tensor": (
        "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/raw/hodge_tensor.zarr",
        ["w", "K", "C_S", "B_infinity_mid"],
    ),
    "figure15_operational_magic": (
        "data/figure_sources/runs/b8722dc68f6c7ed04cc5d48023a0d7f52f1fe92153bb362cc57dc7144cdf17ad/raw/operational_magic.zarr",
        ["q", "complete_eigenvalues", "target_energy", "target_coherence", "external_gap"],
    ),
    "figure17_magic_landscape": (
        "data/figure_sources/runs/b8722dc68f6c7ed04cc5d48023a0d7f52f1fe92153bb362cc57dc7144cdf17ad/raw/magic_landscape.zarr",
        ["K", "theta", "w_over_t", "score_M"],
    ),
}

PARQUET_SOURCES = {
    "figure01_errors": "data/figure_sources/runs/5f785708371af9f1335a4e084146aa48e37b495c8f88737b551055c5cd8b3de3/derived/moire_length_errors.parquet",
    "figure03_exponent": "data/figure_sources/runs/5f785708371af9f1335a4e084146aa48e37b495c8f88737b551055c5cd8b3de3/derived/arithmetic_exponent.parquet",
    "figure03_locking": "data/figure_sources/runs/5f785708371af9f1335a4e084146aa48e37b495c8f88737b551055c5cd8b3de3/derived/radial_locking_extrapolation.parquet",
    "figure04_square": "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/raw/square_fivestate_exact.parquet",
    "figure04_root_scan": "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/raw/character_root.parquet",
    "figure04_root_summary": "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/derived/character_root_summary.parquet",
    "figure05_shells": "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/derived/full_kernel_shells.parquet",
    "figure05_displacement": "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/raw/root_displacement.parquet",
    "figure07_symmetry": "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/raw/symmetry_breaking_scan.parquet",
    "figure07_robustness": "data/figure_sources/runs/b517afe774be43a1b885808807771069f4670585485bc48bbde3fb4c6f88e619/raw/multiorbital_robustness.parquet",
    "figure07_comparison": "data/figure_sources/runs/b8722dc68f6c7ed04cc5d48023a0d7f52f1fe92153bb362cc57dc7144cdf17ad/derived/symmetry_vs_flatness.parquet",
    "figure11_dos": "data/figure_sources/runs/ed350b4ade947be9a310532f8fd8e643030e35205b74ea14bf01ca4c48844ec4/derived/d15_publication_outputs/figure_ready_data/figure12_public_hyperbloch_dos.parquet",
    "figure11_graph": "data/figure_sources/runs/ed350b4ade947be9a310532f8fd8e643030e35205b74ea14bf01ca4c48844ec4/derived/d10_reproduce_public_graphs.parquet",
    "figure11_circuit": "data/figure_sources/runs/ed350b4ade947be9a310532f8fd8e643030e35205b74ea14bf01ca4c48844ec4/derived/d15_publication_outputs/figure_ready_data/figure12_circuit_reconstruction.parquet",
    "figure12_reverse": "data/figure_sources/runs/b8722dc68f6c7ed04cc5d48023a0d7f52f1fe92153bb362cc57dc7144cdf17ad/derived/reverse_falsification_residuals.parquet",
    "figure12_nc03": "data/figure_sources/runs/ed350b4ade947be9a310532f8fd8e643030e35205b74ea14bf01ca4c48844ec4/derived/nc_03.parquet",
    "figure12_nc07": "data/figure_sources/runs/ed350b4ade947be9a310532f8fd8e643030e35205b74ea14bf01ca4c48844ec4/derived/nc_07.parquet",
    "figure12_matrix": "data/figure_sources/runs/23ea5ca46de279646c795385cedbca38052478dd41f7b40d95dec44ca3c43fc2/validation_matrix.parquet",
    "figure13_complexity": "data/figure_sources/runs/b8722dc68f6c7ed04cc5d48023a0d7f52f1fe92153bb362cc57dc7144cdf17ad/derived/magic_complexity.parquet",
    "figure14_joint_limit": "data/figure_sources/runs/b8722dc68f6c7ed04cc5d48023a0d7f52f1fe92153bb362cc57dc7144cdf17ad/derived/incommensurate_joint_limit.parquet",
}


def jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, object] = {
        "operation": "lossless plotting-format extraction only",
        "scientific_results_computed": False,
        "datasets": {},
    }
    for alias, (relative, array_names) in ZARR_SOURCES.items():
        source = root / relative
        group = zarr.open_group(str(source), mode="r")
        arrays = {name: np.asarray(group[name][:]) for name in array_names}
        destination = output / f"{alias}.npz"
        np.savez_compressed(destination, **arrays)
        metadata["datasets"][alias] = {
            "source": relative,
            "arrays_copied_without_transformation": array_names,
            "source_attrs": jsonable(dict(group.attrs)),
            "shapes": {name: list(value.shape) for name, value in arrays.items()},
            "output": destination.name,
        }
    for alias, relative in PARQUET_SOURCES.items():
        source = root / relative
        frame = pd.read_parquet(source)
        destination = output / f"{alias}.csv"
        frame.to_csv(destination, index=False, float_format="%.17g")
        metadata["datasets"][alias] = {
            "source": relative,
            "columns_copied_without_transformation": list(frame.columns),
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "format_conversion": "Parquet to UTF-8 CSV with 17 significant digits",
            "output": destination.name,
        }
    (output / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

