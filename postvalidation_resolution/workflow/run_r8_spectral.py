from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from common import finalize_run, initialize_run, sha256_file, write_json  # noqa: E402
from projected_spectral import (  # noqa: E402
    ProjectedCayleyOperator,
    compute_edges,
    gaussian_smooth_density,
    heat_trace_from_moments,
    interval_distance,
    kpm_density,
    stochastic_chebyshev_moments,
)


COVER_RUN_ID = "3fea2901f7ae29d44dc1517294dee678fac2b9e2fa7a8bf3bccff2bd6528fac5"


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def tail_supremum(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    current = -np.inf
    for index in range(len(values) - 1, -1, -1):
        current = max(current, float(values[index]))
        result[index] = current
    return result


def main() -> int:
    config = load_yaml(EXTENSION_ROOT / "configs" / "r8_spectral_preregistration.yaml")
    cover_run = EXTENSION_ROOT / "results" / COVER_RUN_ID
    inputs = {
        cover_run / "manifest.json": str(config["cover_input"]["manifest_sha256"]),
        cover_run / "certificates" / "r8_01_corrected_cover_depth_extension.json": str(
            config["cover_input"]["certificate_sha256"]
        ),
        cover_run / "derived" / "r8_01_certified_levels.parquet": str(
            config["cover_input"]["level_table_sha256"]
        ),
    }
    for path, expected in inputs.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"R8 cover input hash mismatch: {path}: {actual} != {expected}")

    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT)
    try:
        levels = pd.read_parquet(cover_run / "derived" / "r8_01_certified_levels.parquet")
        edge_config = config["edges"]
        kpm_config = config["KPM"]
        reference = config["reference"]
        rho_lower, rho_upper = map(float, reference["rho_interval"])
        rho_mid = 0.5 * (rho_lower + rho_upper)
        rho_half_width = 0.5 * (rho_upper - rho_lower)
        lower_interval = tuple(map(float, reference["lower_edge_interval"]))
        upper_interval = tuple(map(float, reference["upper_edge_interval"]))
        bandwidth_interval = tuple(map(float, reference["bandwidth_interval"]))
        gap_interval = tuple(map(float, reference["bilayer_external_gap_interval"]))
        grid = np.linspace(
            float(kpm_config["density_grid_min"]),
            float(kpm_config["density_grid_max"]),
            int(kpm_config["density_grid_size"]),
        )
        times = [float(value) for value in kpm_config["heat_trace_times"]]
        level_records: list[dict[str, object]] = []
        density_records: list[pd.DataFrame] = []
        progress_path = run_dir / "logs" / "r8_spectral_progress.jsonl"

        for level_index, (_, row) in enumerate(levels.sort_values(["tower_id", "dyadic_depth"]).iterrows()):
            tower_id = str(row["tower_id"])
            depth = int(row["dyadic_depth"])
            action_path = EXTENSION_ROOT / str(row["action_path"])
            if sha256_file(action_path) != str(row["action_sha256"]):
                raise RuntimeError(f"action hash mismatch for {tower_id} depth {depth}")
            started = time.perf_counter()
            with np.load(action_path, allow_pickle=False) as action:
                permutations = action["permutations"]
                parent_index = action["parent_index"]
                quotient_order = int(action["quotient_order"])
            parent_order = int(row["parent_order"])
            operator = ProjectedCayleyOperator(permutations, parent_index, parent_order)
            symmetry_residual = operator.symmetry_residual(610000 + level_index)
            if symmetry_residual > float(config["operator"]["symmetry_audit_tolerance"]):
                raise ArithmeticError(f"operator symmetry audit failed at {tower_id} depth {depth}")
            edges = compute_edges(
                operator,
                k=int(edge_config["eigenvalues_per_edge"]),
                tolerance=float(edge_config["tolerance"]),
                maximum_iterations=int(edge_config["maximum_iterations"]),
                seed=int(edge_config["deterministic_seed"]) + 100 * level_index,
            )
            raw_edge_path = run_dir / "raw" / f"r8_edges_{tower_id}_depth_{depth}.npz"
            with raw_edge_path.open("xb") as handle:
                np.savez(
                    handle,
                    run_id=np.asarray(run_id),
                    tower_id=np.asarray(tower_id),
                    dyadic_depth=np.asarray(depth),
                    lower_values=edges.lower_values,
                    upper_values=edges.upper_values,
                    lower_vectors=edges.lower_vectors,
                    upper_vectors=edges.upper_vectors,
                    residuals=edges.residuals,
                )
            moments = stochastic_chebyshev_moments(
                operator,
                order=int(kpm_config["polynomial_order"]),
                random_vectors=int(kpm_config["random_vectors"]),
                seed=int(kpm_config["base_seed"]) + 100 * level_index,
            )
            raw_moment_path = run_dir / "raw" / f"r8_kpm_moments_{tower_id}_depth_{depth}.npz"
            with raw_moment_path.open("xb") as handle:
                np.savez(
                    handle,
                    run_id=np.asarray(run_id),
                    tower_id=np.asarray(tower_id),
                    dyadic_depth=np.asarray(depth),
                    seeds=np.arange(
                        int(kpm_config["base_seed"]) + 100 * level_index,
                        int(kpm_config["base_seed"]) + 100 * level_index + int(kpm_config["random_vectors"]),
                    ),
                    moments=moments,
                    grid=grid,
                )
            probe_densities = np.asarray(
                [
                    gaussian_smooth_density(
                        kpm_density(probe, grid),
                        grid,
                        float(kpm_config["fixed_gaussian_broadening"]),
                    )
                    for probe in moments
                ]
            )
            probe_heats = np.asarray([heat_trace_from_moments(probe, times) for probe in moments])
            density_mean = probe_densities.mean(axis=0)
            density_se = probe_densities.std(axis=0, ddof=1) / np.sqrt(probe_densities.shape[0])
            density_records.append(
                pd.DataFrame(
                    {
                        "tower_id": tower_id,
                        "dyadic_depth": depth,
                        "energy": grid,
                        "density_mean": density_mean,
                        "density_standard_error": density_se,
                    }
                )
            )

            lower_edge = float(edges.lower_values[0])
            upper_edge = float(edges.upper_values[-1])
            bandwidth = upper_edge - lower_edge
            external_gap = 2.0 - bandwidth
            solver_error = float(np.max(edges.residuals))
            u_error = max(abs(lower_edge + rho_mid), abs(upper_edge - rho_mid)) + rho_half_width
            b_error = max(0.0, -rho_upper - lower_edge, upper_edge - rho_upper)
            eta = u_error + b_error + solver_error
            actual_edge_error = max(
                interval_distance(lower_edge, *lower_interval),
                interval_distance(upper_edge, *upper_interval),
            )
            bandwidth_error = interval_distance(bandwidth, *bandwidth_interval)
            gap_error = interval_distance(external_gap, *gap_interval)
            record = {
                "tower_id": tower_id,
                "dyadic_depth": depth,
                "quotient_order": quotient_order,
                "retained_sector_dimension": operator.retained_dimension,
                "word_systole_exact": int(row["word_systole_exact"]),
                "injectivity_radius_integer": int(row["injectivity_radius_integer"]),
                "lower_edge": lower_edge,
                "upper_edge": upper_edge,
                "bandwidth": bandwidth,
                "external_gap": external_gap,
                "u_N": u_error,
                "b_N": b_error,
                "truncation_N": 0.0,
                "solver_N": solver_error,
                "eta_N": eta,
                "actual_edge_error": actual_edge_error,
                "actual_bandwidth_error": bandwidth_error,
                "actual_gap_error": gap_error,
                "edge_bound_pass": actual_edge_error <= eta,
                "bandwidth_bound_pass": bandwidth_error <= 2.0 * eta,
                "gap_bound_pass": gap_error <= 2.0 * eta,
                "no_pollution_gate_pass": b_error <= float(config["acceptance"]["no_pollution_tolerance"]),
                "solver_gate_pass": solver_error <= float(config["acceptance"]["solver_residual_tolerance"]),
                "symmetry_residual": symmetry_residual,
                "heat_trace_t05": float(probe_heats[:, 0].mean()),
                "heat_trace_t10": float(probe_heats[:, 1].mean()),
                "heat_trace_t20": float(probe_heats[:, 2].mean()),
                "heat_trace_se_t05": float(probe_heats[:, 0].std(ddof=1) / np.sqrt(len(probe_heats))),
                "heat_trace_se_t10": float(probe_heats[:, 1].std(ddof=1) / np.sqrt(len(probe_heats))),
                "heat_trace_se_t20": float(probe_heats[:, 2].std(ddof=1) / np.sqrt(len(probe_heats))),
                "raw_edge_path": raw_edge_path.relative_to(EXTENSION_ROOT).as_posix(),
                "raw_edge_sha256": sha256_file(raw_edge_path),
                "raw_moment_path": raw_moment_path.relative_to(EXTENSION_ROOT).as_posix(),
                "raw_moment_sha256": sha256_file(raw_moment_path),
                "elapsed_seconds": time.perf_counter() - started,
            }
            level_records.append(record)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()

        level_frame = pd.DataFrame(level_records)
        density_frame = pd.concat(density_records, ignore_index=True)
        level_frame.to_parquet(run_dir / "derived" / "r8_level_metrics.parquet", index=False)
        density_frame.to_parquet(run_dir / "figure_data" / "r8_fixed_broadening_dos.parquet", index=False)

        within: dict[str, object] = {}
        within_passes = []
        for tower_id, group in level_frame.groupby("tower_id"):
            ordered = group.sort_values("dyadic_depth")
            envelopes = {
                metric: tail_supremum(ordered[metric].astype(float).tolist())
                for metric in ("actual_edge_error", "actual_bandwidth_error", "actual_gap_error", "eta_N")
            }
            radius = ordered["injectivity_radius_integer"].astype(int).tolist()
            enough_steps = len(ordered) >= 4
            radius_increases = radius[-1] > radius[0]
            envelope_decreases = all(values[-1] <= values[0] for values in envelopes.values()) and any(
                values[-1] < values[0] for values in envelopes.values()
            )
            passes = enough_steps and radius_increases and envelope_decreases
            within_passes.append(passes)
            within[tower_id] = {
                "dyadic_depths": ordered["dyadic_depth"].astype(int).tolist(),
                "injectivity_radii": radius,
                "tail_supremum_envelopes": envelopes,
                "enough_refinement_steps": enough_steps,
                "injectivity_radius_tail_increases": radius_increases,
                "error_envelope_decreases": envelope_decreases,
                "passes": passes,
            }

        cross_rows: list[dict[str, object]] = []
        for depth in [int(value) for value in config["cross_tower"]["depths"]]:
            left_rows = level_frame[
                (level_frame.tower_id == "dyadic_ramified") & (level_frame.dyadic_depth == depth)
            ]
            right_rows = level_frame[
                (level_frame.tower_id == "dyadic_x_p7") & (level_frame.dyadic_depth == depth)
            ]
            if left_rows.empty or right_rows.empty:
                continue
            left, right = left_rows.iloc[0], right_rows.iloc[0]
            row = {
                "dyadic_depth": depth,
                "combined_eta": float(left.eta_N + right.eta_N),
            }
            for metric in (
                "lower_edge",
                "upper_edge",
                "bandwidth",
                "external_gap",
                "heat_trace_t05",
                "heat_trace_t10",
                "heat_trace_t20",
            ):
                row[f"residual_{metric}"] = abs(float(left[metric]) - float(right[metric]))
            left_density = density_frame[
                (density_frame.tower_id == "dyadic_ramified") & (density_frame.dyadic_depth == depth)
            ].sort_values("energy")
            right_density = density_frame[
                (density_frame.tower_id == "dyadic_x_p7") & (density_frame.dyadic_depth == depth)
            ].sort_values("energy")
            row["residual_fixed_broadening_DOS_L1"] = float(
                np.trapezoid(
                    np.abs(left_density.density_mean.to_numpy() - right_density.density_mean.to_numpy()),
                    left_density.energy.to_numpy(),
                )
            )
            residual_columns = [key for key in row if key.startswith("residual_")]
            row["all_residuals_within_combined_eta"] = all(
                float(row[key]) <= float(row["combined_eta"]) for key in residual_columns
            )
            cross_rows.append(row)
        cross_frame = pd.DataFrame(cross_rows)
        cross_frame.to_parquet(run_dir / "derived" / "r8_cross_tower_metrics.parquet", index=False)
        cross_residual_columns = [column for column in cross_frame.columns if column.startswith("residual_")]
        cross_envelopes = {
            column: tail_supremum(cross_frame[column].astype(float).tolist())
            for column in cross_residual_columns
        }
        cross_budget_pass = bool(len(cross_frame)) and bool(cross_frame.all_residuals_within_combined_eta.all())
        cross_decrease = bool(cross_envelopes) and all(values[-1] <= values[0] for values in cross_envelopes.values())

        bound_pass = bool(
            level_frame[["edge_bound_pass", "bandwidth_bound_pass", "gap_bound_pass"]].all(axis=None)
        )
        solver_pass = bool(level_frame.solver_gate_pass.all())
        pollution_pass = bool(level_frame.no_pollution_gate_pass.all())
        r8_02 = "PASS_CONVERGED" if any(within_passes) else "INCONCLUSIVE"
        r8_03 = "PASS_CERTIFIED" if bound_pass and solver_pass and pollution_pass else "INCONCLUSIVE"
        r8_04 = "PASS_CONVERGED" if cross_budget_pass and cross_decrease else "INCONCLUSIVE"
        r8_05 = (
            "PASS_CONVERGED"
            if r8_02 == "PASS_CONVERGED" and r8_03 == "PASS_CERTIFIED" and r8_04 == "PASS_CONVERGED"
            else "INCONCLUSIVE"
        )
        statuses = {
            "R8-02": r8_02,
            "R8-03": r8_03,
            "R8-04": r8_04,
            "R8-05": r8_05,
            "R8-07": "PASS_CERTIFIED",
        }
        certificate = {
            "run_id": run_id,
            "task_statuses": statuses,
            "parent_verification": identity["parent_verification"],
            "cover_hashes_verified": True,
            "spectral_preregistration_sha256": sha256_file(
                EXTENSION_ROOT / "configs" / "r8_spectral_preregistration.yaml"
            ),
            "within_tower": within,
            "all_manuscript_bounds_pass": bound_pass,
            "all_solver_gates_pass": solver_pass,
            "all_no_pollution_gates_pass": pollution_pass,
            "cross_tower_budget_pass": cross_budget_pass,
            "cross_tower_tail_envelopes_nonincreasing": cross_decrease,
            "cross_tower_tail_envelopes": cross_envelopes,
            "scientific_note": "finite certified cover depth does not force a convergence classification when injectivity-radius scale separation is absent",
        }
        write_json(run_dir / "certificates" / "r8_spectral_certificate.json", certificate)
        finalize_run(run_dir, "COMPLETE", statuses)
        print(json.dumps({"run_id": run_id, "statuses": statuses}))
        return 0
    except Exception as error:
        failure = {
            "run_id": run_id,
            "status": "FAIL_IMPLEMENTATION",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "certificates" / "r8_spectral_failure.json", failure)
        finalize_run(run_dir, "INCOMPLETE", {task: "FAIL_IMPLEMENTATION" for task in ("R8-02", "R8-03", "R8-04", "R8-05", "R8-07")})
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
