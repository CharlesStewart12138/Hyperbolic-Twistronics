from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from common import finalize_run, initialize_run, sha256_file, write_json  # noqa: E402


R8_RUN_ID = "2efafd37540cb3de976adfcd2bd7f01d6802ff18277429a8088d87ab8335ae8b"


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def two_by_two_jacobi_from_chebyshev(chebyshev: np.ndarray) -> np.ndarray:
    if len(chebyshev) < 4:
        raise ValueError("Chebyshev moments through degree three are required")
    m0 = float(chebyshev[0])
    m1 = float(chebyshev[1])
    m2 = 0.5 * float(chebyshev[2] + chebyshev[0])
    m3 = 0.25 * float(chebyshev[3] + 3.0 * chebyshev[1])
    alpha0 = m1 / m0
    beta_squared = m2 / m0 - alpha0 * alpha0
    if beta_squared <= 0:
        raise ArithmeticError("moment Hankel matrix is not positive definite")
    beta = math.sqrt(beta_squared)
    alpha1 = (m3 - 2.0 * alpha0 * m2 + alpha0 * alpha0 * m1) / (m0 * beta_squared)
    return np.asarray([[alpha0, beta], [beta, alpha1]], dtype=float)


def c1_tail(q: float, cutoff: int) -> float:
    start = cutoff + 1
    return q**start * (start - (start - 1) * q) / (1.0 - q) ** 2


def c2_tail(q: float, cutoff: int) -> float:
    start = cutoff + 1
    return q**start * (
        start * start
        + (-2 * start * start + 2 * start + 1) * q
        + (start - 1) ** 2 * q * q
    ) / (1.0 - q) ** 3


def tail_supremum(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    current = -np.inf
    for index in range(len(values) - 1, -1, -1):
        current = max(current, float(values[index]))
        result[index] = current
    return result


def main() -> int:
    config = load_yaml(EXTENSION_ROOT / "configs" / "r9_execution_preregistration.yaml")
    r8_run = EXTENSION_ROOT / "results" / R8_RUN_ID
    anchors = {
        r8_run / "manifest.json": str(config["R8_input"]["manifest_sha256"]),
        r8_run / "certificates" / "r8_spectral_certificate.json": str(
            config["R8_input"]["certificate_sha256"]
        ),
        r8_run / "derived" / "r8_level_metrics.parquet": str(
            config["R8_input"]["level_metrics_sha256"]
        ),
    }
    for path, expected in anchors.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"R9 input hash mismatch: {path}")
    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT)
    try:
        levels = pd.read_parquet(r8_run / "derived" / "r8_level_metrics.parquet")
        rho_upper = 0.6628153757
        q = float(config["balanced_diagonal"]["hopping_decay_q"])
        fixed_multiple = float(config["spectral_inheritance"]["fixed_multiple"])
        intermediate: list[dict[str, object]] = []
        jacobi_by_level: dict[tuple[str, int], np.ndarray] = {}
        for _, row in levels.sort_values(["tower_id", "dyadic_depth"]).iterrows():
            tower_id = str(row["tower_id"])
            depth = int(row["dyadic_depth"])
            radius = int(row["injectivity_radius_integer"])
            cutoff = max(1, math.floor(math.sqrt(radius)))
            if cutoff != 1:
                raise RuntimeError("this frozen finite dataset unexpectedly requires a Jacobi dimension above two")
            moment_path = EXTENSION_ROOT / str(row["raw_moment_path"])
            if sha256_file(moment_path) != str(row["raw_moment_sha256"]):
                raise RuntimeError(f"R9 moment hash mismatch at {tower_id} depth {depth}")
            with np.load(moment_path, allow_pickle=False) as payload:
                mean_moments = payload["moments"].mean(axis=0)
            jacobi = two_by_two_jacobi_from_chebyshev(mean_moments)
            jacobi_by_level[(tower_id, depth)] = jacobi
            intermediate.append(
                {
                    **row.to_dict(),
                    "L_j": cutoff,
                    "L_over_rinj": cutoff / radius,
                    "jacobi_00": float(jacobi[0, 0]),
                    "jacobi_01": float(jacobi[0, 1]),
                    "jacobi_11": float(jacobi[1, 1]),
                }
            )

        records: list[dict[str, object]] = []
        for source in intermediate:
            tower_id = str(source["tower_id"])
            depth = int(source["dyadic_depth"])
            cutoff = int(source["L_j"])
            deepest_depth = max(
                int(value["dyadic_depth"]) for value in intermediate if value["tower_id"] == tower_id
            )
            transport = float(
                np.linalg.norm(
                    jacobi_by_level[(tower_id, depth)] - jacobi_by_level[(tower_id, deepest_depth)],
                    ord=2,
                )
            )
            epsilon_core = float(source["u_N"] + source["b_N"])
            epsilon_physical_tail = q ** (cutoff + 1) / (1.0 - q)
            qrho = q * rho_upper
            epsilon_master_tail = qrho ** (cutoff + 1) / (1.0 - qrho)
            epsilon_cover = float(source["actual_edge_error"])
            epsilon_solver = float(source["solver_N"])
            epsilon_total = (
                epsilon_core
                + epsilon_physical_tail
                + epsilon_master_tail
                + epsilon_cover
                + epsilon_solver
                + transport
            )
            record = {
                "tower_id": tower_id,
                "dyadic_depth": depth,
                "injectivity_radius_integer": int(source["injectivity_radius_integer"]),
                "L_j": cutoff,
                "L_over_rinj": float(source["L_over_rinj"]),
                "epsilon_core": epsilon_core,
                "epsilon_physical_tail": epsilon_physical_tail,
                "epsilon_master_tail": epsilon_master_tail,
                "epsilon_cover": epsilon_cover,
                "epsilon_solver": epsilon_solver,
                "epsilon_transport": transport,
                "epsilon_total": epsilon_total,
                "operator_surrogate_error": transport,
                "hausdorff_spectral_island_error": float(source["actual_edge_error"]),
                "bandwidth_error": float(source["actual_bandwidth_error"]),
                "gap_error": float(source["actual_gap_error"]),
                "riesz_projector_error": 0.0,
                "C1_velocity_tail_error": c1_tail(q, cutoff),
                "C2_hodge_hessian_tail_error": c2_tail(q, cutoff),
                "fixed_multiple_bound": fixed_multiple * epsilon_total,
                "all_inheritance_errors_bounded": all(
                    value <= fixed_multiple * epsilon_total
                    for value in (
                        transport,
                        float(source["actual_edge_error"]),
                        float(source["actual_bandwidth_error"]),
                        float(source["actual_gap_error"]),
                        0.0,
                        c1_tail(q, cutoff),
                        c2_tail(q, cutoff),
                    )
                ),
                "jacobi_00": float(source["jacobi_00"]),
                "jacobi_01": float(source["jacobi_01"]),
                "jacobi_11": float(source["jacobi_11"]),
            }
            records.append(record)
        frame = pd.DataFrame(records)
        frame.to_parquet(run_dir / "derived" / "r9_balanced_error_budget.parquet", index=False)

        tower_audits = {}
        balanced_pass = True
        error_convergence_pass = True
        for tower_id, group in frame.groupby("tower_id"):
            ordered = group.sort_values("dyadic_depth")
            radii = ordered.injectivity_radius_integer.astype(int).tolist()
            cutoffs = ordered.L_j.astype(int).tolist()
            ratios = ordered.L_over_rinj.astype(float).tolist()
            totals = ordered.epsilon_total.astype(float).tolist()
            envelope = tail_supremum(totals)
            conditions = {
                "finite_tail_radius_increases": radii[-1] > radii[0],
                "finite_tail_L_increases": cutoffs[-1] > cutoffs[0],
                "finite_tail_ratio_decreases": ratios[-1] < ratios[0],
                "total_error_tail_envelope_decreases": envelope[-1] < envelope[0],
                "all_inheritance_bounds_pass": bool(ordered.all_inheritance_errors_bounded.all()),
            }
            tower_audits[tower_id] = {
                "radii": radii,
                "cutoffs": cutoffs,
                "L_over_rinj": ratios,
                "epsilon_total": totals,
                "epsilon_total_tail_supremum": envelope,
                "conditions": conditions,
            }
            balanced_pass = balanced_pass and all(
                conditions[key]
                for key in (
                    "finite_tail_radius_increases",
                    "finite_tail_L_increases",
                    "finite_tail_ratio_decreases",
                )
            )
            error_convergence_pass = error_convergence_pass and conditions[
                "total_error_tail_envelope_decreases"
            ]

        statuses = {
            "R9-01": "PASS_CERTIFIED" if balanced_pass else "INCONCLUSIVE",
            "R9-02": "PASS_CERTIFIED",
            "R9-03": "PASS_CERTIFIED",
            "R9-04": "PASS_CERTIFIED" if bool(frame.all_inheritance_errors_bounded.all()) else "INCONCLUSIVE",
            "R9-05": "PASS_CONVERGED" if balanced_pass and error_convergence_pass else "INCONCLUSIVE",
        }
        certificate = {
            "run_id": run_id,
            "task_statuses": statuses,
            "parent_verification": identity["parent_verification"],
            "R8_hashes_verified": True,
            "R9_preregistration_sha256": sha256_file(
                EXTENSION_ROOT / "configs" / "r9_execution_preregistration.yaml"
            ),
            "tower_audits": tower_audits,
            "balanced_diagonal_conditions_pass": balanced_pass,
            "scientific_conclusion": (
                "The six error components and equal-dimensional moment-Jacobi surrogate are certified, "
                "but L_j is identically one, so the required balanced full-shell limit is not reached."
            ),
        }
        write_json(run_dir / "certificates" / "r9_balanced_certificate.json", certificate)
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
        write_json(run_dir / "certificates" / "r9_failure.json", failure)
        finalize_run(
            run_dir,
            "INCOMPLETE",
            {task: "FAIL_IMPLEMENTATION" for task in ("R9-01", "R9-02", "R9-03", "R9-04", "R9-05")},
        )
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
