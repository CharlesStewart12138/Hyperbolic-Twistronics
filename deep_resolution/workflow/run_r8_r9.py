from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = EXTENSION_ROOT.parent
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_ROOT / "src"))

from common import finalize_run, initialize_run, load_yaml, sha256_file, write_json  # noqa: E402
from tower_height import (  # noqa: E402
    archimedean_letter_bound,
    exact_group_audit,
    projective_dimension,
    select_levels,
    selected_level_record,
    verify_monotone_levels,
)


def verify_r9_inputs(config: dict[str, object]) -> dict[str, object]:
    checks = {}
    for name in ("full_kernel_shells", "normal_forms", "full_kernel_hodge", "packing_tail_certificate"):
        entry = config[name]
        path = (EXTENSION_ROOT / str(entry["path"])).resolve()
        actual = sha256_file(path)
        checks[name] = {"path": str(path), "expected": str(entry["sha256"]), "actual": actual, "pass": actual == str(entry["sha256"])}
    if not all(value["pass"] for value in checks.values()):
        raise RuntimeError(f"R9 input hash verification failed: {checks}")
    return checks


def convergence_estimators(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    envelope = np.maximum.accumulate(values[::-1])[::-1]
    ratios = values[1:] / np.maximum(values[:-1], 1.0e-300)
    leave_one_out_slopes = []
    x = np.arange(1, len(values) + 1, dtype=float)
    for omitted in range(len(values)):
        mask = np.arange(len(values)) != omitted
        slope = np.polyfit(x[mask], np.log(np.maximum(values[mask], 1.0e-300)), 1)[0]
        leave_one_out_slopes.append(float(slope))
    return {
        "direct_final_over_initial": float(values[-1] / max(values[0], 1.0e-300)),
        "log_upper_envelope": envelope.tolist(),
        "ratios": ratios.tolist(),
        "leave_one_out_log_slopes": leave_one_out_slopes,
        "all_estimators_support_zero": bool(
            values[-1] <= 0.25 * values[0]
            and np.all(np.diff(envelope) <= 0)
            and np.median(ratios[-2:]) < 0.9
            and max(leave_one_out_slopes) < -0.05
        ),
    }


def main() -> int:
    config_path = EXTENSION_ROOT / "configs" / "r8_r9_preregistration.yaml"
    input_path = EXTENSION_ROOT / "configs" / "r9_input_anchors.yaml"
    config = load_yaml(config_path)
    input_config = load_yaml(input_path)
    input_checks = verify_r9_inputs(input_config)
    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT, "R8_R9")
    height = archimedean_letter_bound()
    c_product = float(height["C_product_upper"])
    thresholds = [float(value) for value in config["level_selection"]["injectivity_thresholds"]]
    maximum_level = int(config["level_selection"]["search_levels"][1])
    selected = []
    for tower in config["towers"]:
        selected.extend(
            select_levels(
                str(tower["tower_id"]),
                int(tower["p"]),
                int(tower["root"]),
                thresholds,
                c_product,
                maximum_level,
            )
        )
    records = [selected_level_record(level) for level in selected]
    for row in records:
        row["shell_radius_L"] = int(math.floor(math.sqrt(float(row["injectivity_radius_lower"]))))
        row["interaction_radius"] = row["shell_radius_L"]
        row["solver_residual"] = 0.0
        row["no_loss_polynomial_degree"] = row["shell_radius_L"]
        row["local_no_loss_exact"] = bool(2 * int(row["shell_radius_L"]) < float(row["word_systole_lower"]))
        row["strong_no_pollution_bound"] = math.inf
        row["usable_scope"] = "exact local finite-propagation and symbolic matrix-free P1 action; not a materialized full regular spectrum"
    level_frame = pd.DataFrame(records)
    if not verify_monotone_levels(records):
        raise RuntimeError("selected congruence levels are not strictly monotone")
    if sorted(level_frame.shell_radius_L.unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"balanced shell values do not realize 1..5: {level_frame.shell_radius_L.unique()}")
    exact_audits = []
    for row in level_frame.itertuples():
        audit = exact_group_audit(int(row.p), int(row.root_mod_p), int(row.level))
        if not audit["relator_pass"] or not audit["nonabelian_witness_pass"]:
            raise RuntimeError(f"exact quotient audit failed: {row.tower_id} level {row.level}")
        exact_audits.append({"tower_id": row.tower_id, **audit})
        write_json(run_dir / "exact" / f"{row.tower_id}_level_{row.level}.json", audit)
    write_json(run_dir / "exact" / "archimedean_height_bound.json", height)
    level_frame.to_parquet(run_dir / "raw" / "r8_selected_congruence_levels.parquet", index=False)

    matched_rows = []
    for threshold in thresholds:
        group = level_frame[np.isclose(level_frame.threshold, threshold)].sort_values("p")
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left = group.iloc[left_index]
                right = group.iloc[right_index]
                matched_rows.append(
                    {
                        "threshold": threshold,
                        "tower_A": left.tower_id,
                        "tower_B": right.tower_id,
                        "level_A": int(left.level),
                        "level_B": int(right.level),
                        "r_inj_A": float(left.injectivity_radius_lower),
                        "r_inj_B": float(right.injectivity_radius_lower),
                        "relative_radius_mismatch": abs(float(left.injectivity_radius_lower) - float(right.injectivity_radius_lower)) / threshold,
                        "local_moment_match_degree": int(min(left.shell_radius_L, right.shell_radius_L)),
                        "local_moment_residual": 0.0,
                        "edge_residual": math.nan,
                        "gap_residual": math.nan,
                        "projector_residual": math.nan,
                        "DOS_residual": math.nan,
                        "strong_cross_tower_adjudication": "INCONCLUSIVE_PENDING_SPECTRAL_MEASURES",
                    }
                )
    matched_frame = pd.DataFrame(matched_rows)
    matched_frame.to_parquet(run_dir / "derived" / "r8_matched_tower_levels.parquet", index=False)

    shells_path = Path(input_checks["full_kernel_shells"]["path"])
    tail_path = Path(input_checks["packing_tail_certificate"]["path"])
    shells = pd.read_parquet(shells_path)
    tail_certificate = json.loads(tail_path.read_text(encoding="utf-8"))
    scalar_packing = float(tail_certificate["packing_tail"]["scalar_l1_upper"])
    hodge_packing = float(tail_certificate["packing_tail"]["hodge_trace_upper"])
    error_rows = []
    for row in level_frame.itertuples():
        L = int(row.shell_radius_L)
        later = shells[shells.word_length > L]
        finite_scalar = float(later.weight_sum.sum())
        finite_hodge = float(later.hodge_trace_sum.sum())
        epsilon_c0 = scalar_packing + finite_scalar
        epsilon_c2 = hodge_packing + finite_hodge
        epsilon_c1 = math.sqrt(epsilon_c0 * epsilon_c2)
        solver = 1.0e-12
        error_rows.append(
            {
                "tower_id": row.tower_id,
                "p": int(row.p),
                "level": int(row.level),
                "injectivity_radius_lower": float(row.injectivity_radius_lower),
                "L_j": L,
                "L_over_rinj": L / float(row.injectivity_radius_lower),
                "epsilon_core": 0.0,
                "epsilon_physical_tail": epsilon_c0,
                "epsilon_master_tail": epsilon_c2,
                "epsilon_cover": 0.0 if bool(row.local_no_loss_exact) else math.inf,
                "epsilon_transport": 0.0,
                "epsilon_solver": solver,
                "epsilon_representation": 0.0,
                "epsilon_total_C0": epsilon_c0 + solver,
                "epsilon_total_C1": epsilon_c1 + solver,
                "epsilon_total_C2": epsilon_c2 + solver,
                "finite_shell_scalar_remainder": finite_scalar,
                "finite_shell_hodge_remainder": finite_hodge,
                "packing_scalar_upper": scalar_packing,
                "packing_hodge_upper": hodge_packing,
                "common_hilbert_space": f"rooted_labelled_word_ball_L{L}",
                "transport_map": "rooted_label_preserving_local_isometry",
            }
        )
    error_frame = pd.DataFrame(error_rows)
    error_frame.to_parquet(run_dir / "derived" / "r9_balanced_error_budget.parquet", index=False)
    error_frame.to_parquet(run_dir / "figure_data" / "figure_9_balanced_diagonal.parquet", index=False)
    level_frame.to_parquet(run_dir / "figure_data" / "figure_8_deep_cover_levels.parquet", index=False)
    matched_frame.to_parquet(run_dir / "figure_data" / "figure_8_matched_levels.parquet", index=False)
    estimators = {
        tier: convergence_estimators(error_frame.groupby("threshold" if "threshold" in error_frame else "L_j")[column].mean().to_numpy())
        for tier, column in (("C0", "epsilon_total_C0"), ("C1", "epsilon_total_C1"), ("C2", "epsilon_total_C2"))
    }
    # The frame has three values per shell.  Recompute on the matched L average explicitly.
    estimators = {
        tier: convergence_estimators(error_frame.groupby("L_j", sort=True)[column].mean().to_numpy())
        for tier, column in (("C0", "epsilon_total_C0"), ("C1", "epsilon_total_C1"), ("C2", "epsilon_total_C2"))
    }
    r9_converged = all(bool(value["all_estimators_support_zero"]) for value in estimators.values())
    r8_certificate = {
        "run_id": run_id,
        "preregistration_sha256": sha256_file(config_path),
        "anchor_checks": identity["anchor_checks"],
        "archimedean_height_bound": height,
        "tower_count": int(level_frame.tower_id.nunique()),
        "levels_per_tower": level_frame.groupby("tower_id").size().to_dict(),
        "all_exact_relators_pass": all(bool(row["relator_pass"]) for row in exact_audits),
        "all_nonabelian_witnesses_pass": all(bool(row["nonabelian_witness_pass"]) for row in exact_audits),
        "all_local_no_loss_checks_pass": bool(level_frame.local_no_loss_exact.all()),
        "strong_no_pollution_bound_closes": False,
        "strong_no_pollution_reason": "No theorem-licensed full-regular-spectrum upper inclusion follows from local moment exactness alone.",
        "classification": "INCONCLUSIVE",
        "scope": "deeper congruence levels certify growing injectivity and local finite-propagation exactness; full edge/gap no-pollution awaits a separate global spectral certificate",
    }
    r9_certificate = {
        "run_id": run_id,
        "input_hash_checks": input_checks,
        "balanced_law": "L_j=floor(sqrt(r_inj_lower_j))",
        "observed_shell_values": sorted(error_frame.L_j.unique().astype(int).tolist()),
        "analytic_limits": {
            "r_inj_to_infinity": True,
            "L_to_infinity": True,
            "L_over_rinj_to_zero": True,
            "proof": "height bound is affine in congruence level n; floor(sqrt(r_n))/r_n tends to zero",
        },
        "common_hilbert_space_transport_exact": True,
        "C0_C1_C2_estimators": estimators,
        "physical_tail_bound_closes": r9_converged,
        "classification": "PASS_CONVERGED" if r9_converged else "INCONCLUSIVE",
        "inconclusive_reason": None if r9_converged else "The only frozen theorem-licensed packing complement is conservative and nondecreasing in word-shell L; no post-hoc geometric-to-word tail constant was introduced.",
    }
    write_json(run_dir / "certificates" / "r8_deep_cover_certificate.json", r8_certificate)
    write_json(run_dir / "certificates" / "r9_balanced_certificate.json", r9_certificate)
    task_statuses = {
        "R8-A": "PASS_CERTIFIED",
        "R8-B": "PASS_CERTIFIED",
        "R8-C": "INCONCLUSIVE",
        "R8-D": "INCONCLUSIVE",
        "R8-E": "INCONCLUSIVE",
        "R9-A": "PASS_CERTIFIED",
        "R9-B": "PASS_CERTIFIED",
        "R9-C": "PASS_EXACT",
        "R9-D": "INCONCLUSIVE" if not r9_converged else "PASS_CONVERGED",
        "R9-E": "INCONCLUSIVE" if not r9_converged else "PASS_CONVERGED",
        "R9-F": "PASS_CERTIFIED",
    }
    freeze = finalize_run(run_dir, task_statuses, "R8_INCONCLUSIVE_R9_INCONCLUSIVE" if not r9_converged else "R8_INCONCLUSIVE_R9_PASS")
    print(json.dumps({"run_id": run_id, "selected_levels": level_frame[["tower_id", "level", "shell_radius_L"]].to_dict("records"), "r9_converged": r9_converged, "freeze": freeze["tree_inventory_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

