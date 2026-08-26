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
from cover_towers import reduction_map, save_action  # noqa: E402
from dyadic_ring import build_ring  # noqa: E402
from group_actions_v2 import enumerate_marked_group_v2  # noqa: E402
from word_automaton_v2 import export_word_acceptor, shortest_kernel_normal_word  # noqa: E402


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def commutator_witness(permutations: np.ndarray) -> tuple[bool, list[int], int]:
    word = [1, 2, -1, -2]
    state = 0
    for letter in word:
        move = letter - 1 if letter > 0 else 4 + (-letter - 1)
        state = int(permutations[move, state])
    return state != 0, word, state


def main() -> int:
    base = load_yaml(EXTENSION_ROOT / "configs" / "r8_r9_r10_preregistration.yaml")
    recovery = load_yaml(EXTENSION_ROOT / "configs" / "r8_cover_recovery_preregistration.yaml")
    extension = load_yaml(EXTENSION_ROOT / "configs" / "extension.yaml")
    predecessor_path = EXTENSION_ROOT / str(recovery["immutable_predecessor"]["path"])
    expected_predecessor_hash = str(recovery["immutable_predecessor"]["sha256"])
    if sha256_file(predecessor_path) != expected_predecessor_hash:
        raise RuntimeError("R8 predecessor preregistration hash mismatch")

    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT)
    task_status = "FAIL_IMPLEMENTATION"
    try:
        exact_dir = run_dir / "exact"
        logs_dir = run_dir / "logs"
        acceptor_path = exact_dir / "kbmag_shortlex_word_acceptor.txt"
        acceptor = export_word_acceptor(
            EXTENSION_ROOT,
            acceptor_path,
            gap_bash=str(extension["gap_backend"]["gap_bash"]),
            gap_binary_cygwin=str(extension["gap_backend"]["gap_binary_cygwin"]),
            stdout_log=logs_dir / "kbmag_word_acceptor.stdout.log",
            stderr_log=logs_dir / "kbmag_word_acceptor.stderr.log",
        )

        raw_actions = run_dir / "raw" / "cover_actions"
        raw_actions.mkdir(parents=True, exist_ok=False)
        candidate_depths = [int(value) for value in recovery["recovery_cover_rule"]["candidate_dyadic_depths"]]
        maximum_order = int(recovery["recovery_cover_rule"]["resource_cap_group_order"])
        maximum_product_states = int(
            recovery["recovery_cover_rule"]["resource_cap_product_automaton_states"]
        )
        ring_cache = {}
        candidate_rows: list[dict[str, object]] = []
        admissible_rows: list[dict[str, object]] = []
        resource_limits: list[dict[str, object]] = []

        for tower_config in recovery["recovery_cover_rule"]["tower_definitions"]:
            tower_id = str(tower_config["tower_id"])
            with_auxiliary = tower_config["fixed_auxiliary_quotient"] is not None
            parent = None
            for depth in range(1, max(candidate_depths) + 1):
                started = time.perf_counter()
                ring = ring_cache.setdefault(depth, build_ring(depth))
                try:
                    group = enumerate_marked_group_v2(
                        tower_id,
                        ring,
                        with_auxiliary_p7=with_auxiliary,
                        maximum_order=maximum_order,
                    )
                except MemoryError as error:
                    resource_limits.append(
                        {
                            "tower_id": tower_id,
                            "dyadic_depth": depth,
                            "resource": "group_order",
                            "cap": maximum_order,
                            "message": str(error),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                    break
                if parent is None:
                    parent = group
                    continue

                parent_index = reduction_map(group, parent)
                counts = np.bincount(parent_index, minlength=parent.order)
                uniform_fiber = bool(np.all(counts == counts[0]))
                strict_refinement = group.order > parent.order
                nonabelian, commutator_word, commutator_state = commutator_witness(group.permutations)
                retained_dimension = group.order - parent.order
                candidate: dict[str, object] = {
                    "tower_id": tower_id,
                    "dyadic_depth": depth,
                    "quotient_order": group.order,
                    "parent_order": parent.order,
                    "fiber_size": group.order // parent.order,
                    "uniform_reduction_fiber": uniform_fiber,
                    "strict_refinement": strict_refinement,
                    "duplicate_kernel": not strict_refinement,
                    "normal_cover": True,
                    "nonabelian": nonabelian,
                    "commutator_word": json.dumps(commutator_word),
                    "commutator_image_index": commutator_state,
                    "retained_sector_dimension": retained_dimension,
                    "enumeration_elapsed_seconds": time.perf_counter() - started,
                    "ambient_group_order_bound": group.upper_order_bound,
                    "admissible": False,
                }
                if depth in candidate_depths and strict_refinement and retained_dimension > 0 and nonabelian:
                    systole_started = time.perf_counter()
                    try:
                        systole, witness, product_states = shortest_kernel_normal_word(
                            group.permutations,
                            acceptor,
                            maximum_product_states=maximum_product_states,
                        )
                    except MemoryError as error:
                        resource_limits.append(
                            {
                                "tower_id": tower_id,
                                "dyadic_depth": depth,
                                "resource": "product_automaton_states",
                                "cap": maximum_product_states,
                                "message": str(error),
                                "elapsed_seconds": time.perf_counter() - systole_started,
                            }
                        )
                    else:
                        action_path = raw_actions / f"{tower_id}_depth_{depth}.npz"
                        artifact = save_action(action_path, run_id, group, parent_index, systole, witness)
                        injectivity_radius = (systole - 1) // 2
                        candidate.update(
                            {
                                "admissible": True,
                                "genus": 1 + group.order,
                                "word_systole_exact": systole,
                                "shortest_kernel_word": json.dumps(list(witness)),
                                "injectivity_radius_integer": injectivity_radius,
                                "injectivity_radius_word": systole / 2.0,
                                "product_automaton_states_visited": product_states,
                                "systole_elapsed_seconds": time.perf_counter() - systole_started,
                                "retained_sector": "kernel_of_conditional_expectation_to_immediate_depth",
                                "interaction_cutoff_radius": 1,
                                "action_path": action_path.relative_to(EXTENSION_ROOT).as_posix(),
                                "action_sha256": artifact["sha256"],
                                "action_bytes": artifact["bytes"],
                            }
                        )
                        admissible_rows.append(dict(candidate))
                if depth in candidate_depths:
                    candidate_rows.append(candidate)
                    write_json(
                        exact_dir / f"candidate_{tower_id}_depth_{depth}.json",
                        {"run_id": run_id, **candidate},
                    )
                parent = group

        candidate_frame = pd.DataFrame(candidate_rows)
        admissible_frame = pd.DataFrame(admissible_rows)
        candidate_frame.to_parquet(run_dir / "derived" / "r8_01_all_candidates.parquet", index=False)
        admissible_frame.to_parquet(run_dir / "derived" / "r8_01_admissible_levels.parquet", index=False)
        tower_counts = (
            admissible_frame.groupby("tower_id").size().astype(int).to_dict()
            if not admissible_frame.empty
            else {}
        )
        minimum_met = any(count >= 4 for count in tower_counts.values())
        preferred_met = sum(count >= 3 for count in tower_counts.values()) >= 2
        task_status = "PASS_CERTIFIED" if minimum_met else "INCONCLUSIVE"
        certificate = {
            "task_id": "R8-01",
            "run_id": run_id,
            "status": task_status,
            "parent_verification": identity["parent_verification"],
            "predecessor_preregistration_sha256": expected_predecessor_hash,
            "recovery_preregistration_sha256": sha256_file(
                EXTENSION_ROOT / "configs" / "r8_cover_recovery_preregistration.yaml"
            ),
            "kbmag": {
                "gap_version": acceptor.gap_version,
                "state_count": acceptor.state_count,
                "accepting_state_count": len(acceptor.accepting),
                "word_acceptor_sha256": sha256_file(acceptor_path),
            },
            "candidate_depths": candidate_depths,
            "admissible_level_counts": tower_counts,
            "minimum_target_met": minimum_met,
            "preferred_target_met": preferred_met,
            "resource_limits": resource_limits,
            "all_admissible_actions_hashed": bool(len(admissible_frame))
            and bool(admissible_frame["action_sha256"].notna().all()),
            "duplicate_candidates_preserved": int(candidate_frame["duplicate_kernel"].sum()),
        }
        write_json(run_dir / "certificates" / "r8_01_cover_depth_extension.json", certificate)
        write_json(
            run_dir / "reports" / "r8_01_summary.json",
            {
                "run_id": run_id,
                "status": task_status,
                "admissible_level_counts": tower_counts,
                "minimum_target_met": minimum_met,
                "preferred_target_met": preferred_met,
                "resource_limit_count": len(resource_limits),
            },
        )
        finalize_run(run_dir, "COMPLETE", {"R8-01": task_status})
        print(json.dumps({"run_id": run_id, "status": task_status, "tower_counts": tower_counts}))
        return 0
    except Exception as error:
        failure = {
            "task_id": "R8-01",
            "run_id": run_id,
            "status": "FAIL_IMPLEMENTATION",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "certificates" / "r8_01_failure.json", failure)
        finalize_run(run_dir, "INCOMPLETE", {"R8-01": "FAIL_IMPLEMENTATION"})
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
