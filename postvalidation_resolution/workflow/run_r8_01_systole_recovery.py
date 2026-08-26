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
from word_automaton_v3 import export_word_acceptor, shortest_kernel_normal_word  # noqa: E402


PREDECESSOR_RUN_ID = "21b8f68592ae722b0f112754974e09cea7c38caa8b12bf75f9afc1a9276316ac"


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def save_corrected_action(
    target: Path,
    source: Path,
    *,
    run_id: str,
    source_sha256: str,
    systole: int,
    witness: tuple[int, ...],
) -> None:
    if target.exists():
        raise FileExistsError(target)
    with np.load(source, allow_pickle=False) as payload:
        retained = {
            name: payload[name]
            for name in payload.files
            if name
            not in {
                "run_id",
                "word_systole_exact",
                "shortest_kernel_word",
            }
        }
    retained.update(
        {
            "run_id": np.asarray(run_id),
            "source_action_sha256": np.asarray(source_sha256),
            "word_systole_exact": np.asarray(systole, dtype=np.int64),
            "shortest_kernel_word": np.asarray(witness, dtype=np.int8),
        }
    )
    with target.open("xb") as handle:
        np.savez(handle, **retained)


def main() -> int:
    extension = load_yaml(EXTENSION_ROOT / "configs" / "extension.yaml")
    recovery = load_yaml(
        EXTENSION_ROOT / "configs" / "r8_systole_mapping_recovery_preregistration.yaml"
    )
    predecessor = EXTENSION_ROOT / "results" / PREDECESSOR_RUN_ID
    anchors = recovery["invalid_predecessor_run_preserved"]
    expected = {
        predecessor / "manifest.json": str(anchors["manifest_sha256"]),
        predecessor / "certificates" / "r8_01_cover_depth_extension.json": str(
            anchors["certificate_sha256"]
        ),
        predecessor / "derived" / "r8_01_admissible_levels.parquet": str(
            anchors["admissible_table_sha256"]
        ),
    }
    mismatches = {
        path.as_posix(): {"expected": digest, "actual": sha256_file(path)}
        for path, digest in expected.items()
        if sha256_file(path) != digest
    }
    if mismatches:
        raise RuntimeError(f"predecessor hash mismatch: {mismatches}")

    run_id, run_dir, identity = initialize_run(EXTENSION_ROOT)
    try:
        progress = run_dir / "logs" / "r8_01_systole_progress.jsonl"
        acceptor_path = run_dir / "exact" / "kbmag_shortlex_word_acceptor_v3.txt"
        acceptor = export_word_acceptor(
            EXTENSION_ROOT,
            acceptor_path,
            gap_bash=str(extension["gap_backend"]["gap_bash"]),
            gap_binary_cygwin=str(extension["gap_backend"]["gap_binary_cygwin"]),
            stdout_log=run_dir / "logs" / "kbmag_word_acceptor_v3.stdout.log",
            stderr_log=run_dir / "logs" / "kbmag_word_acceptor_v3.stderr.log",
        )
        raw_actions = run_dir / "raw" / "cover_actions"
        raw_actions.mkdir(parents=True, exist_ok=False)
        source_frame = pd.read_parquet(predecessor / "derived" / "r8_01_admissible_levels.parquet")
        invalid_columns = {
            "word_systole_exact",
            "shortest_kernel_word",
            "injectivity_radius_integer",
            "injectivity_radius_word",
            "product_automaton_states_visited",
            "systole_elapsed_seconds",
            "action_path",
            "action_sha256",
            "action_bytes",
        }
        records: list[dict[str, object]] = []
        cap = int(recovery["systole_rule"]["maximum_product_states"])
        for _, source_row in source_frame.sort_values(["tower_id", "dyadic_depth"]).iterrows():
            tower_id = str(source_row["tower_id"])
            depth = int(source_row["dyadic_depth"])
            source_action = EXTENSION_ROOT / str(source_row["action_path"])
            actual_source_hash = sha256_file(source_action)
            expected_source_hash = str(source_row["action_sha256"])
            if actual_source_hash != expected_source_hash:
                raise RuntimeError(f"source action hash mismatch for {tower_id} depth {depth}")
            started = time.perf_counter()
            with np.load(source_action, allow_pickle=False) as payload:
                permutations = payload["permutations"]
            systole, witness, visited = shortest_kernel_normal_word(
                permutations,
                acceptor,
                maximum_product_states=cap,
            )
            elapsed = time.perf_counter() - started
            target = raw_actions / f"{tower_id}_depth_{depth}.npz"
            save_corrected_action(
                target,
                source_action,
                run_id=run_id,
                source_sha256=actual_source_hash,
                systole=systole,
                witness=witness,
            )
            record = {
                key: source_row[key]
                for key in source_frame.columns
                if key not in invalid_columns
            }
            record.update(
                {
                    "word_systole_exact": systole,
                    "shortest_kernel_word": json.dumps(list(witness)),
                    "injectivity_radius_integer": (systole - 1) // 2,
                    "injectivity_radius_word": systole / 2.0,
                    "product_automaton_states_visited": visited,
                    "systole_elapsed_seconds": elapsed,
                    "source_action_path": source_action.relative_to(EXTENSION_ROOT).as_posix(),
                    "source_action_sha256": actual_source_hash,
                    "action_path": target.relative_to(EXTENSION_ROOT).as_posix(),
                    "action_sha256": sha256_file(target),
                    "action_bytes": target.stat().st_size,
                }
            )
            records.append(record)
            write_json(
                run_dir / "exact" / f"systole_{tower_id}_depth_{depth}.json",
                {"run_id": run_id, **record},
            )
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "tower_id": tower_id,
                            "depth": depth,
                            "systole": systole,
                            "states_visited": visited,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    + "\n"
                )
                handle.flush()

        frame = pd.DataFrame(records)
        frame.to_parquet(run_dir / "derived" / "r8_01_certified_levels.parquet", index=False)
        counts = frame.groupby("tower_id").size().astype(int).to_dict()
        within_tower_growth = {
            tower_id: {
                "systoles": group.sort_values("dyadic_depth")["word_systole_exact"].astype(int).tolist(),
                "injectivity_radii": group.sort_values("dyadic_depth")[
                    "injectivity_radius_integer"
                ].astype(int).tolist(),
                "tail_maximum_increases": bool(
                    group.sort_values("dyadic_depth")["word_systole_exact"].iloc[-1]
                    > group.sort_values("dyadic_depth")["word_systole_exact"].iloc[0]
                ),
            }
            for tower_id, group in frame.groupby("tower_id")
        }
        minimum_met = any(value >= 4 for value in counts.values())
        preferred_met = sum(value >= 3 for value in counts.values()) >= 2
        status = "PASS_CERTIFIED" if minimum_met else "INCONCLUSIVE"
        certificate = {
            "task_id": "R8-01",
            "run_id": run_id,
            "status": status,
            "supersedes_extension_run": PREDECESSOR_RUN_ID,
            "supersession_reason": "corrected KBMAG inverse-letter mapping",
            "parent_verification": identity["parent_verification"],
            "predecessor_hashes_verified": True,
            "systole_mapping_preregistration_sha256": sha256_file(
                EXTENSION_ROOT / "configs" / "r8_systole_mapping_recovery_preregistration.yaml"
            ),
            "word_acceptor_sha256": sha256_file(acceptor_path),
            "word_acceptor_state_count": acceptor.state_count,
            "canonical_signed_alphabet": list(acceptor.alphabet_letters),
            "admissible_level_counts": counts,
            "minimum_target_met": minimum_met,
            "preferred_target_met": preferred_met,
            "within_tower_growth": within_tower_growth,
            "all_corrected_action_hashes_recorded": bool(frame["action_sha256"].notna().all()),
        }
        write_json(run_dir / "certificates" / "r8_01_corrected_cover_depth_extension.json", certificate)
        finalize_run(run_dir, "COMPLETE", {"R8-01": status})
        print(json.dumps({"run_id": run_id, "status": status, "growth": within_tower_growth}))
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
        write_json(run_dir / "certificates" / "r8_01_systole_recovery_failure.json", failure)
        finalize_run(run_dir, "INCOMPLETE", {"R8-01": "FAIL_IMPLEMENTATION"})
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
