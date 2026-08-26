from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root)]
    from audit.run_manifest import sha256_file
    from bulk.finite_cover_model import load_action
    from representation.compact_conjugacy import compact_character_data

    config = yaml.safe_load(
        (root / "configs" / "phase_b_b07_character_recovery.yaml").read_text(encoding="utf-8")
    )
    source_run = root / "results" / str(config["character_recovery_checkpoint"]["source_run_id"])
    source_group = source_run / "raw" / "representation" / "congruence_p7_r2_level_2"
    table = json.loads((source_group / "character_table.json").read_text(encoding="utf-8"))
    workspace = source_group / "character_table.workspace"
    action = (
        root
        / str(config["tower_gate"]["raw_directory"])
        / "congruence_p7_r2_level_2.npz"
    )
    permutations, _metadata = load_action(action)
    work = root / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compact_p7_l2_smoke_", dir=work) as temporary:
        base = Path(temporary)
        raw_dir = base / "raw"
        log_dir = base / "logs"
        raw_dir.mkdir()
        log_dir.mkdir()
        state = raw_dir / "stage_state.json"
        state.write_text(
            json.dumps(
                {
                    "task_id": "B-07",
                    "run_id": "SMOKE_COMPACT_P7_L2",
                    "current_stage": "compact_character_alignment",
                    "last_completed_irrep": 0,
                    "last_completed_block": 0,
                }
            ),
            encoding="utf-8",
        )
        class_map, characters, metadata_path = compact_character_data(
            action_path=action,
            permutations=permutations,
            workspace=workspace,
            raw_dir=raw_dir,
            log_dir=log_dir,
            state_path=state,
            group_name="congruence_p7_r2_level_2",
            order=int(table["order"]),
            degrees=[int(value) for value in table["degrees"]],
            config=config,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        peak = int(metadata["profile"]["peak_job_memory_bytes"])
        maximum = int(config["gap_backend"]["recovery_timeout_policy"]["maximum_peak_memory_bytes"])
        summary = {
            "status": "PASS" if peak < maximum else "FAIL",
            "order": len(class_map),
            "class_count": int(class_map.max()) + 1,
            "character_count": len(characters),
            "peak_job_memory_bytes": peak,
            "maximum_job_memory_bytes": maximum,
            "compact_classes_sha256": metadata["compact_classes_sha256"],
            "alignment_sha256": metadata["raw_sha256"],
            "workspace_sha256": sha256_file(workspace),
        }
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
