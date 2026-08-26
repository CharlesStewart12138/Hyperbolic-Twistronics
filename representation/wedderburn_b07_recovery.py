from __future__ import annotations

import json
from pathlib import Path

import h5py

from audit.run_manifest import sha256_file
from representation import wedderburn_resumable as implementation
from representation.gap_job_runner import StageExecutionError
from representation.gap_job_runner_recovery import run_streamed_adaptive
from representation.wedderburn_resumable import B07StageFailure
from representation.wedderburn_resumable_repsn_v2 import repsn_irrep_script


_ORIGINAL_EXECUTE = implementation._execute_gap_stage
_ORIGINAL_STATE_UPDATE = implementation._state_update
_ORIGINAL_BUILD_BLOCK = implementation._build_block
_ORIGINAL_FAILURE_PAYLOAD = implementation._failure_payload


def _state_update_monotone(state_path: Path, **updates: object) -> dict[str, object]:
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        for field in ("last_completed_irrep", "last_completed_block"):
            if field in updates:
                updates[field] = max(int(existing.get(field, 0)), int(updates[field]))
    return _ORIGINAL_STATE_UPDATE(state_path, **updates)


def _record_history(log_dir: Path, stage: str, group_name: str, attempt: int, result) -> None:
    history = log_dir / "stage_history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": stage, "group": group_name, "attempt": attempt, **result.to_dict()},
                sort_keys=True,
            )
            + "\n"
        )


def _execute_gap_stage_recovery(
    *,
    config: dict[str, object],
    script_text: str,
    stage: str,
    group_name: str,
    log_dir: Path,
    state_path: Path,
    timeout_seconds: int,
    workspace: Path | None = None,
):
    if not stage.startswith("irrep_"):
        return _ORIGINAL_EXECUTE(
            config=config,
            script_text=script_text,
            stage=stage,
            group_name=group_name,
            log_dir=log_dir,
            state_path=state_path,
            timeout_seconds=timeout_seconds,
            workspace=workspace,
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    attempt = implementation._attempt_number(log_dir, stage)
    prefix = f"{stage}_attempt_{attempt:03d}"
    script = log_dir / f"{prefix}.g"
    stdout = log_dir / f"{prefix}.stdout.log"
    stderr = log_dir / f"{prefix}.stderr.log"
    heartbeat = log_dir / f"{prefix}.heartbeat.json"
    extension = log_dir / f"{prefix}.timeout_extension.json"
    script.write_text(script_text, encoding="ascii")
    policy = config["gap_backend"]["recovery_timeout_policy"]
    try:
        result = run_streamed_adaptive(
            implementation.gap_command(config, script, workspace),
            stdout_path=stdout,
            stderr_path=stderr,
            heartbeat_path=heartbeat,
            extension_path=extension,
            soft_timeout_seconds=float(policy["soft_seconds"]),
            hard_timeout_seconds=float(policy["hard_seconds"]),
            heartbeat_seconds=float(config["gap_backend"].get("heartbeat_seconds", 10)),
            progress_lookback_seconds=float(policy.get("progress_lookback_seconds", 120)),
            minimum_cpu_fraction=float(policy.get("minimum_cpu_fraction", 0.25)),
            maximum_peak_memory_bytes=int(policy.get("maximum_peak_memory_bytes", 4 * 1024**3)),
            stage_metadata={
                "task_id": "B-07",
                "stage": stage,
                "group": group_name,
                "attempt": attempt,
                "timeout_policy": "7200_soft_to_14400_hard_on_legitimate_progress",
            },
            progress_reader=implementation._checkpoint_reader(state_path),
        )
    except StageExecutionError as error:
        _record_history(log_dir, stage, group_name, attempt, error.result)
        raise
    _record_history(log_dir, stage, group_name, attempt, result)
    return result


def _build_block_generators_only(irrep_path: Path, block_path: Path, index: int, action_hash: str):
    record = _ORIGINAL_BUILD_BLOCK(irrep_path, block_path, index, action_hash)
    with h5py.File(block_path, "r+") as handle:
        handle.attrs["group_element_materialization"] = "four_generators_and_inverses_only"
        handle.attrs["non_generator_evaluation"] = "lazy_from_words_with_bounded_cache"
        handle.attrs["all_group_elements_materialized"] = False
    record["block_sha256"] = sha256_file(block_path)
    return record


def _failure_payload_recovery(**kwargs):
    payload = _ORIGINAL_FAILURE_PAYLOAD(**kwargs)
    error = kwargs["error"]
    result = error.result if isinstance(error, StageExecutionError) else None
    if result is not None:
        payload.update(
            {
                "cpu_time_seconds": result.cpu_time_seconds,
                "soft_timeout_seconds": getattr(result, "soft_timeout_seconds", None),
                "hard_timeout_seconds": getattr(result, "hard_timeout_seconds", None),
                "timeout_extended": getattr(result, "timeout_extended", False),
                "timeout_boundary": getattr(result, "timeout_boundary", "none"),
                "extension_reason": getattr(result, "extension_reason", ""),
            }
        )
        if getattr(result, "timeout_boundary", "none") == "hard":
            payload["required_next_route"] = "character_projector_isotypic_decomposition"
            payload["elapsed_time_alone_is_not_fail_implementation"] = True
    return payload


def install() -> None:
    implementation._irrep_script = repsn_irrep_script
    implementation._execute_gap_stage = _execute_gap_stage_recovery
    implementation._state_update = _state_update_monotone
    implementation._build_block = _build_block_generators_only
    implementation._failure_payload = _failure_payload_recovery


def prepare_wedderburn(*args, **kwargs):
    install()
    return implementation.prepare_wedderburn(*args, **kwargs)


def run(*args, **kwargs):
    install()
    return implementation.run(*args, **kwargs)
