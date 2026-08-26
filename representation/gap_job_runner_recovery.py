from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from representation.gap_job_runner import (
    CREATE_NEW_PROCESS_GROUP,
    CREATE_NO_WINDOW,
    StageExecutionError,
    StageTimeoutError,
    WindowsJob,
    _write_heartbeat,
    atomic_json,
    tail_line,
    utc_now,
)


@dataclass(frozen=True)
class RecoveryStageResult:
    command: list[str]
    pid: int
    returncode: int
    timed_out: bool
    tree_terminated: bool
    elapsed_seconds: float
    started_at_utc: str
    completed_at_utc: str
    stdout_log: str
    stderr_log: str
    heartbeat_file: str
    peak_job_memory_bytes: int
    cpu_time_seconds: float
    total_processes: int
    last_stdout_line: str
    last_stderr_line: str
    soft_timeout_seconds: float
    hard_timeout_seconds: float
    timeout_extended: bool
    extension_reason: str
    timeout_boundary: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _log_is_valid(stdout_path: Path, stderr_path: Path) -> tuple[bool, str]:
    stdout_tail = tail_line(stdout_path).lower()
    stderr_tail = tail_line(stderr_path).lower()
    invalid = ("syntax error", "error,", "fatal", "segmentation fault", "traceback")
    if any(token in stdout_tail or token in stderr_tail for token in invalid):
        return False, "error marker in streamed log tail"
    if not stdout_path.exists() or stdout_path.stat().st_size == 0:
        return False, "stdout contains no stage marker"
    return True, "streamed output remains syntactically valid"


def _extension_decision(
    samples: deque[dict[str, float]],
    *,
    stdout_path: Path,
    stderr_path: Path,
    maximum_peak_memory_bytes: int,
    minimum_cpu_fraction: float,
    lookback_seconds: float,
) -> tuple[bool, dict[str, object]]:
    if not samples:
        return False, {"reason": "no Job Object samples"}
    current = samples[-1]
    candidates = [sample for sample in samples if current["elapsed"] - sample["elapsed"] >= lookback_seconds]
    baseline = candidates[-1] if candidates else samples[0]
    wall_delta = max(1.0, current["elapsed"] - baseline["elapsed"])
    cpu_delta = max(0.0, current["cpu"] - baseline["cpu"])
    cpu_fraction = cpu_delta / wall_delta
    log_valid, log_reason = _log_is_valid(stdout_path, stderr_path)
    checks = {
        "active_processes_positive": current["active"] > 0,
        "cpu_progress": cpu_fraction >= minimum_cpu_fraction,
        "memory_controlled": current["peak_memory"] <= maximum_peak_memory_bytes,
        "output_valid": log_valid,
    }
    payload: dict[str, object] = {
        "checks": checks,
        "lookback_wall_seconds": wall_delta,
        "lookback_cpu_seconds": cpu_delta,
        "observed_cpu_fraction": cpu_fraction,
        "minimum_cpu_fraction": minimum_cpu_fraction,
        "peak_job_memory_bytes": int(current["peak_memory"]),
        "maximum_peak_job_memory_bytes": maximum_peak_memory_bytes,
        "log_assessment": log_reason,
    }
    payload["reason"] = (
        "legitimate CPU-bound progress with controlled memory and valid streamed output"
        if all(checks.values())
        else "soft-budget extension criteria were not all satisfied"
    )
    return all(checks.values()), payload


def run_streamed_adaptive(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    heartbeat_path: Path,
    extension_path: Path,
    soft_timeout_seconds: float,
    hard_timeout_seconds: float,
    heartbeat_seconds: float = 10.0,
    progress_lookback_seconds: float = 120.0,
    minimum_cpu_fraction: float = 0.25,
    maximum_peak_memory_bytes: int = 4 * 1024**3,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stage_metadata: Mapping[str, object] | None = None,
    progress_reader=None,
) -> RecoveryStageResult:
    if not (0 < soft_timeout_seconds <= hard_timeout_seconds):
        raise ValueError("require 0 < soft timeout <= hard timeout")
    if heartbeat_seconds <= 0 or progress_lookback_seconds <= 0:
        raise ValueError("heartbeat and lookback intervals must be positive")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    extension_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage_metadata or {})
    started_at = utc_now()
    start = time.monotonic()
    final_stats: dict[str, int | float] = {
        "active_processes": 0,
        "total_processes": 0,
        "terminated_processes": 0,
        "cpu_time_seconds": 0.0,
        "peak_process_memory_bytes": 0,
        "peak_job_memory_bytes": 0,
    }
    samples: deque[dict[str, float]] = deque(maxlen=max(32, int(progress_lookback_seconds / heartbeat_seconds) + 8))
    timed_out = False
    tree_terminated = False
    timeout_extended = False
    extension_reason = "soft budget not reached"
    timeout_boundary = "none"
    job: WindowsJob | None = None
    process: subprocess.Popen[bytes] | None = None
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
        creationflags = 0
        popen_kwargs: dict[str, object] = {}
        if os.name == "nt":
            creationflags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
            **popen_kwargs,
        )
        try:
            if os.name == "nt":
                job = WindowsJob()
                try:
                    job.assign(process)
                except Exception:
                    process.kill()
                    process.wait(timeout=10)
                    raise
            while True:
                returncode = process.poll()
                elapsed = time.monotonic() - start
                if job is not None:
                    final_stats = job.stats()
                samples.append(
                    {
                        "elapsed": elapsed,
                        "cpu": float(final_stats.get("cpu_time_seconds", 0.0)),
                        "active": float(final_stats.get("active_processes", 1 if returncode is None else 0)),
                        "peak_memory": float(final_stats.get("peak_job_memory_bytes", 0)),
                    }
                )
                if returncode is not None:
                    break
                if not timeout_extended and elapsed >= soft_timeout_seconds:
                    allowed, decision = _extension_decision(
                        samples,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        maximum_peak_memory_bytes=maximum_peak_memory_bytes,
                        minimum_cpu_fraction=minimum_cpu_fraction,
                        lookback_seconds=progress_lookback_seconds,
                    )
                    extension_reason = str(decision["reason"])
                    atomic_json(
                        extension_path,
                        {
                            **metadata,
                            "decision_at_utc": utc_now(),
                            "soft_timeout_seconds": soft_timeout_seconds,
                            "hard_timeout_seconds": hard_timeout_seconds,
                            "extended": allowed,
                            **decision,
                        },
                    )
                    if allowed and hard_timeout_seconds > soft_timeout_seconds:
                        timeout_extended = True
                    else:
                        timed_out = True
                        timeout_boundary = "soft"
                elif timeout_extended and elapsed >= hard_timeout_seconds:
                    timed_out = True
                    timeout_boundary = "hard"
                    extension_reason = "hard recovery budget exhausted after a valid soft-budget extension"
                if timed_out:
                    if job is not None:
                        job.terminate(124)
                        deadline = time.monotonic() + 10.0
                        while time.monotonic() < deadline:
                            final_stats = job.stats()
                            if int(final_stats["active_processes"]) == 0:
                                tree_terminated = True
                                break
                            time.sleep(0.1)
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                        tree_terminated = True
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    break
                _write_heartbeat(
                    heartbeat_path,
                    command=command,
                    pid=process.pid,
                    stage_metadata={
                        **metadata,
                        "soft_timeout_seconds": soft_timeout_seconds,
                        "hard_timeout_seconds": hard_timeout_seconds,
                        "timeout_extended": timeout_extended,
                        "extension_reason": extension_reason,
                    },
                    started_at=started_at,
                    elapsed=elapsed,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    job_stats=final_stats,
                    state="RUNNING_EXTENDED" if timeout_extended else "RUNNING",
                    progress_reader=progress_reader,
                )
                remaining = (hard_timeout_seconds if timeout_extended else soft_timeout_seconds) - elapsed
                time.sleep(min(heartbeat_seconds, max(0.01, remaining)))
            if job is not None:
                final_stats = job.stats()
            returncode = int(process.returncode if process.returncode is not None else 124)
        finally:
            if job is not None:
                job.close()
    elapsed = time.monotonic() - start
    result = RecoveryStageResult(
        command=list(command),
        pid=int(process.pid),
        returncode=returncode,
        timed_out=timed_out,
        tree_terminated=tree_terminated if timed_out else True,
        elapsed_seconds=elapsed,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        heartbeat_file=str(heartbeat_path),
        peak_job_memory_bytes=int(final_stats.get("peak_job_memory_bytes", 0)),
        cpu_time_seconds=float(final_stats.get("cpu_time_seconds", 0.0)),
        total_processes=int(final_stats.get("total_processes", 0)),
        last_stdout_line=tail_line(stdout_path),
        last_stderr_line=tail_line(stderr_path),
        soft_timeout_seconds=soft_timeout_seconds,
        hard_timeout_seconds=hard_timeout_seconds,
        timeout_extended=timeout_extended,
        extension_reason=extension_reason,
        timeout_boundary=timeout_boundary,
    )
    _write_heartbeat(
        heartbeat_path,
        command=command,
        pid=process.pid,
        stage_metadata={**metadata, **result.to_dict()},
        started_at=started_at,
        elapsed=elapsed,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        job_stats={**final_stats, "active_processes": 0 if result.tree_terminated else final_stats.get("active_processes", 0)},
        state="TIMEOUT" if timed_out else ("COMPLETE" if returncode == 0 else "FAILED"),
        progress_reader=progress_reader,
    )
    if timed_out:
        raise StageTimeoutError(
            f"stage exhausted the {timeout_boundary} recovery budget; descendant tree terminated={result.tree_terminated}",
            result,
        )
    if returncode != 0:
        raise StageExecutionError(f"stage exited with return code {returncode}", result)
    return result
