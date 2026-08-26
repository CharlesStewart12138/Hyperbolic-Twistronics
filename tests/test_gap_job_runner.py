from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

from representation.gap_job_runner import StageTimeoutError, run_streamed
from representation.wedderburn_resumable import _build_block, _valid_block, _valid_irrep, _irrep_script


def process_has_exited(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False
    synchronize = 0x00100000
    wait_object_0 = 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return True
    try:
        return kernel32.WaitForSingleObject(handle, 0) == wait_object_0
    finally:
        kernel32.CloseHandle(handle)


def test_streamed_logs_and_heartbeat(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    heartbeat = tmp_path / "heartbeat.json"
    code = (
        "import sys,time; "
        "print('alpha', flush=True); "
        "print('beta', file=sys.stderr, flush=True); "
        "time.sleep(0.3); print('omega', flush=True)"
    )
    result = run_streamed(
        [sys.executable, "-c", code],
        stdout_path=stdout,
        stderr_path=stderr,
        heartbeat_path=heartbeat,
        timeout_seconds=5,
        heartbeat_seconds=0.05,
        stage_metadata={"stage": "stream-regression"},
    )
    assert result.returncode == 0
    assert result.timed_out is False
    assert "alpha" in stdout.read_text(encoding="utf-8")
    assert "omega" in stdout.read_text(encoding="utf-8")
    assert "beta" in stderr.read_text(encoding="utf-8")
    metadata = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert metadata["state"] == "COMPLETE"
    assert metadata["stage"] == "stream-regression"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_streamed_job_memory_cap_allows_controlled_process(tmp_path: Path) -> None:
    stdout = tmp_path / "memory.stdout.log"
    stderr = tmp_path / "memory.stderr.log"
    heartbeat = tmp_path / "memory.heartbeat.json"
    maximum = 256 * 1024 * 1024
    result = run_streamed(
        [sys.executable, "-c", "value=bytearray(1024*1024); print(len(value), flush=True)"],
        stdout_path=stdout,
        stderr_path=stderr,
        heartbeat_path=heartbeat,
        timeout_seconds=5,
        heartbeat_seconds=0.05,
        stage_metadata={"stage": "memory-cap-regression"},
        maximum_job_memory_bytes=maximum,
    )
    assert result.returncode == 0
    assert result.peak_job_memory_bytes < maximum
    metadata = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert metadata["state"] == "COMPLETE"
    assert metadata["peak_job_memory_bytes"] < maximum


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_timeout_terminates_complete_descendant_tree(tmp_path: Path) -> None:
    stdout = tmp_path / "tree.stdout.log"
    stderr = tmp_path / "tree.stderr.log"
    heartbeat = tmp_path / "tree.heartbeat.json"
    child_pid_file = tmp_path / "child.pid"
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); "
        "print('CHILD_PID='+str(p.pid), flush=True); time.sleep(60)"
    )
    with pytest.raises(StageTimeoutError) as caught:
        run_streamed(
            [sys.executable, "-c", parent_code],
            stdout_path=stdout,
            stderr_path=stderr,
            heartbeat_path=heartbeat,
            timeout_seconds=0.8,
            heartbeat_seconds=0.05,
            stage_metadata={"stage": "tree-timeout-regression"},
        )
    result = caught.value.result
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    assert result.timed_out is True
    assert result.tree_terminated is True
    assert result.total_processes >= 2
    assert process_has_exited(result.pid)
    assert process_has_exited(child_pid)
    metadata = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert metadata["state"] == "TIMEOUT"
    assert metadata["active_processes"] == 0


def test_irrep_artifact_and_block_are_restart_valid(tmp_path: Path) -> None:
    irrep = tmp_path / "irrep_0001.txt"
    lines = ["REP_BEGIN index=1 degree=1"]
    for generator in range(1, 5):
        lines.append(f"TRACE_CHECK generator={generator} equal=true")
        lines.append(
            f"GEN_ENTRY rep=1 generator={generator} inverse=false row=1 col=1 value=1"
        )
        lines.append(
            f"GEN_ENTRY rep=1 generator={generator} inverse=true row=1 col=1 value=1"
        )
    lines.append("REP_END")
    irrep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    block = tmp_path / "block_0001.h5"
    action_hash = "a" * 64
    record = _build_block(irrep, block, 1, action_hash)
    assert _valid_irrep(irrep, 1)
    assert _valid_block(block, 1, action_hash)
    assert record["degree"] == 1
    with h5py.File(block, "r") as handle:
        assert handle["adjacency_eigenvalues"][0] == pytest.approx(8.0)
        assert bool(handle.attrs["trace_checks_passed"])


def test_irrep_gap_stage_is_singular_and_atomic(tmp_path: Path) -> None:
    script = _irrep_script(7, tmp_path / "irrep_0007.part")
    assert "IrreducibleRepresentationsDixon(B07_G,B07_IRR[idx])" in script
    assert "REP_BEGIN" in script
    assert "TRACE_CHECK" in script
    assert "REP_END" in script
    assert "IrreducibleRepresentationsDixon(B07_G);" not in script
