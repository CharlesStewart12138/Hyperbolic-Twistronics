from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence


CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def tail_line(path: Path, maximum_bytes: int = 8192) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - maximum_bytes))
        data = handle.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-1] if lines else ""


if os.name == "nt":
    ULONG_PTR = ctypes.c_size_t
    SIZE_T = ctypes.c_size_t

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", SIZE_T),
            ("MaximumWorkingSetSize", SIZE_T),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ULONG_PTR),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", SIZE_T),
            ("JobMemoryLimit", SIZE_T),
            ("PeakProcessMemoryUsed", SIZE_T),
            ("PeakJobMemoryUsed", SIZE_T),
        ]

    class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    class JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicInfo", JOBOBJECT_BASIC_ACCOUNTING_INFORMATION),
            ("IoInfo", IO_COUNTERS),
        ]


class WindowsJob:
    def __init__(self, maximum_job_memory_bytes: int | None = None) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects are available only on Windows")
        if maximum_job_memory_bytes is not None and maximum_job_memory_bytes <= 0:
            raise ValueError("maximum Job memory must be positive")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32 = kernel32
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if maximum_job_memory_bytes is not None:
            limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY
            limits.JobMemoryLimit = maximum_job_memory_bytes
        if not kernel32.SetInformationJobObject(
            self.handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel32.CloseHandle(self.handle)
            self.handle = None
            raise error

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        if not self._kernel32.AssignProcessToJobObject(self.handle, wintypes.HANDLE(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def stats(self) -> dict[str, int | float]:
        accounting = JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION()
        returned = wintypes.DWORD()
        if not self._kernel32.QueryInformationJobObject(
            self.handle,
            JOB_OBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            ctypes.byref(returned),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        if not self._kernel32.QueryInformationJobObject(
            self.handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
            ctypes.byref(returned),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        basic = accounting.BasicInfo
        return {
            "active_processes": int(basic.ActiveProcesses),
            "total_processes": int(basic.TotalProcesses),
            "terminated_processes": int(basic.TotalTerminatedProcesses),
            "cpu_time_seconds": float(basic.TotalUserTime + basic.TotalKernelTime) / 10_000_000.0,
            "peak_process_memory_bytes": int(limits.PeakProcessMemoryUsed),
            "peak_job_memory_bytes": int(limits.PeakJobMemoryUsed),
        }

    def terminate(self, exit_code: int = 124) -> None:
        if self.handle and not self._kernel32.TerminateJobObject(self.handle, exit_code):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self._kernel32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> "WindowsJob":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class StageResult:
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class StageExecutionError(RuntimeError):
    def __init__(self, message: str, result: StageResult):
        super().__init__(message)
        self.result = result


class StageTimeoutError(StageExecutionError):
    pass


ProgressReader = Callable[[], Mapping[str, object]]


def _write_heartbeat(
    path: Path,
    *,
    command: Sequence[str],
    pid: int,
    stage_metadata: Mapping[str, object],
    started_at: str,
    elapsed: float,
    stdout_path: Path,
    stderr_path: Path,
    job_stats: Mapping[str, object],
    state: str,
    progress_reader: ProgressReader | None,
) -> None:
    progress: Mapping[str, object] = {}
    if progress_reader is not None:
        try:
            progress = dict(progress_reader())
        except Exception as error:  # heartbeat collection must never kill science work
            progress = {"progress_reader_error": repr(error)}
    payload = {
        **dict(stage_metadata),
        "state": state,
        "command": list(command),
        "pid": pid,
        "started_at_utc": started_at,
        "updated_at_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_bytes": stdout_path.stat().st_size if stdout_path.exists() else 0,
        "stderr_bytes": stderr_path.stat().st_size if stderr_path.exists() else 0,
        "last_stdout_line": tail_line(stdout_path),
        "last_stderr_line": tail_line(stderr_path),
        **dict(job_stats),
        **dict(progress),
    }
    atomic_json(path, payload)


def run_streamed(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    heartbeat_path: Path,
    timeout_seconds: float,
    heartbeat_seconds: float = 10.0,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stage_metadata: Mapping[str, object] | None = None,
    progress_reader: ProgressReader | None = None,
    maximum_job_memory_bytes: int | None = None,
) -> StageResult:
    if timeout_seconds <= 0 or heartbeat_seconds <= 0:
        raise ValueError("timeouts and heartbeat intervals must be positive")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage_metadata or {})
    started_at = utc_now()
    start = time.monotonic()
    job: WindowsJob | None = None
    process: subprocess.Popen[bytes] | None = None
    final_stats: dict[str, int | float] = {
        "active_processes": 0,
        "total_processes": 0,
        "terminated_processes": 0,
        "cpu_time_seconds": 0.0,
        "peak_process_memory_bytes": 0,
        "peak_job_memory_bytes": 0,
    }
    timed_out = False
    tree_terminated = False
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
                job = WindowsJob(maximum_job_memory_bytes=maximum_job_memory_bytes)
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
                if returncode is not None:
                    break
                if elapsed >= timeout_seconds:
                    timed_out = True
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
                    stage_metadata=metadata,
                    started_at=started_at,
                    elapsed=elapsed,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    job_stats=final_stats,
                    state="RUNNING",
                    progress_reader=progress_reader,
                )
                time.sleep(min(heartbeat_seconds, max(0.01, timeout_seconds - elapsed)))
            if job is not None:
                final_stats = job.stats()
            returncode = int(process.returncode if process.returncode is not None else 124)
        finally:
            if job is not None:
                job.close()
    elapsed = time.monotonic() - start
    result = StageResult(
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
    )
    _write_heartbeat(
        heartbeat_path,
        command=command,
        pid=process.pid,
        stage_metadata=metadata,
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
            f"stage timed out after {timeout_seconds} seconds; descendant tree terminated={result.tree_terminated}",
            result,
        )
    if returncode != 0:
        raise StageExecutionError(f"stage exited with return code {returncode}", result)
    return result
