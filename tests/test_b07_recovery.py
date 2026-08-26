from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from representation.gap_job_runner import StageTimeoutError
from representation.gap_job_runner_recovery import run_streamed_adaptive
from representation.lazy_word_representation import LazyWordRepresentation, reduce_word
from representation import wedderburn_resumable as implementation
from representation import wedderburn_b07_recovery as recovery
from representation.wedderburn_resumable_repsn_v2 import repsn_irrep_script


def test_lazy_words_reduce_evaluate_and_cache(tmp_path: Path) -> None:
    block = tmp_path / "block.h5"
    identity = np.eye(2, dtype=np.complex128)
    generators = {
        1: np.array([[0, 1], [1, 0]], dtype=np.complex128),
        2: np.diag([1, -1]).astype(np.complex128),
        3: identity,
        4: identity,
    }
    with h5py.File(block, "w") as handle:
        for index, matrix in generators.items():
            handle.create_dataset(f"generator_{index}", data=matrix)
            handle.create_dataset(f"generator_{index}_inverse", data=np.linalg.inv(matrix))
    lazy = LazyWordRepresentation.from_block(block, maximum_cached_words=3)
    assert reduce_word((1, -1, 2)) == (2,)
    value = lazy.evaluate((1, 2, -1))
    assert np.allclose(value, generators[1] @ generators[2] @ generators[1])
    assert lazy.evaluate((1, 2, -1)) is value
    assert lazy.materialized_group_element_count == 8
    assert (1, 2, -1) in lazy.cached_words


def test_monotone_state_update_preserves_seeded_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"last_completed_irrep": 3, "last_completed_block": 3}), encoding="utf-8")
    recovery._state_update_monotone(path, last_completed_irrep=0, last_completed_block=0)
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["last_completed_irrep"] == 3
    assert state["last_completed_block"] == 3


def test_repsn_script_materializes_generators_only(tmp_path: Path) -> None:
    script = repsn_irrep_script(4, tmp_path / "irrep.part")
    assert "for g in [1..Length(B07_GENS)]" in script
    assert "Image(rep,B07_GENS[g])" in script
    assert "Image(rep,B07_GENS[g]^-1)" in script
    assert "Elements(B07_G)" not in script
    assert "for g in B07_G" not in script


@pytest.mark.skipif(sys.platform != "win32", reason="Job Object CPU accounting is Windows-specific")
def test_adaptive_timeout_extends_on_cpu_progress(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import time; print('STAGE_BEGIN=test', flush=True); t=time.time()+0.8; x=0; "
        "exec(\"while time.time()<t:\\n x+=1\"); print('STAGE_COMPLETE=test', flush=True)",
    ]
    result = run_streamed_adaptive(
        command,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        heartbeat_path=tmp_path / "heartbeat.json",
        extension_path=tmp_path / "extension.json",
        soft_timeout_seconds=0.3,
        hard_timeout_seconds=2.0,
        heartbeat_seconds=0.05,
        progress_lookback_seconds=0.1,
        minimum_cpu_fraction=0.01,
    )
    assert result.timeout_extended
    assert result.returncode == 0
    decision = json.loads((tmp_path / "extension.json").read_text(encoding="utf-8"))
    assert decision["extended"] is True


@pytest.mark.skipif(sys.platform != "win32", reason="Job Object CPU accounting is Windows-specific")
def test_adaptive_hard_timeout_terminates_tree(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import time; print('STAGE_BEGIN=test', flush=True); t=time.time()+10; x=0; "
        "exec(\"while time.time()<t:\\n x+=1\")",
    ]
    with pytest.raises(StageTimeoutError) as caught:
        run_streamed_adaptive(
            command,
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
            heartbeat_path=tmp_path / "heartbeat.json",
            extension_path=tmp_path / "extension.json",
            soft_timeout_seconds=0.3,
            hard_timeout_seconds=0.7,
            heartbeat_seconds=0.05,
            progress_lookback_seconds=0.1,
            minimum_cpu_fraction=0.01,
        )
    assert caught.value.result.timeout_extended
    assert caught.value.result.timeout_boundary == "hard"
    assert caught.value.result.tree_terminated


def test_recovery_install_selects_all_recovery_hooks() -> None:
    recovery.install()
    assert implementation._irrep_script is repsn_irrep_script
    assert implementation._execute_gap_stage is recovery._execute_gap_stage_recovery
    assert implementation._state_update is recovery._state_update_monotone
    assert implementation._build_block is recovery._build_block_generators_only


def test_b07_certificate_metadata_is_included_in_single_immutable_write(tmp_path: Path) -> None:
    (tmp_path / "certificates").mkdir()
    eigenvalue = np.sqrt(8.0)
    frame = pd.DataFrame(
        [
            {
                "tower_id": "test_tower",
                "level": 1,
                "rep_index": 1,
                "degree": 1,
                "adjacency_eigenvalue": -eigenvalue,
                "regular_multiplicity": 1,
            },
            {
                "tower_id": "test_tower",
                "level": 1,
                "rep_index": 2,
                "degree": 1,
                "adjacency_eigenvalue": eigenvalue,
                "regular_multiplicity": 1,
            },
        ]
    )
    context = {
        "blocks": frame,
        "wedderburn_diagnostics": [
            {
                "tower_id": "test_tower",
                "level": 1,
                "order": 2,
                "degree_square_identity": True,
            }
        ],
        "wedderburn_outputs": {
            "raw": tmp_path / "raw",
            "derived": tmp_path / "derived.parquet",
        },
        "certificate_metadata": {"recovery_route": "exact_character_isotypic_newton"},
    }
    status, outputs = implementation.run({}, tmp_path, "test-run", tmp_path, context)
    payload = json.loads(outputs["certificate"].read_text(encoding="utf-8"))
    assert status == "PASS_CERTIFIED"
    assert payload["recovery_route"] == "exact_character_isotypic_newton"
    assert payload["status"] == "PASS_CERTIFIED"
