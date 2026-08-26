from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from external.reproduce_hyperbloch_dos import GraphParseError, PARSER_SCHEMA_VERSION, parse_graph


ROOT = Path(__file__).resolve().parents[1]
REVISION = "b13cc279bea13dda81abdfca880abad05da2565d"
EXAMPLES = ROOT / "public_data" / "HyperBloch" / REVISION / "repo" / "Paclet" / "Resources" / "ExampleData"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def hyperbloch_hcm_fixture() -> Path:
    path = EXAMPLES / "{8,3}-tess_T2.1_3.hcm"
    assert digest(path) == "a4c46e7288080cf05fab8cc79ff6f9322d16a272c078a35979fbbc59587d94a6"
    return path


@pytest.fixture
def hyperbloch_hcs_fixture() -> Path:
    path = EXAMPLES / "{8,3}-tess_T2.1_3_sc-T5.1.hcs"
    assert digest(path) == "c03c3f8ec440e0e2872191deb19d4cacf10de75a2cae9dfa93550d42fd2c7200"
    return path


def test_known_hcm_fixture_has_exact_schema_and_24_edges(hyperbloch_hcm_fixture: Path):
    adjacency, edges, diagnostics = parse_graph(hyperbloch_hcm_fixture, return_diagnostics=True)
    assert PARSER_SCHEMA_VERSION == "hypercells_tess_content_v1"
    assert diagnostics["tess_marker_line"] == 7
    assert diagnostics["vertex_record_line"] == 8
    assert diagnostics["edge_record_line"] == 9
    assert adjacency.shape == (16, 16)
    assert len(edges) == 24
    assert np.array_equal(adjacency, adjacency.T)
    assert np.all(np.sum(adjacency, axis=1) == 3)


def test_known_hcs_fixture_distinguishes_metadata_and_has_96_edges(hyperbloch_hcs_fixture: Path):
    adjacency, edges, diagnostics = parse_graph(hyperbloch_hcs_fixture, return_diagnostics=True)
    assert diagnostics["tess_marker_line"] == 12
    assert diagnostics["vertex_record_line"] == 13
    assert diagnostics["edge_record_line"] == 15
    assert any(row["line_number"] == 14 and row["kind"] == "auxiliary_group_word_record" for row in diagnostics["records"])
    assert adjacency.shape == (64, 64)
    assert len(edges) == 96
    assert np.array_equal(adjacency, adjacency.T)
    assert np.all(np.sum(adjacency, axis=1) == 3)


def test_synthetic_extra_metadata_before_edge_list_is_content_parsed(tmp_path: Path):
    path = tmp_path / "metadata_inserted.hcs"
    path.write_text(
        "\n".join([
            "HyperCells HCS version 1.0",
            '[ "TESS", [ 8, 3 ], [ "VEF", [ [ 2 ], [ 1 ], [ 3 ] ] ] ]',
            "[ [ 2, 1, 1 ], [ 2, 2, 1 ], [ 2, 3, 1 ], [ 2, 4, 1 ] ]",
            "[ 1, g1^-1, g2*g3^-1, (g4*g1^-1)^2 ]",
            "[ [ 1, 2, [ 1, [ [ 1, 1 ], 1, 2 ] ] ], [ 2, 3, [ 1, [ [ 1, 2 ], 2, 3 ] ] ], [ 3, 4, [ 1, [ [ 1, 3 ], 3, 4 ] ] ], [ 4, 1, [ 1, [ [ 1, 4 ], 4, 1 ] ] ] ]",
            "[ 1, 1, 1, 1 ]",
            "[ ]",
        ]),
        encoding="utf-8",
    )
    adjacency, edges, diagnostics = parse_graph(path, return_diagnostics=True)
    assert adjacency.shape == (4, 4)
    assert len(edges) == 4
    assert diagnostics["edge_record_line"] == 5
    assert diagnostics["records"][1]["kind"] == "auxiliary_group_word_record"


@pytest.mark.parametrize(
    "records, message",
    [
        (["[ [ 2, 1 ], [ 2, 2 ] ]", "[ ]"], "nonempty edge record"),
        (["[ [ 2, 1 ], [ 2, 2 ] ]", "[ [ 1, 3, [ 1 ] ] ]"], "outside 1..2"),
        (["[ [ 2, 1 ], [ 2, 2 ] ]", '[ "UNKNOWN", 1 ]', "[ [ 1, 2, [ 1 ] ] ]"], "refusing to skip"),
        (["[ [ 2, 1 ], [ 2, 2 ], [ 2, 3 ] ]", "[ [ 1, 2, [ 1 ] ], [ 2, 3 ] ]"], "malformed mixed edge record"),
    ],
)
def test_malformed_or_ambiguous_records_fail_explicitly(tmp_path: Path, records: list[str], message: str):
    path = tmp_path / "malformed.hcm"
    path.write_text("\n".join(["HyperCells HCM version 1.0", '[ "TESS" ]', *records]), encoding="utf-8")
    with pytest.raises(GraphParseError, match=message):
        parse_graph(path)


def test_d09_recovery_contract_is_frozen_before_rerun():
    config = yaml.safe_load((ROOT / "configs" / "phase_d09_parser_recovery.yaml").read_text(encoding="utf-8"))
    contract = config["parser_contract"]
    assert contract["frozen_before_scientific_rerun"] is True
    assert contract["parser_schema_version"] == PARSER_SCHEMA_VERSION
    assert contract["known_fixture_edge_counts"] == {
        "{8,3}-tess_T2.1_3.hcm": 24,
        "{8,3}-tess_T2.1_3_sc-T5.1.hcs": 96,
    }
    assert contract["unknown_record_policy"] == "FAIL_EXPLICITLY"
    assert config["execution_order"][:2] == ["D-09", "D-10"]


def test_d14_workbook_requires_reconciled_summary_and_readable_dates():
    builder = (ROOT / "src" / "audit" / "theorem_validation_workbook.mjs").read_text(encoding="utf-8")
    acceptance = (ROOT / "src" / "audit" / "validation_matrix.py").read_text(encoding="utf-8")
    assert '"PASS*"' not in builder
    assert "passStatuses" in builder
    assert "summaryReconciled" in builder
    assert 'numberFormat = "yyyy-mm-dd hh:mm:ss"' in builder
    assert 'audit.get("summary_reconciled") is True' in acceptance
    assert 'audit.get("provenance_date_number_format")' in acceptance
