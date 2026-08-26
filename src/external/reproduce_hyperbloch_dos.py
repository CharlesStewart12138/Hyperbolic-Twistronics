from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from audit.data_io import write_json
from audit.run_manifest import sha256_file
from dos.common import gaussian_density


PARSER_SCHEMA_VERSION = "hypercells_tess_content_v1"
KNOWN_EDGE_COUNTS = {
    "{8,3}-tess_T2.1_3.hcm": 24,
    "{8,3}-tess_T2.1_3_sc-T5.1.hcs": 96,
}
_INTEGER = re.compile(r"[+-]?\d+")
_GROUP_WORD = re.compile(r"[A-Za-z0-9_()*^+\-\s]+")


class GraphParseError(ValueError):
    """Raised when a HyperCells graph record is absent, ambiguous, or malformed."""


def _split_top_level_list(record: str, *, path: Path, line_number: int) -> list[str]:
    text = record.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise GraphParseError(f"{path}:{line_number}: record is not a bracketed list")
    inner = text[1:-1].strip()
    if not inner:
        return []
    fields: list[str] = []
    start = 0
    square_depth = 0
    paren_depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(inner):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            square_depth += 1
        elif character == "]":
            square_depth -= 1
            if square_depth < 0:
                raise GraphParseError(f"{path}:{line_number}: unmatched closing bracket")
        elif character == "(":
            paren_depth += 1
        elif character == ")":
            paren_depth -= 1
            if paren_depth < 0:
                raise GraphParseError(f"{path}:{line_number}: unmatched closing parenthesis")
        elif character == "," and square_depth == 0 and paren_depth == 0:
            fields.append(inner[start:index].strip())
            start = index + 1
    if in_string or square_depth != 0 or paren_depth != 0:
        raise GraphParseError(f"{path}:{line_number}: unbalanced record")
    fields.append(inner[start:].strip())
    if any(not field for field in fields):
        raise GraphParseError(f"{path}:{line_number}: empty top-level field")
    return fields


def _integer_tuple(element: str, *, path: Path, line_number: int) -> tuple[int, ...] | None:
    if not (element.strip().startswith("[") and element.strip().endswith("]")):
        return None
    fields = _split_top_level_list(element, path=path, line_number=line_number)
    if not fields or not all(_INTEGER.fullmatch(field) for field in fields):
        return None
    return tuple(int(field) for field in fields)


def _vertex_record(record: str, *, path: Path, line_number: int) -> list[tuple[int, ...]] | None:
    fields = _split_top_level_list(record, path=path, line_number=line_number)
    if not fields:
        return None
    tuples = [_integer_tuple(field, path=path, line_number=line_number) for field in fields]
    parsed = [item for item in tuples if item is not None]
    if not parsed:
        return None
    if len(parsed) != len(fields):
        raise GraphParseError(f"{path}:{line_number}: malformed mixed vertex record")
    widths = {len(item) for item in parsed}
    if len(widths) != 1 or not widths.issubset({2, 3}):
        raise GraphParseError(f"{path}:{line_number}: vertex tuples must have uniform width two or three")
    if any(any(value <= 0 for value in item) for item in parsed):
        raise GraphParseError(f"{path}:{line_number}: vertex labels must be positive")
    return parsed


def _edge_record(record: str, *, path: Path, line_number: int) -> list[tuple[int, int]] | None:
    fields = _split_top_level_list(record, path=path, line_number=line_number)
    if not fields:
        return None
    edges: list[tuple[int, int]] = []
    candidate_count = 0
    for element in fields:
        if not (element.startswith("[") and element.endswith("]")):
            continue
        components = _split_top_level_list(element, path=path, line_number=line_number)
        candidate = (
            len(components) == 3
            and _INTEGER.fullmatch(components[0]) is not None
            and _INTEGER.fullmatch(components[1]) is not None
            and components[2].startswith("[")
            and components[2].endswith("]")
        )
        if not candidate:
            continue
        candidate_count += 1
        _split_top_level_list(components[2], path=path, line_number=line_number)
        first, second = int(components[0]), int(components[1])
        if first <= 0 or second <= 0:
            raise GraphParseError(f"{path}:{line_number}: edge indices must be positive")
        edges.append((first, second))
    if candidate_count == 0:
        return None
    if candidate_count != len(fields):
        raise GraphParseError(f"{path}:{line_number}: malformed mixed edge record")
    return edges


def _group_word_record(record: str, *, path: Path, line_number: int) -> bool:
    fields = _split_top_level_list(record, path=path, line_number=line_number)
    return bool(fields) and all("[" not in field and "]" not in field and _GROUP_WORD.fullmatch(field) for field in fields)


def _face_boundary_record(record: str, *, path: Path, line_number: int) -> bool:
    fields = _split_top_level_list(record, path=path, line_number=line_number)
    if not fields:
        return True
    for face in fields:
        if not (face.startswith("[") and face.endswith("]")):
            return False
        entries = _split_top_level_list(face, path=path, line_number=line_number)
        if not entries:
            return False
        pairs = [_integer_tuple(entry, path=path, line_number=line_number) for entry in entries]
        if any(pair is None or len(pair) != 2 for pair in pairs):
            return False
    return True


def parse_graph(path: Path, *, return_diagnostics: bool = False):
    lines = path.read_text(encoding="utf-8").splitlines()
    markers = [index for index, line in enumerate(lines) if '"TESS"' in line]
    if len(markers) != 1:
        raise GraphParseError(f"{path}: expected exactly one TESS marker, found {len(markers)}")
    marker = markers[0]
    vertices: list[tuple[int, ...]] | None = None
    edges: list[tuple[int, int]] | None = None
    vertex_line_number: int | None = None
    edge_line_number: int | None = None
    records: list[dict[str, Any]] = []
    for index in range(marker + 1, len(lines)):
        record = lines[index].strip()
        if not record:
            continue
        line_number = index + 1
        if not (record.startswith("[") and record.endswith("]")):
            raise GraphParseError(f"{path}:{line_number}: unrecognized non-list record after TESS")
        if not _split_top_level_list(record, path=path, line_number=line_number) and edges is None:
            raise GraphParseError(f"{path}:{line_number}: nonempty edge record required after vertex record")
        candidate_edges = _edge_record(record, path=path, line_number=line_number)
        if candidate_edges is not None:
            if edges is not None:
                raise GraphParseError(f"{path}:{line_number}: multiple edge records")
            edges = candidate_edges
            edge_line_number = line_number
            records.append({"line_number": line_number, "kind": "edge_record", "item_count": len(candidate_edges)})
            continue
        candidate_vertices = _vertex_record(record, path=path, line_number=line_number)
        if candidate_vertices is not None:
            if edges is not None:
                raise GraphParseError(f"{path}:{line_number}: vertex record appears after edge record")
            if vertices is not None:
                raise GraphParseError(f"{path}:{line_number}: multiple vertex records")
            vertices = candidate_vertices
            vertex_line_number = line_number
            records.append({"line_number": line_number, "kind": "vertex_record", "item_count": len(candidate_vertices)})
            continue
        if _group_word_record(record, path=path, line_number=line_number):
            records.append({"line_number": line_number, "kind": "auxiliary_group_word_record", "item_count": len(_split_top_level_list(record, path=path, line_number=line_number))})
            continue
        if edges is not None and _face_boundary_record(record, path=path, line_number=line_number):
            records.append({"line_number": line_number, "kind": "auxiliary_face_boundary_record", "item_count": len(_split_top_level_list(record, path=path, line_number=line_number))})
            continue
        position = "before" if edges is None else "after"
        raise GraphParseError(f"{path}:{line_number}: unrecognized {position}-edge record; refusing to skip it")
    if vertices is None or not vertices:
        raise GraphParseError(f"{path}: nonempty vertex record not found after TESS")
    if edges is None or not edges:
        raise GraphParseError(f"{path}: nonempty edge record not found after TESS")
    order = len(vertices)
    invalid = sorted({value for edge in edges for value in edge if value < 1 or value > order})
    if invalid:
        raise GraphParseError(f"{path}:{edge_line_number}: edge indices outside 1..{order}: {invalid}")
    rows = np.asarray([a - 1 for edge in edges for a in edge], dtype=int)
    cols = np.asarray([b - 1 for edge in edges for b in edge[::-1]], dtype=int)
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(order, order)).toarray()
    normalized_edges = sorted(tuple(sorted(edge)) for edge in edges)
    diagnostics = {
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "source": str(path),
        "line_count": len(lines),
        "tess_marker_line": marker + 1,
        "vertex_record_line": vertex_line_number,
        "edge_record_line": edge_line_number,
        "vertex_count": order,
        "edge_count": len(edges),
        "all_vertex_indices_valid": True,
        "records": records,
    }
    if return_diagnostics:
        return adjacency, normalized_edges, diagnostics
    return adjacency, normalized_edges


def benchmark_paths(root: Path, config):
    revision = str(config["public_data"]["hyperbloch_revision"])
    base = root / "public_data" / "HyperBloch" / revision / "repo" / "Paclet" / "Resources" / "ExampleData"
    return [
        base / "{8,3}-tess_T2.1_3.hcm",
        base / "{8,3}-tess_T2.1_3_sc-T5.1.hcs",
    ]


def run(config, run_dir: Path, run_id: str, root: Path, context):
    raw = run_dir / "raw" / "d09_reproduce_hyperbloch_dos"
    raw.mkdir(parents=True, exist_ok=False)
    source_dir = raw / "source_files"
    diagnostics_dir = raw / "parser_diagnostics"
    source_dir.mkdir()
    diagnostics_dir.mkdir()
    derived = run_dir / "derived" / "d09_reproduce_hyperbloch_dos.parquet"
    certificate = run_dir / "certificates" / "d09_reproduce_hyperbloch_dos.json"
    records = []
    dos_rows = []
    contract = context.get("d09_parser_contract", {})
    expected_counts = dict(KNOWN_EDGE_COUNTS)
    expected_counts.update(contract.get("known_fixture_edge_counts", {}))
    for path in benchmark_paths(root, config):
        source_copy = source_dir / path.name
        shutil.copyfile(path, source_copy)
        adjacency, edges, diagnostics = parse_graph(path, return_diagnostics=True)
        diagnostics.update({
            "source_relative": path.relative_to(root).as_posix(),
            "source_sha256": sha256_file(path),
            "preserved_copy": source_copy.relative_to(root).as_posix(),
            "preserved_copy_sha256": sha256_file(source_copy),
            "source_copy_hash_equal": sha256_file(path) == sha256_file(source_copy),
        })
        diagnostics_path = diagnostics_dir / f"{path.name}.parser.json"
        write_json(diagnostics_path, diagnostics)
        eigenvalues = np.linalg.eigvalsh(adjacency)
        degree = np.sum(adjacency, axis=1)
        grid = np.linspace(float(eigenvalues.min()) - 0.5, float(eigenvalues.max()) + 0.5, 401)
        density = gaussian_density(eigenvalues, np.ones_like(eigenvalues), grid, 0.08)
        np.savez_compressed(raw / f"{path.stem}_spectrum.npz", eigenvalues=eigenvalues, source_sha256=np.asarray(sha256_file(path)))
        records.append({
            "source": path.relative_to(root).as_posix(), "source_sha256": sha256_file(path),
            "preserved_source": source_copy.relative_to(root).as_posix(),
            "preserved_source_sha256": sha256_file(source_copy),
            "parser_diagnostics": diagnostics_path.relative_to(root).as_posix(),
            "parser_schema_version": PARSER_SCHEMA_VERSION,
            "vertex_count": adjacency.shape[0], "edge_count": len(edges),
            "expected_edge_count": expected_counts.get(path.name),
            "minimum_degree": float(degree.min()), "maximum_degree": float(degree.max()),
            "hermiticity_residual": float(np.max(np.abs(adjacency - adjacency.T))),
            "trace": float(np.trace(adjacency)),
            "second_moment_per_vertex": float(np.mean(eigenvalues ** 2)),
            "minimum_eigenvalue": float(eigenvalues.min()), "maximum_eigenvalue": float(eigenvalues.max()),
        })
        for energy, value in zip(grid, density):
            dos_rows.append({"benchmark": path.stem, "energy": float(energy), "density": float(value), "broadening": 0.08})
    pd.DataFrame(dos_rows).to_parquet(derived, index=False)
    passed = all(
        row["minimum_degree"] == row["maximum_degree"] == 3.0
        and row["hermiticity_residual"] == 0.0
        and row["source_sha256"] == row["preserved_source_sha256"]
        and row["expected_edge_count"] is not None
        and row["edge_count"] == row["expected_edge_count"]
        for row in records
    )
    status = "PASS_EXTERNAL" if passed else "FAIL_IMPLEMENTATION"
    write_json(certificate, {
        "task_id": "D-09", "run_id": run_id, "status": status,
        "records": records,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "parser_contract_frozen_before_rerun": bool(contract.get("frozen_before_scientific_rerun", False)),
        "strict_validation": ["nonempty_edges", "valid_vertex_indices", "explicit_malformed_record_failure", "known_fixture_edge_counts"],
        "raw_external_files_preserved": True,
        "parser_diagnostics_preserved": True,
        "implementation_check_only": True, "originality_claim": False,
        "benchmark": "HyperBloch bundled {8,3} primitive/supercell model graphs",
        "public_revision": config["public_data"]["hyperbloch_revision"],
    })
    context["d09_records"] = records
    return status, {"raw": raw, "derived": derived, "certificate": certificate}
