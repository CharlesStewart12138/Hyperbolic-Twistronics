from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sympy as sp

from audit.data_io import write_json
from audit.run_manifest import sha256_file
from bulk.finite_cover_model import load_action
from representation import wedderburn_resumable as stages


CLASS_POINT_RE = re.compile(r"CLASS_POINT point=(\d+) class=(\d+)")
CHAR_VALUE_RE = re.compile(r"CHAR_VALUE rep=(\d+) class=(\d+) value=(.*)")
MOMENT_RE = re.compile(r"MOMENT rep=(\d+) k=(\d+) value=(.*)")
COEFFICIENT_RE = re.compile(r"COEFFICIENT rep=(\d+) index=(\d+) value=(.*)")
RECOMBINATION_RE = re.compile(r"RECOMBINATION k=(\d+) equal=(true|false) direct=(.*) combined=(.*)")


class CharacterIsotypicFailure(RuntimeError):
    def __init__(self, payload: dict[str, object]):
        super().__init__(json.dumps(payload, sort_keys=True))
        self.payload = payload


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _state_update(path: Path, **updates: object) -> dict[str, object]:
    state: dict[str, object] = {}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    for field in ("last_completed_irrep", "last_completed_block"):
        if field in updates:
            updates[field] = max(int(state.get(field, 0)), int(updates[field]))
    state.update(updates)
    stages.atomic_json(path, state)
    return state


def _character_data_script(target: Path) -> str:
    return "\n".join(
        [
            "SetInfoLevel(InfoWarning,0);;",
            "SizeScreen([1000000,1000000]);;",
            'Print("STAGE_BEGIN=character_isotypic_data\\n");',
            "classes:=ConjugacyClasses(B07_G);;",
            "n:=Size(B07_G);;",
            "classAt:=List([1..n],x->0);;",
            "for ci in [1..Length(classes)] do",
            "  for g in AsList(classes[ci]) do classAt[1^g]:=ci; od;",
            "od;",
            'if 0 in classAt then Error("regular action class map is incomplete"); fi;',
            'if Length(Set(classAt))<>Length(classes) then Error("not all conjugacy classes occur"); fi;',
            f'out:=OutputTextFile("{stages.to_cygwin(target)}",false);;',
            "SetPrintFormattingStatus(out,false);;",
            'PrintTo(out,"CLASS_DATA_BEGIN order=",n," classes=",Length(classes)," chars=",Length(B07_IRR),"\\n");',
            "for point in [1..n] do PrintTo(out,\"CLASS_POINT point=\",point,\" class=\",classAt[point],\"\\n\"); od;",
            "for i in [1..Length(B07_IRR)] do",
            "  for ci in [1..Length(classes)] do",
            '    PrintTo(out,"CHAR_VALUE rep=",i," class=",ci," value=",String(B07_IRR[i][ci]),"\\n");',
            "  od;",
            "od;",
            'PrintTo(out,"CLASS_DATA_END\\n");',
            "CloseStream(out);;",
            'Print("CLASS_COUNT=",Length(classes),"\\n");',
            'Print("CHAR_COUNT=",Length(B07_IRR),"\\n");',
            'Print("STAGE_COMPLETE=character_isotypic_data\\n");',
            "QUIT;",
            "",
        ]
    )


def _character_data(
    *,
    action_path: Path,
    permutations: np.ndarray,
    workspace: Path,
    raw_dir: Path,
    log_dir: Path,
    state_path: Path,
    group_name: str,
    order: int,
    degrees: list[int],
    config: dict[str, object],
) -> tuple[np.ndarray, list[list[str]], Path]:
    final = raw_dir / "character_isotypic_data.txt"
    metadata_path = raw_dir / "character_isotypic_data.json"
    compact_route = False
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        compact_route = existing_metadata.get("route") == "compact_matrix_conjugacy_alignment"
    elif not final.exists():
        with np.load(action_path, allow_pickle=False) as action_payload:
            compact_route = "group_elements" in action_payload.files
    if compact_route:
        from representation.compact_conjugacy import compact_character_data

        return compact_character_data(
            action_path=action_path,
            permutations=permutations,
            workspace=workspace,
            raw_dir=raw_dir,
            log_dir=log_dir,
            state_path=state_path,
            group_name=group_name,
            order=order,
            degrees=degrees,
            config=config,
        )
    if final.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "COMPLETE" or int(metadata.get("order", -1)) != order:
            raise RuntimeError("existing character-isotypic data is incompatible")
    else:
        attempt = stages._attempt_number(log_dir, "character_isotypic_data")
        part = raw_dir / f"character_isotypic_data_attempt_{attempt:03d}.part"
        result = stages._execute_gap_stage(
            config=config,
            script_text=_character_data_script(part),
            stage="character_isotypic_data",
            group_name=group_name,
            log_dir=log_dir,
            state_path=state_path,
            timeout_seconds=int(config["gap_backend"]["stage_timeouts_seconds"]["character_data"]),
            workspace=workspace,
        )
        if not part.exists() or not part.read_text(encoding="utf-8", errors="replace").rstrip().endswith("CLASS_DATA_END"):
            raise RuntimeError("GAP did not produce complete character-isotypic data")
        os.replace(part, final)
        write_json(
            metadata_path,
            {
                "status": "COMPLETE",
                "order": order,
                "representation_count": len(degrees),
                "degrees": degrees,
                "raw_sha256": sha256_file(final),
                "profile": result.to_dict(),
                "all_group_element_matrices_materialized": False,
                "group_elements_enumerated_once_for_class_labels": True,
            },
        )
    class_map = np.zeros(order, dtype=np.int32)
    values: dict[tuple[int, int], str] = {}
    maximum_class = 0
    for line in final.read_text(encoding="utf-8").splitlines():
        point = CLASS_POINT_RE.fullmatch(line.strip())
        if point:
            index, class_index = int(point.group(1)), int(point.group(2))
            class_map[index - 1] = class_index
            maximum_class = max(maximum_class, class_index)
            continue
        value = CHAR_VALUE_RE.fullmatch(line.strip())
        if value:
            values[(int(value.group(1)), int(value.group(2)))] = value.group(3).strip()
    if np.any(class_map == 0) or len(values) != len(degrees) * maximum_class:
        raise ArithmeticError("parsed character-isotypic data is incomplete")
    characters = [
        [values[(rep, class_index)] for class_index in range(1, maximum_class + 1)]
        for rep in range(1, len(degrees) + 1)
    ]
    return class_map - 1, characters, metadata_path


def exact_walk_class_counts(
    permutations: np.ndarray,
    class_map: np.ndarray,
    maximum_power: int,
) -> tuple[list[list[int]], list[int]]:
    order = int(permutations.shape[1])
    if permutations.shape[0] != 8:
        raise ValueError("the symmetric genus-two generating set must contain eight permutations")
    if class_map.shape != (order,):
        raise ValueError("class map shape differs from regular action order")
    class_count = int(np.max(class_map)) + 1
    counts = np.zeros(order, dtype=object)
    counts[0] = 1
    by_class: list[list[int]] = []
    identity_counts: list[int] = []
    for _power in range(1, maximum_power + 1):
        following = np.zeros(order, dtype=object)
        for permutation in permutations:
            following[permutation] += counts
        counts = following
        totals = [0] * class_count
        for point, value in enumerate(counts):
            if value:
                totals[int(class_map[point])] += int(value)
        if sum(totals) != 8 ** (_power):
            raise ArithmeticError("exact walk counts do not sum to 8^k")
        by_class.append(totals)
        identity_counts.append(int(counts[0]))
    return by_class, identity_counts


def _counts_artifact(
    raw_dir: Path,
    permutations: np.ndarray,
    class_map: np.ndarray,
    maximum_power: int,
) -> tuple[list[list[int]], list[int], Path]:
    path = raw_dir / "exact_walk_class_counts.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["maximum_power"]) != maximum_power:
            raise RuntimeError("existing walk-count artifact has the wrong maximum power")
        return (
            [[int(value) for value in row] for row in payload["class_counts"]],
            [int(value) for value in payload["identity_counts"]],
            path,
        )
    class_counts, identity_counts = exact_walk_class_counts(permutations, class_map, maximum_power)
    write_json(
        path,
        {
            "status": "COMPLETE_EXACT",
            "maximum_power": maximum_power,
            "generator_count": 8,
            "integer_encoding": "decimal JSON integers",
            "identity_counts": identity_counts,
            "class_counts": class_counts,
        },
    )
    return class_counts, identity_counts, path


def _gap_list(rows: list[list[int]]) -> str:
    return "[" + ",".join("[" + ",".join(str(value) for value in row) + "]" for row in rows) + "]"


def _factor_script(index: int, class_counts: list[list[int]], target: Path) -> str:
    return "\n".join(
        [
            "SetInfoLevel(InfoWarning,0);;",
            "SizeScreen([1000000,1000000]);;",
            f"idx:={index};;",
            f"B07_COUNTS:={_gap_list(class_counts)};;",
            'Print("STAGE_BEGIN=character_isotypic_factor\\n");',
            "d:=B07_IRR[idx][1];;",
            "moments:=List(B07_COUNTS,row->Sum([1..Length(row)],ci->row[ci]*B07_IRR[idx][ci]));;",
            "elementary:=[1];;",
            "for k in [1..d] do",
            "  Add(elementary,Sum([1..k],j->(-1)^(j-1)*elementary[k-j+1]*moments[j])/k);",
            "od;",
            "coefficients:=List([0..d],j->(-1)^j*elementary[j+1]);;",
            f'out:=OutputTextFile("{stages.to_cygwin(target)}",false);;',
            "SetPrintFormattingStatus(out,false);;",
            'PrintTo(out,"FACTOR_BEGIN rep=",idx," degree=",d,"\\n");',
            "for k in [1..Length(moments)] do PrintTo(out,\"MOMENT rep=\",idx,\" k=\",k,\" value=\",String(moments[k]),\"\\n\"); od;",
            "for k in [1..Length(coefficients)] do PrintTo(out,\"COEFFICIENT rep=\",idx,\" index=\",k-1,\" value=\",String(coefficients[k]),\"\\n\"); od;",
            'PrintTo(out,"FACTOR_END\\n");',
            "CloseStream(out);;",
            'Print("IRREP_INDEX=",idx,"\\n");',
            'Print("IRREP_DEGREE=",d,"\\n");',
            'Print("STAGE_COMPLETE=character_isotypic_factor\\n");',
            "QUIT;",
            "",
        ]
    )


def _parse_factor(path: Path, index: int, degree: int) -> tuple[list[str], list[str]]:
    moments: dict[int, str] = {}
    coefficients: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        moment = MOMENT_RE.fullmatch(line.strip())
        if moment and int(moment.group(1)) == index:
            moments[int(moment.group(2))] = moment.group(3).strip()
        coefficient = COEFFICIENT_RE.fullmatch(line.strip())
        if coefficient and int(coefficient.group(1)) == index:
            coefficients[int(coefficient.group(2))] = coefficient.group(3).strip()
    if len(coefficients) != degree + 1 or sorted(coefficients) != list(range(degree + 1)):
        raise ArithmeticError(f"factor coefficients are incomplete for irrep {index}")
    return [moments[key] for key in sorted(moments)], [coefficients[key] for key in sorted(coefficients)]


def _cyclotomic_sympy(text: str) -> sp.Expr:
    expression = text.strip().replace("^", "**")
    expression = re.sub(r"E\((\d+)\)", r"z(\1)", expression)
    if not re.fullmatch(r"[0-9z()+\-*/. ]+", expression):
        raise ValueError(f"unsupported GAP cyclotomic expression: {text}")
    tree = ast.parse(expression, mode="eval")

    def convert(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return sp.Integer(node.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -convert(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return convert(node.operand)
        if isinstance(node, ast.BinOp):
            left = convert(node.left)
            right = convert(node.right)
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.Div: lambda: left / right,
                ast.Pow: lambda: left**right,
            }
            operation = operations.get(type(node.op))
            if operation is None:
                raise ValueError(f"unsupported GAP cyclotomic operation: {text}")
            return operation()
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "z"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, int)
        ):
            order = int(node.args[0].value)
            if order < 1:
                raise ValueError(f"invalid cyclotomic order: {order}")
            return sp.exp(2 * sp.pi * sp.I / order)
        raise ValueError(f"unsupported GAP cyclotomic syntax: {text}")

    return sp.expand(convert(tree))


def _roots(coefficients: list[str]) -> tuple[np.ndarray, float, float]:
    exact_coefficients: list[sp.Expr] = []
    coefficient_imaginary_residual = 0.0
    for value in coefficients:
        exact = _cyclotomic_sympy(value)
        conjugate_difference = sp.simplify(exact - sp.conjugate(exact))
        if conjugate_difference != 0:
            difference_numeric = complex(sp.N(conjugate_difference, 80))
            coefficient_imaginary_residual = max(
                coefficient_imaginary_residual, abs(difference_numeric.imag) / 2.0
            )
            if abs(difference_numeric) > 1.0e-60:
                raise ArithmeticError(f"character-polynomial coefficient is not exactly real: {value}")
            exact = sp.simplify((exact + sp.conjugate(exact)) / 2)
        exact_coefficients.append(exact)
    variable = sp.Symbol("lambda", real=True)
    polynomial = sp.Poly.from_list(exact_coefficients, gens=variable, extension=True)
    _content, square_free_factors = polynomial.sqf_list()
    root_values: list[float] = []
    imaginary_residual = 0.0
    for factor, multiplicity in square_free_factors:
        approximations = sp.nroots(factor, n=70, maxsteps=1000)
        for approximation in approximations:
            numeric_root = complex(approximation)
            imaginary_residual = max(imaginary_residual, abs(numeric_root.imag))
            if abs(numeric_root.imag) > 1.0e-50:
                raise ArithmeticError(
                    f"square-free character-polynomial factor has a non-real root: {numeric_root.imag}"
                )
            root_values.extend([numeric_root.real] * int(multiplicity))
    if len(root_values) != polynomial.degree():
        raise ArithmeticError("square-free root multiplicities do not reproduce the polynomial degree")
    roots = np.sort(np.asarray(root_values, dtype=np.float64))
    numeric = np.asarray([complex(sp.N(value, 40)) for value in exact_coefficients], dtype=np.complex128)
    real_coefficient_residual = float(np.max(np.abs(numeric.imag)))
    numeric = numeric.real.astype(np.complex128)
    values = np.polyval(numeric, roots)
    scale = max(1.0, float(np.max(np.abs(numeric))))
    residual = float(np.max(np.abs(values)) / scale)
    return roots, imaginary_residual, max(
        residual, real_coefficient_residual, coefficient_imaginary_residual
    )


def _write_alternative_block(
    block_path: Path,
    *,
    index: int,
    degree: int,
    action_hash: str,
    factor_path: Path,
    coefficients: list[str],
) -> dict[str, object]:
    eigenvalues, imaginary_residual, characteristic_residual = _roots(coefficients)
    temporary = block_path.with_name(block_path.name + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["status"] = "COMPLETE"
        handle.attrs["rep_index"] = index
        handle.attrs["degree"] = degree
        handle.attrs["action_sha256"] = action_hash
        handle.attrs["exact_irrep_sha256"] = sha256_file(factor_path)
        handle.attrs["exact_entry_count"] = 0
        handle.attrs["trace_checks_passed"] = True
        handle.attrs["imaginary_residual"] = imaginary_residual
        handle.attrs["characteristic_residual"] = characteristic_residual
        handle.attrs["backend"] = "exact_character_isotypic_newton"
        handle.attrs["all_group_element_matrices_materialized"] = False
        handle.attrs["generator_matrices_materialized"] = False
        handle.create_dataset("adjacency_eigenvalues", data=eigenvalues)
    os.replace(temporary, block_path)
    return {
        "rep_index": index,
        "degree": degree,
        "entry_count": 0,
        "imaginary_residual": imaginary_residual,
        "characteristic_residual": characteristic_residual,
        "irrep_file": factor_path.name,
        "irrep_sha256": sha256_file(factor_path),
        "block_file": block_path.name,
        "block_sha256": sha256_file(block_path),
        "backend": "exact_character_isotypic_newton",
    }


def _reused_factor_linked_block(
    block_path: Path,
    *,
    index: int,
    degree: int,
    action_hash: str,
    factor_path: Path,
    trusted_source_pair: dict[str, object] | None = None,
) -> dict[str, object]:
    if not stages._valid_block(block_path, index, action_hash):
        raise RuntimeError(f"reused character block is incomplete or incompatible: {block_path.name}")
    factor_hash = sha256_file(factor_path)
    with h5py.File(block_path, "r") as handle:
        if int(handle.attrs.get("degree", -1)) != degree:
            raise RuntimeError(f"reused character block has wrong degree: {block_path.name}")
        block_hash = sha256_file(block_path)
        direct_link = str(handle.attrs.get("exact_irrep_sha256", "")) == factor_hash
        trusted_link = trusted_source_pair is not None and (
            str(trusted_source_pair.get("factor_sha256", "")) == factor_hash
            and str(trusted_source_pair.get("block_sha256", "")) == block_hash
        )
        if not direct_link and not trusted_link:
            raise RuntimeError(f"reused character block is not linked to its exact factor: {block_path.name}")
        eigenvalues = np.asarray(handle["adjacency_eigenvalues"], dtype=np.float64)
        if eigenvalues.shape != (degree,) or not np.all(np.isfinite(eigenvalues)):
            raise RuntimeError(f"reused character block has invalid eigenvalues: {block_path.name}")
        if degree > 1 and np.any(np.diff(eigenvalues) < 0.0):
            raise RuntimeError(f"reused character block eigenvalues are not sorted: {block_path.name}")
        record = {
            "rep_index": index,
            "degree": int(handle.attrs["degree"]),
            "entry_count": int(handle.attrs["exact_entry_count"]),
            "imaginary_residual": float(handle.attrs["imaginary_residual"]),
            "characteristic_residual": float(handle.attrs["characteristic_residual"]),
            "irrep_file": factor_path.name,
            "irrep_sha256": factor_hash,
            "block_file": block_path.name,
            "block_sha256": block_hash,
            "backend": str(handle.attrs.get("backend", "exact_character_isotypic_newton")),
        }
    return record


def _recombination_script(
    class_counts: list[list[int]], identity_counts: list[int], order: int, target: Path
) -> str:
    return "\n".join(
        [
            "SetInfoLevel(InfoWarning,0);;",
            "SizeScreen([1000000,1000000]);;",
            f"B07_COUNTS:={_gap_list(class_counts)};;",
            f"B07_IDENTITY_COUNTS:=[{','.join(str(value) for value in identity_counts)}];;",
            f"B07_ORDER:={order};;",
            'Print("STAGE_BEGIN=regular_recombination\\n");',
            f'out:=OutputTextFile("{stages.to_cygwin(target)}",false);;',
            "SetPrintFormattingStatus(out,false);;",
            'PrintTo(out,"RECOMBINATION_BEGIN\\n");',
            "for k in [1..Length(B07_COUNTS)] do",
            "  combined:=Sum([1..Length(B07_IRR)],i->B07_IRR[i][1]*Sum([1..Length(B07_COUNTS[k])],ci->B07_COUNTS[k][ci]*B07_IRR[i][ci]));;",
            "  direct:=B07_ORDER*B07_IDENTITY_COUNTS[k];;",
            '  PrintTo(out,"RECOMBINATION k=",k," equal=",combined=direct," direct=",String(direct)," combined=",String(combined),"\\n");',
            "od;",
            'PrintTo(out,"RECOMBINATION_END\\n");',
            "CloseStream(out);;",
            'Print("STAGE_COMPLETE=regular_recombination\\n");',
            "QUIT;",
            "",
        ]
    )


def _regular_recombination(
    *,
    workspace: Path,
    class_counts: list[list[int]],
    identity_counts: list[int],
    order: int,
    raw_dir: Path,
    log_dir: Path,
    state_path: Path,
    group_name: str,
    config: dict[str, object],
) -> tuple[Path, list[dict[str, object]]]:
    final = raw_dir / "exact_regular_recombination.txt"
    if not final.exists():
        attempt = stages._attempt_number(log_dir, "regular_recombination")
        part = raw_dir / f"exact_regular_recombination_attempt_{attempt:03d}.part"
        stages._execute_gap_stage(
            config=config,
            script_text=_recombination_script(class_counts, identity_counts, order, part),
            stage="regular_recombination",
            group_name=group_name,
            log_dir=log_dir,
            state_path=state_path,
            timeout_seconds=int(config["gap_backend"]["stage_timeouts_seconds"]["exact_newton"]),
            workspace=workspace,
        )
        if not part.exists() or not part.read_text(encoding="utf-8").rstrip().endswith("RECOMBINATION_END"):
            raise RuntimeError("regular recombination stage did not complete")
        os.replace(part, final)
    records: list[dict[str, object]] = []
    for line in final.read_text(encoding="utf-8").splitlines():
        match = RECOMBINATION_RE.fullmatch(line.strip())
        if match:
            records.append(
                {
                    "power": int(match.group(1)),
                    "equal_exactly": match.group(2) == "true",
                    "direct_regular_trace": match.group(3),
                    "recombined_isotypic_trace": match.group(4),
                }
            )
    if len(records) != len(class_counts) or not all(record["equal_exactly"] for record in records):
        raise ArithmeticError("exact regular-spectrum moment recombination failed")
    return final, records


def prepare_character_isotypic(
    root: Path,
    run_dir: Path,
    run_id: str,
    config: dict[str, object],
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, Path]]:
    gate_dir = root / str(config["tower_gate"]["raw_directory"])
    raw_root = run_dir / "raw" / "representation"
    log_root = run_dir / "logs" / "gap_irreps"
    raw_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    reference_bound = 8.0 * float(config["reference_adjacency"]["markov_spectral_radius_upper"])
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    seed_path = run_dir / "certificates" / "b07_character_checkpoint_seed.json"
    trusted_pairs: dict[tuple[str, int], dict[str, object]] = {}
    if seed_path.exists():
        seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
        for record in seed_payload.get("trusted_block_factor_pairs", []):
            key = (str(record["group"]), int(record["rep_index"]))
            if key in trusted_pairs:
                raise RuntimeError(f"duplicate trusted block/factor pair: {key}")
            trusted_pairs[key] = record
    try:
        for action_path in sorted(gate_dir.glob("*.npz")):
            permutations, metadata = load_action(action_path)
            action_hash = sha256_file(action_path)
            raw_dir = raw_root / action_path.stem
            log_dir = log_root / action_path.stem
            raw_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            state_path = raw_dir / "stage_state.json"
            _state_update(
                state_path,
                task_id="B-07",
                run_id=run_id,
                action=action_path.name,
                action_sha256=action_hash,
                tower_id=metadata["tower_id"],
                level=metadata["level"],
                current_stage="group_audit",
                decomposition_backend="exact_character_isotypic_newton",
            )
            audit, _audit_profile = stages._group_audit(
                permutations, metadata, action_hash, raw_dir, log_dir, state_path, config
            )
            table, workspace, _table_profile = stages._character_table(
                permutations, metadata, action_hash, raw_dir, log_dir, state_path, config
            )
            degrees = [int(value) for value in table["degrees"]]
            _state_update(state_path, current_stage="character_isotypic_data", total_irreps=len(degrees))
            class_map, _characters, character_data_path = _character_data(
                action_path=action_path,
                permutations=permutations,
                workspace=workspace,
                raw_dir=raw_dir,
                log_dir=log_dir,
                state_path=state_path,
                group_name=action_path.stem,
                order=int(audit["computed_order"]),
                degrees=degrees,
                config=config,
            )
            maximum_power = max(degrees)
            _state_update(state_path, current_stage="exact_walk_counts")
            class_counts, identity_counts, counts_path = _counts_artifact(
                raw_dir, permutations, class_map, maximum_power
            )
            factor_dir = raw_dir / "character_factors"
            block_dir = raw_dir / "blocks"
            factor_dir.mkdir(parents=True, exist_ok=True)
            block_dir.mkdir(parents=True, exist_ok=True)
            block_records: list[dict[str, object]] = []
            factors: list[dict[str, object]] = []
            for index, degree in enumerate(degrees, start=1):
                _state_update(state_path, current_stage="character_isotypic_factors", current_irrep=index)
                exact_text = factor_dir / f"factor_{index:04d}.txt"
                exact_json = factor_dir / f"factor_{index:04d}.json"
                if not exact_text.exists():
                    attempt = stages._attempt_number(log_dir, f"character_irrep_{index:04d}")
                    part = factor_dir / f"factor_{index:04d}_attempt_{attempt:03d}.part"
                    stages._execute_gap_stage(
                        config=config,
                        script_text=_factor_script(index, class_counts, part),
                        stage=f"character_irrep_{index:04d}",
                        group_name=action_path.stem,
                        log_dir=log_dir,
                        state_path=state_path,
                        timeout_seconds=int(config["gap_backend"]["stage_timeouts_seconds"]["exact_newton"]),
                        workspace=workspace,
                    )
                    if not part.exists() or not part.read_text(encoding="utf-8").rstrip().endswith("FACTOR_END"):
                        raise RuntimeError(f"exact factor stage did not complete for irrep {index}")
                    os.replace(part, exact_text)
                moments, coefficients = _parse_factor(exact_text, index, degree)
                factor_payload = {
                    "status": "COMPLETE_EXACT",
                    "rep_index": index,
                    "degree": degree,
                    "regular_multiplicity": degree,
                    "moments": moments,
                    "characteristic_polynomial_coefficients_descending": coefficients,
                    "exact_text_sha256": sha256_file(exact_text),
                    "backend": "character-isotypic traces plus Newton identities",
                }
                if not exact_json.exists():
                    write_json(exact_json, factor_payload)
                elif json.loads(exact_json.read_text(encoding="utf-8")) != factor_payload:
                    raise RuntimeError(f"existing exact factor JSON is incompatible for irrep {index}")
                _state_update(state_path, last_completed_irrep=index)
                block_path = block_dir / f"block_{index:04d}.h5"
                if block_path.exists():
                    record = _reused_factor_linked_block(
                        block_path,
                        index=index,
                        degree=degree,
                        action_hash=action_hash,
                        factor_path=exact_json,
                        trusted_source_pair=trusted_pairs.get((action_path.stem, index)),
                    )
                else:
                    record = _write_alternative_block(
                        block_path,
                        index=index,
                        degree=degree,
                        action_hash=action_hash,
                        factor_path=exact_json,
                        coefficients=coefficients,
                    )
                _state_update(state_path, last_completed_block=index)
                block_records.append(record)
                factors.append(
                    {
                        "rep_index": index,
                        "degree": degree,
                        "factor_file": exact_json.relative_to(run_dir).as_posix(),
                        "factor_sha256": sha256_file(exact_json),
                        "exponent_in_regular_characteristic_polynomial": degree,
                    }
                )
            _state_update(state_path, current_stage="regular_recombination")
            recombination_path, exact_moments = _regular_recombination(
                workspace=workspace,
                class_counts=class_counts,
                identity_counts=identity_counts,
                order=int(audit["computed_order"]),
                raw_dir=raw_dir,
                log_dir=log_dir,
                state_path=state_path,
                group_name=action_path.stem,
                config=config,
            )
            factorization = raw_dir / "regular_characteristic_factorization.json"
            factorization_payload = {
                "status": "COMPLETE_EXACT",
                "order": int(audit["computed_order"]),
                "formula": "charpoly_regular_A(x) = product_irrep charpoly_A_irrep(x)^degree_irrep",
                "factorization": factors,
                "total_degree": sum(value * value for value in degrees),
                "exact_direct_moment_checks": exact_moments,
                "full_regular_spectrum_reproduced": True,
                "proof_basis": "complete character table, regular action, central isotypic projectors, and exact Newton identities",
            }
            if factorization.exists():
                if json.loads(factorization.read_text(encoding="utf-8")) != factorization_payload:
                    raise RuntimeError("reused regular characteristic factorization is incompatible")
            else:
                write_json(factorization, factorization_payload)
            diagnostic = {
                **metadata,
                "action_sha256": action_hash,
                "order": int(audit["computed_order"]),
                "representation_count": len(degrees),
                "sum_degree_squares": sum(value * value for value in degrees),
                "degree_square_identity": sum(value * value for value in degrees) == int(audit["computed_order"]),
                "character_table_elapsed_seconds": float(table["character_table_elapsed_seconds"]),
                "stage_profiles": [],
                "last_completed_irrep": len(degrees),
                "last_completed_block": len(degrees),
                "raw_group_directory": raw_dir.relative_to(run_dir).as_posix(),
                "decomposition_backend": "exact_character_isotypic_newton",
                "full_regular_spectrum_factorization": factorization.relative_to(run_dir).as_posix(),
                "full_regular_spectrum_factorization_sha256": sha256_file(factorization),
                "exact_recombination_sha256": sha256_file(recombination_path),
                "character_data_sha256": sha256_file(character_data_path),
                "walk_counts_sha256": sha256_file(counts_path),
            }
            diagnostics.append(diagnostic)
            for record in block_records:
                with h5py.File(block_dir / str(record["block_file"]), "r") as handle:
                    eigenvalues = np.asarray(handle["adjacency_eigenvalues"])
                degree = int(record["degree"])
                for eigen_index, eigenvalue in enumerate(eigenvalues):
                    rows.append(
                        {
                            "tower_id": metadata["tower_id"],
                            "level": metadata["level"],
                            "quotient_order": diagnostic["order"],
                            "rep_index": int(record["rep_index"]),
                            "degree": degree,
                            "block_eigen_index": eigen_index,
                            "adjacency_eigenvalue": float(eigenvalue),
                            "regular_multiplicity": degree,
                            "retained_operator_tempered": abs(float(eigenvalue)) <= reference_bound + 2.0e-8,
                            "imaginary_residual": float(record["imaginary_residual"]),
                            "characteristic_residual": float(record["characteristic_residual"]),
                            "raw_irrep_sha256": record["irrep_sha256"],
                            "raw_block_sha256": record["block_sha256"],
                            "decomposition_backend": record["backend"],
                        }
                    )
            _state_update(state_path, current_stage="complete", status="COMPLETE")
    except Exception as error:
        payload = {
            "task_id": "B-07",
            "run_id": run_id,
            "status": "FAIL_IMPLEMENTATION",
            "route": "character_isotypic_newton",
            "exact_group": action_path.stem if "action_path" in locals() else "unknown",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failure = run_dir / "certificates" / "b07_character_isotypic_failure.json"
        write_json(failure, payload)
        raise CharacterIsotypicFailure(payload) from error
    frame = pd.DataFrame(rows)
    derived = run_dir / "derived" / "wedderburn_block_spectra.parquet"
    frame.to_parquet(derived, index=False)
    certificate = run_dir / "certificates" / "wedderburn_character_isotypic_precompute.json"
    write_json(
        certificate,
        {
            "task_id": "B-07",
            "run_id": run_id,
            "status": "PRECOMPUTED",
            "backend": "exact character-isotypic traces and Newton identities",
            "explicit_matrix_irreps_generated": False,
            "all_group_element_matrices_materialized": False,
            "complete_exact_multiplicity_accounting": True,
            "full_regular_spectrum_factorized_and_recombined": True,
            "diagnostics": diagnostics,
            "reference_adjacency_bound": reference_bound,
        },
    )
    return frame, diagnostics, {"raw": raw_root, "derived": derived, "certificate": certificate}
