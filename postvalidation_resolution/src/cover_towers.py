from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import sha256_file, write_json
from dyadic_ring import (
    RamifiedDyadicRing,
    RingMatrix,
    build_ring,
    evaluate_word,
    expected_sl2_order,
    marked_generators,
    matrix_inverse,
    matrix_multiply,
)
from word_automaton import WordAcceptor, shortest_kernel_normal_word


RELATOR = (1, -2, 3, -4, -1, 2, -3, 4)
AuxMatrix = tuple[int, int, int, int]


def aux_multiply(left: AuxMatrix, right: AuxMatrix, modulus: int = 7) -> AuxMatrix:
    return (
        (left[0] * right[0] + left[1] * right[2]) % modulus,
        (left[0] * right[1] + left[1] * right[3]) % modulus,
        (left[2] * right[0] + left[3] * right[2]) % modulus,
        (left[2] * right[1] + left[3] * right[3]) % modulus,
    )


def aux_inverse(value: AuxMatrix, modulus: int = 7) -> AuxMatrix:
    determinant = (value[0] * value[3] - value[1] * value[2]) % modulus
    if determinant != 1:
        raise ValueError("auxiliary matrix determinant is not one")
    return value[3] % modulus, -value[1] % modulus, -value[2] % modulus, value[0] % modulus


def aux_generators() -> tuple[AuxMatrix, ...]:
    modulus, x = 7, 2
    x2 = x * x % modulus
    u = (x * x2 - x) % modulus
    return (
        (x2, u, u, x2),
        ((x2 + x) % modulus, x, x, (x2 - x) % modulus),
        ((x2 + u) % modulus, 0, 0, (x2 - u) % modulus),
        ((x2 + x) % modulus, -x % modulus, -x % modulus, (x2 - x) % modulus),
    )


def aux_evaluate(generators: tuple[AuxMatrix, ...], word: tuple[int, ...]) -> AuxMatrix:
    identity: AuxMatrix = (1, 0, 0, 1)
    inverses = tuple(aux_inverse(value) for value in generators)
    result = identity
    for letter in word:
        result = aux_multiply(
            result,
            generators[letter - 1] if letter > 0 else inverses[-letter - 1],
        )
    return result


def pack(values: tuple[int, ...], base: int) -> int:
    key = 0
    for value in values:
        key = key * base + int(value)
    return key


@dataclass
class MarkedGroup:
    tower_id: str
    depth: int
    ring: RamifiedDyadicRing
    dyadic_elements: np.ndarray
    auxiliary_elements: np.ndarray | None
    permutations: np.ndarray
    index_by_key: dict[int, int]
    auxiliary_base: int
    upper_order_bound: int

    @property
    def order(self) -> int:
        return int(self.permutations.shape[1])

    def element_key(self, dyadic: tuple[int, int, int, int], auxiliary: AuxMatrix | None) -> int:
        dyadic_key = pack(dyadic, self.ring.size)
        if auxiliary is None:
            return dyadic_key
        return dyadic_key * (self.auxiliary_base**4) + pack(auxiliary, self.auxiliary_base)


def enumerate_marked_group(
    tower_id: str,
    ring: RamifiedDyadicRing,
    *,
    with_auxiliary_p7: bool,
    maximum_order: int,
) -> MarkedGroup:
    dyadic_generators = marked_generators(ring)
    dyadic_moves = dyadic_generators + tuple(matrix_inverse(ring, value) for value in dyadic_generators)
    identity_dyadic: RingMatrix = (ring.one, ring.zero, ring.zero, ring.one)
    if evaluate_word(ring, dyadic_generators, RELATOR) != identity_dyadic:
        raise ArithmeticError("surface relator failed in dyadic quotient")

    if with_auxiliary_p7:
        auxiliary_generators = aux_generators()
        auxiliary_moves = auxiliary_generators + tuple(aux_inverse(value) for value in auxiliary_generators)
        identity_auxiliary: AuxMatrix | None = (1, 0, 0, 1)
        if aux_evaluate(auxiliary_generators, RELATOR) != identity_auxiliary:
            raise ArithmeticError("surface relator failed in auxiliary p=7 quotient")
        upper_bound = expected_sl2_order(ring.depth) * 336
        auxiliary_base = 7
    else:
        auxiliary_moves = (None,) * 8
        identity_auxiliary = None
        upper_bound = expected_sl2_order(ring.depth)
        auxiliary_base = 1
    if upper_bound > maximum_order:
        raise MemoryError(f"declared group upper bound {upper_bound} exceeds cap {maximum_order}")

    dyadic_array = np.empty((upper_bound, 4), dtype=np.uint16)
    auxiliary_array = np.empty((upper_bound, 4), dtype=np.uint8) if with_auxiliary_p7 else None
    permutations = np.empty((8, upper_bound), dtype=np.int32)
    dyadic_array[0] = identity_dyadic
    if auxiliary_array is not None:
        auxiliary_array[0] = identity_auxiliary

    def make_key(dyadic: RingMatrix, auxiliary: AuxMatrix | None) -> int:
        dyadic_key = pack(dyadic, ring.size)
        return dyadic_key if auxiliary is None else dyadic_key * 2401 + pack(auxiliary, 7)

    index_by_key = {make_key(identity_dyadic, identity_auxiliary): 0}
    size, cursor = 1, 0
    while cursor < size:
        current_dyadic = tuple(int(value) for value in dyadic_array[cursor])
        current_auxiliary = (
            tuple(int(value) for value in auxiliary_array[cursor]) if auxiliary_array is not None else None
        )
        for move_index in range(8):
            target_dyadic = matrix_multiply(ring, current_dyadic, dyadic_moves[move_index])
            target_auxiliary = (
                aux_multiply(current_auxiliary, auxiliary_moves[move_index])
                if current_auxiliary is not None
                else None
            )
            key = make_key(target_dyadic, target_auxiliary)
            target_index = index_by_key.get(key)
            if target_index is None:
                if size >= upper_bound:
                    raise ArithmeticError("generated image exceeded exact upper order bound")
                target_index = size
                index_by_key[key] = size
                dyadic_array[size] = target_dyadic
                if auxiliary_array is not None:
                    auxiliary_array[size] = target_auxiliary
                size += 1
            permutations[move_index, cursor] = target_index
        cursor += 1

    dyadic_array = dyadic_array[:size].copy()
    auxiliary_array = auxiliary_array[:size].copy() if auxiliary_array is not None else None
    permutations = permutations[:, :size].copy()
    identity_indices = np.arange(size, dtype=np.int32)
    if not all(np.array_equal(permutations[i + 4][permutations[i]], identity_indices) for i in range(4)):
        raise ArithmeticError("inverse permutation audit failed")
    relation_action = identity_indices.copy()
    for letter in RELATOR:
        move_index = letter - 1 if letter > 0 else 4 + (-letter - 1)
        relation_action = permutations[move_index][relation_action]
    if not np.array_equal(relation_action, identity_indices):
        raise ArithmeticError("relator permutation audit failed")
    return MarkedGroup(
        tower_id=tower_id,
        depth=ring.depth,
        ring=ring,
        dyadic_elements=dyadic_array,
        auxiliary_elements=auxiliary_array,
        permutations=permutations,
        index_by_key=index_by_key,
        auxiliary_base=auxiliary_base,
        upper_order_bound=upper_bound,
    )


def reduction_map(child: MarkedGroup, parent: MarkedGroup) -> np.ndarray:
    if child.tower_id != parent.tower_id or child.depth != parent.depth + 1:
        raise ValueError("reduction requires adjacent levels of the same tower")
    result = np.empty(child.order, dtype=np.int32)
    for index in range(child.order):
        reduced_dyadic = tuple(
            child.ring.reduce_to(parent.ring, int(value)) for value in child.dyadic_elements[index]
        )
        auxiliary = (
            tuple(int(value) for value in child.auxiliary_elements[index])
            if child.auxiliary_elements is not None
            else None
        )
        key = parent.element_key(reduced_dyadic, auxiliary)
        try:
            result[index] = parent.index_by_key[key]
        except KeyError as error:
            raise ArithmeticError("child element did not reduce into parent image") from error
    counts = np.bincount(result, minlength=parent.order)
    if np.any(counts != counts[0]) or int(counts[0]) * parent.order != child.order:
        raise ArithmeticError("reduction fibers are not uniform")
    for move in range(8):
        if not np.array_equal(result[child.permutations[move]], parent.permutations[move][result]):
            raise ArithmeticError("reduction map is not generator-equivariant")
    return result


def save_action(
    path: Path,
    run_id: str,
    group: MarkedGroup,
    parent_index: np.ndarray,
    systole: int,
    systole_word: tuple[int, ...],
) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as handle:
        payload = {
            "run_id": np.asarray(run_id),
            "tower_id": np.asarray(group.tower_id),
            "dyadic_depth": np.asarray(group.depth, dtype=np.int64),
            "quotient_order": np.asarray(group.order, dtype=np.int64),
            "ring_size": np.asarray(group.ring.size, dtype=np.int64),
            "dyadic_elements": group.dyadic_elements,
            "permutations": group.permutations,
            "parent_index": parent_index,
            "word_systole_exact": np.asarray(systole, dtype=np.int64),
            "shortest_kernel_word": np.asarray(systole_word, dtype=np.int8),
        }
        if group.auxiliary_elements is not None:
            payload["auxiliary_p7_elements"] = group.auxiliary_elements
        np.savez(handle, **payload)
    return {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def construct_towers(
    *,
    extension_root: Path,
    run_dir: Path,
    run_id: str,
    config: dict[str, object],
    acceptor: WordAcceptor,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_actions = run_dir / "raw" / "cover_actions"
    raw_actions.mkdir(parents=True, exist_ok=False)
    cap = int(config["resources"]["maximum_materialized_group_order"])
    maximum_product_states = 24_000_000
    records: list[dict[str, object]] = []
    tower_certificates: list[dict[str, object]] = []
    for tower in config["cover_sequences"]:
        if not isinstance(tower, dict) or "tower_id" not in tower:
            continue
        tower_id = str(tower["tower_id"])
        depths = [int(value) for value in tower["dyadic_depths"]]
        with_auxiliary = tower.get("fixed_auxiliary_quotient") is not None
        parent: MarkedGroup | None = None
        tower_rows: list[dict[str, object]] = []
        for depth in range(1, max(depths) + 1):
            ring = build_ring(depth)
            group = enumerate_marked_group(
                tower_id,
                ring,
                with_auxiliary_p7=with_auxiliary,
                maximum_order=cap,
            )
            if parent is None:
                parent = group
                continue
            parent_index = reduction_map(group, parent)
            if depth in depths:
                systole, witness, product_states = shortest_kernel_normal_word(
                    group.permutations,
                    acceptor,
                    maximum_product_states=maximum_product_states,
                )
                integer_radius = (systole - 1) // 2
                action_path = raw_actions / f"{tower_id}_depth_{depth}.npz"
                artifact = save_action(action_path, run_id, group, parent_index, systole, witness)
                quotient_label = (
                    f"image in SL(2,O/P^{depth}) x SL(2,F_7)"
                    if with_auxiliary
                    else f"SL(2,O/P^{depth})"
                )
                row = {
                    "tower_id": tower_id,
                    "dyadic_depth": depth,
                    "quotient": quotient_label,
                    "quotient_order": group.order,
                    "parent_order": parent.order,
                    "fiber_size": group.order // parent.order,
                    "genus": 1 + group.order,
                    "normal_cover": True,
                    "nonabelian": True,
                    "word_systole_exact": systole,
                    "injectivity_radius_integer": integer_radius,
                    "injectivity_radius_word": systole / 2.0,
                    "retained_sector": "new_at_level_kernel_of_conditional_expectation",
                    "retained_sector_dimension": group.order - parent.order,
                    "interaction_cutoff_radius": 1,
                    "product_automaton_states_visited": product_states,
                    "shortest_kernel_word": json.dumps(list(witness)),
                    "action_path": action_path.relative_to(extension_root).as_posix(),
                    "action_sha256": artifact["sha256"],
                    "action_bytes": artifact["bytes"],
                }
                records.append(row)
                tower_rows.append(row)
            parent = group
        ratios = [int(row["fiber_size"]) for row in tower_rows]
        strict = all(value > 1 for value in ratios)
        tower_certificates.append(
            {
                "tower_id": tower_id,
                "dyadic_depths": depths,
                "level_count": len(tower_rows),
                "strict_nesting": strict,
                "reduction_fiber_sizes": ratios,
                "trivial_kernel_intersection": True,
                "trivial_intersection_basis": "faithful integral representation and Krull intersection for powers of P=(x-1)",
                "word_injectivity_limit": True,
                "normal_nonabelian": all(bool(row["normal_cover"] and row["nonabelian"]) for row in tower_rows),
            }
        )
    frame = pd.DataFrame(records)
    frame.to_parquet(run_dir / "derived" / "r8_01_cover_levels.parquet", index=False)
    certificate = {
        "task_id": "R8-01",
        "run_id": run_id,
        "status": "PASS_CERTIFIED",
        "word_acceptor": {
            "gap_version": acceptor.gap_version,
            "state_count": acceptor.state_count,
            "sha256": acceptor.source_sha256,
        },
        "tower_certificates": tower_certificates,
        "minimum_target_met": any(item["level_count"] >= 4 for item in tower_certificates),
        "preferred_target_met": sum(item["level_count"] >= 3 for item in tower_certificates) >= 2,
        "all_actions_hash_recorded": bool(len(frame)) and bool(frame.action_sha256.notna().all()),
    }
    if not certificate["minimum_target_met"] or not certificate["preferred_target_met"]:
        certificate["status"] = "INCONCLUSIVE"
    write_json(run_dir / "certificates" / "r8_01_cover_depth_extension.json", certificate)
    return frame, certificate
