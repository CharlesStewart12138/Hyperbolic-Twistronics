from __future__ import annotations

import json
import os
import re
from collections import deque
from pathlib import Path

import numpy as np

from audit.data_io import write_json
from audit.run_manifest import sha256_file
from representation import wedderburn_resumable as stages


Matrix = tuple[int, int, int, int]
ALIGNMENT_RE = re.compile(
    r"CLASS_ALIGNMENT compact=(\d+) gap=(\d+) compact_size=(\d+) gap_size=(\d+)"
)
CHARACTER_RE = re.compile(r"CHAR_VALUE rep=(\d+) class=(\d+) value=(.*)")


def _multiply(left: Matrix, right: Matrix, modulus: int) -> Matrix:
    return (
        (left[0] * right[0] + left[1] * right[2]) % modulus,
        (left[0] * right[1] + left[1] * right[3]) % modulus,
        (left[2] * right[0] + left[3] * right[2]) % modulus,
        (left[2] * right[1] + left[3] * right[3]) % modulus,
    )


def _matrix(row: np.ndarray) -> Matrix:
    return tuple(int(value) for value in row)  # type: ignore[return-value]


def compact_conjugacy_classes(
    group_elements: np.ndarray,
    permutations: np.ndarray,
    modulus: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = int(group_elements.shape[0])
    if group_elements.shape != (order, 4):
        raise ValueError("compact group elements must have shape (order,4)")
    if permutations.shape != (8, order):
        raise ValueError("regular action must have shape (8,order)")
    identity = (1 % modulus, 0, 0, 1 % modulus)
    if _matrix(group_elements[0]) != identity:
        raise ArithmeticError("compact regular action does not begin at the identity")
    index = {_matrix(row): number for number, row in enumerate(group_elements)}
    if len(index) != order:
        raise ArithmeticError("compact group-element list contains duplicates")
    conjugations: list[tuple[Matrix, Matrix]] = []
    for generator_index in range(4):
        generator = _matrix(group_elements[int(permutations[generator_index, 0])])
        inverse = _matrix(group_elements[int(permutations[generator_index + 4, 0])])
        if _multiply(generator, inverse, modulus) != identity:
            raise ArithmeticError("compact generator inverse check failed")
        conjugations.extend([(inverse, generator), (generator, inverse)])
    class_map = np.full(order, -1, dtype=np.int32)
    representatives: list[int] = []
    sizes: list[int] = []
    for start in range(order):
        if class_map[start] >= 0:
            continue
        class_index = len(representatives)
        representatives.append(start)
        class_map[start] = class_index
        queue: deque[int] = deque([start])
        size = 0
        while queue:
            point = queue.popleft()
            size += 1
            element = _matrix(group_elements[point])
            for left, right in conjugations:
                target_matrix = _multiply(_multiply(left, element, modulus), right, modulus)
                target = index.get(target_matrix)
                if target is None:
                    raise ArithmeticError("compact conjugate is absent from the enumerated group")
                if class_map[target] < 0:
                    class_map[target] = class_index
                    queue.append(target)
                elif int(class_map[target]) != class_index:
                    raise ArithmeticError("compact conjugacy orbits overlap")
        sizes.append(size)
    if np.any(class_map < 0) or sum(sizes) != order:
        raise ArithmeticError("compact conjugacy classes do not partition the group")
    return (
        class_map,
        np.asarray(representatives, dtype=np.int32),
        np.asarray(sizes, dtype=np.int64),
    )


def representative_words(permutations: np.ndarray, representatives: np.ndarray) -> list[list[int]]:
    order = int(permutations.shape[1])
    parent = np.full(order, -2, dtype=np.int32)
    parent_move = np.zeros(order, dtype=np.int8)
    parent[0] = -1
    target_mask = np.zeros(order, dtype=np.bool_)
    target_mask[representatives] = True
    remaining = int(np.count_nonzero(target_mask)) - int(target_mask[0])
    queue = np.empty(order, dtype=np.int32)
    queue[0] = 0
    head = 0
    tail = 1
    letters = (1, 2, 3, 4, -1, -2, -3, -4)
    while head < tail and remaining:
        point = int(queue[head])
        head += 1
        for move_index, permutation in enumerate(permutations):
            target = int(permutation[point])
            if parent[target] != -2:
                continue
            parent[target] = point
            parent_move[target] = letters[move_index]
            queue[tail] = target
            tail += 1
            if target_mask[target]:
                remaining -= 1
    if remaining:
        raise ArithmeticError("not all compact class representatives are reachable")
    words: list[list[int]] = []
    for representative in representatives:
        point = int(representative)
        reverse: list[int] = []
        while point:
            reverse.append(int(parent_move[point]))
            point = int(parent[point])
        word = list(reversed(reverse))
        endpoint = 0
        for letter in word:
            move_index = letter - 1 if letter > 0 else 4 + (-letter - 1)
            endpoint = int(permutations[move_index, endpoint])
        if endpoint != int(representative):
            raise ArithmeticError("representative word does not reproduce its compact endpoint")
        words.append(word)
    return words


def _write_compact_artifact(
    path: Path,
    class_map: np.ndarray,
    representatives: np.ndarray,
    sizes: np.ndarray,
    words: list[list[int]],
    modulus: int,
) -> None:
    offsets = np.zeros(len(words) + 1, dtype=np.int32)
    flattened: list[int] = []
    for index, word in enumerate(words):
        flattened.extend(word)
        offsets[index + 1] = len(flattened)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            class_map=class_map,
            representatives=representatives,
            class_sizes=sizes,
            word_offsets=offsets,
            word_letters=np.asarray(flattened, dtype=np.int8),
            modulus=np.asarray(modulus, dtype=np.int64),
        )
    os.replace(temporary, path)


def _load_compact_artifact(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], int]:
    with np.load(path, allow_pickle=False) as payload:
        class_map = np.asarray(payload["class_map"], dtype=np.int32)
        representatives = np.asarray(payload["representatives"], dtype=np.int32)
        sizes = np.asarray(payload["class_sizes"], dtype=np.int64)
        offsets = np.asarray(payload["word_offsets"], dtype=np.int32)
        letters = np.asarray(payload["word_letters"], dtype=np.int8)
        modulus = int(payload["modulus"])
    words = [
        [int(value) for value in letters[offsets[index] : offsets[index + 1]]]
        for index in range(len(representatives))
    ]
    return class_map, representatives, sizes, words, modulus


def _gap_alignment_script(target: Path, words: list[list[int]], sizes: np.ndarray) -> str:
    gap_words = "[" + ",".join("[" + ",".join(str(value) for value in word) + "]" for word in words) + "]"
    gap_sizes = "[" + ",".join(str(int(value)) for value in sizes) + "]"
    return "\n".join(
        [
            "SetInfoLevel(InfoWarning,0);;",
            "SizeScreen([1000000,1000000]);;",
            f"B07_COMPACT_WORDS:={gap_words};;",
            f"B07_COMPACT_SIZES:={gap_sizes};;",
            'Print("STAGE_BEGIN=compact_character_alignment\\n");',
            "classes:=ConjugacyClasses(B07_G);;",
            f'out:=OutputTextFile("{stages.to_cygwin(target)}",false);;',
            "SetPrintFormattingStatus(out,false);;",
            'PrintTo(out,"COMPACT_CHARACTER_BEGIN classes=",Length(classes)," chars=",Length(B07_IRR),"\\n");',
            "for compact in [1..Length(B07_COMPACT_WORDS)] do",
            "  g:=One(B07_G);;",
            "  for letter in B07_COMPACT_WORDS[compact] do",
            "    if letter>0 then g:=g*B07_GENS[letter]; else g:=g*(B07_GENS[-letter]^-1); fi;",
            "  od;",
            "  gapClass:=PositionProperty(classes,c->g in c);;",
            '  if gapClass=fail then Error("compact representative has no GAP conjugacy class"); fi;',
            '  PrintTo(out,"CLASS_ALIGNMENT compact=",compact," gap=",gapClass," compact_size=",B07_COMPACT_SIZES[compact]," gap_size=",Size(classes[gapClass]),"\\n");',
            "  for i in [1..Length(B07_IRR)] do",
            '    PrintTo(out,"CHAR_VALUE rep=",i," class=",compact," value=",String(B07_IRR[i][gapClass]),"\\n");',
            "  od;",
            "od;",
            'PrintTo(out,"COMPACT_CHARACTER_END\\n");',
            "CloseStream(out);;",
            'Print("CLASS_COUNT=",Length(classes),"\\n");',
            'Print("CHAR_COUNT=",Length(B07_IRR),"\\n");',
            'Print("STAGE_COMPLETE=compact_character_alignment\\n");',
            "QUIT;",
            "",
        ]
    )


def _parse_alignment(
    final: Path,
    class_map: np.ndarray,
    sizes: np.ndarray,
    degrees: list[int],
) -> tuple[list[list[str]], np.ndarray]:
    alignments: dict[int, tuple[int, int, int]] = {}
    values: dict[tuple[int, int], str] = {}
    for line in final.read_text(encoding="utf-8").splitlines():
        alignment = ALIGNMENT_RE.fullmatch(line.strip())
        if alignment:
            compact, gap, compact_size, gap_size = map(int, alignment.groups())
            alignments[compact] = (gap, compact_size, gap_size)
            continue
        character = CHARACTER_RE.fullmatch(line.strip())
        if character:
            values[(int(character.group(1)), int(character.group(2)))] = character.group(3).strip()
    class_count = len(sizes)
    if sorted(alignments) != list(range(1, class_count + 1)):
        raise ArithmeticError("compact-to-GAP conjugacy alignment is incomplete")
    gap_indices = [alignments[index][0] for index in range(1, class_count + 1)]
    if sorted(gap_indices) != list(range(1, class_count + 1)):
        raise ArithmeticError("compact-to-GAP conjugacy alignment is not bijective")
    for compact in range(1, class_count + 1):
        _gap, compact_size, gap_size = alignments[compact]
        if compact_size != int(sizes[compact - 1]) or compact_size != gap_size:
            raise ArithmeticError("compact and GAP conjugacy-class sizes differ")
    if int(class_map[0]) != 0 or alignments[1][0] != 1:
        raise ArithmeticError("identity conjugacy class alignment failed")
    if len(values) != len(degrees) * class_count:
        raise ArithmeticError("compact character-value export is incomplete")
    characters = [
        [values[(rep, compact)] for compact in range(1, class_count + 1)]
        for rep in range(1, len(degrees) + 1)
    ]
    if [row[0] for row in characters] != [str(degree) for degree in degrees]:
        raise ArithmeticError("character degrees disagree with the compact identity column")
    compact_to_gap = np.asarray(gap_indices, dtype=np.int32) - 1
    return characters, compact_to_gap


def compact_character_data(
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
    compact_path = raw_dir / "compact_conjugacy_classes.npz"
    final = raw_dir / "character_isotypic_data.txt"
    metadata_path = raw_dir / "character_isotypic_data.json"
    if compact_path.exists():
        class_map, representatives, sizes, words, modulus = _load_compact_artifact(compact_path)
    else:
        with np.load(action_path, allow_pickle=False) as payload:
            if "group_elements" not in payload.files:
                raise ValueError("action has no compact group-element representation")
            group_elements = np.asarray(payload["group_elements"], dtype=np.int64)
            modulus = int(payload["modulus"])
        class_map, representatives, sizes = compact_conjugacy_classes(
            group_elements, permutations, modulus
        )
        if len(sizes) != len(degrees):
            raise ArithmeticError("compact class count differs from irreducible-character count")
        words = representative_words(permutations, representatives)
        _write_compact_artifact(
            compact_path, class_map, representatives, sizes, words, modulus
        )
    if class_map.shape != (order,) or len(sizes) != len(degrees):
        raise ArithmeticError("compact conjugacy artifact has incompatible dimensions")
    profile: dict[str, object] | None = None
    if not final.exists():
        attempt = stages._attempt_number(log_dir, "compact_character_alignment")
        part = raw_dir / f"compact_character_alignment_attempt_{attempt:03d}.part"
        result = stages._execute_gap_stage(
            config=config,
            script_text=_gap_alignment_script(part, words, sizes),
            stage="compact_character_alignment",
            group_name=group_name,
            log_dir=log_dir,
            state_path=state_path,
            timeout_seconds=int(config["gap_backend"]["stage_timeouts_seconds"]["character_data"]),
            workspace=workspace,
            maximum_job_memory_bytes=int(
                config["gap_backend"]["recovery_timeout_policy"]["maximum_peak_memory_bytes"]
            ),
        )
        if not part.exists() or not part.read_text(encoding="utf-8", errors="replace").rstrip().endswith(
            "COMPACT_CHARACTER_END"
        ):
            raise RuntimeError("GAP did not produce a complete compact character alignment")
        os.replace(part, final)
        profile = result.to_dict()
    characters, compact_to_gap = _parse_alignment(final, class_map, sizes, degrees)
    gap_order_class_map = compact_to_gap[class_map]
    payload = {
        "status": "COMPLETE",
        "route": "compact_matrix_conjugacy_alignment",
        "order": order,
        "modulus": modulus,
        "representation_count": len(degrees),
        "conjugacy_class_count": len(sizes),
        "degrees": degrees,
        "raw_sha256": sha256_file(final),
        "compact_classes_sha256": sha256_file(compact_path),
        "profile": profile,
        "all_group_element_matrices_materialized": False,
        "regular_permutation_group_elements_enumerated": False,
        "compact_group_elements_used": order,
        "compact_element_encoding": "four residues representing a 2x2 matrix",
        "gap_elements_evaluated": len(sizes),
        "gap_evaluation_scope": "one representative word per compact conjugacy class",
        "returned_class_map_order": "GAP character-table conjugacy-class order",
        "compact_to_gap_class_permutation": [int(value) + 1 for value in compact_to_gap],
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("route") != payload["route"] or existing.get("raw_sha256") != payload["raw_sha256"]:
            raise RuntimeError("existing compact character metadata is incompatible")
    else:
        write_json(metadata_path, payload)
    return gap_order_class_map, characters, metadata_path
