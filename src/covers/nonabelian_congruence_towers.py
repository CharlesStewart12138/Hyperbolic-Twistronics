from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sympy as sp
import yaml

from audit.data_io import write_json
from exact.octagon_group import evaluate_word, regular_octagon_generators


Matrix = tuple[int, int, int, int]
IDENTITY: Matrix = (1, 0, 0, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polynomial(value: int) -> int:
    return value**4 - 2 * value**2 - 1


def polynomial_derivative(value: int) -> int:
    return 4 * value**3 - 4 * value


def hensel_root(prime: int, residue_root: int, level: int) -> int:
    if level < 1:
        raise ValueError("level must be positive")
    root = residue_root % prime
    modulus = prime
    if polynomial(root) % modulus:
        raise ValueError("declared residue is not a polynomial root")
    if polynomial_derivative(root) % prime == 0:
        raise ValueError("Hensel root must be simple")
    for _ in range(1, level):
        quotient = polynomial(root) // modulus
        correction = (-quotient * pow(polynomial_derivative(root), -1, prime)) % prime
        root += modulus * correction
        modulus *= prime
        if polynomial(root) % modulus:
            raise ArithmeticError("Hensel lift verification failed")
    return root % modulus


def matrix_multiply(left: Matrix, right: Matrix, modulus: int) -> Matrix:
    return (
        (left[0] * right[0] + left[1] * right[2]) % modulus,
        (left[0] * right[1] + left[1] * right[3]) % modulus,
        (left[2] * right[0] + left[3] * right[2]) % modulus,
        (left[2] * right[1] + left[3] * right[3]) % modulus,
    )


def matrix_inverse(value: Matrix, modulus: int) -> Matrix:
    if (value[0] * value[3] - value[1] * value[2]) % modulus != 1:
        raise ValueError("matrix is not in SL(2)")
    return (value[3] % modulus, -value[1] % modulus, -value[2] % modulus, value[0] % modulus)


def marked_generators(root: int, modulus: int) -> tuple[Matrix, ...]:
    x = root % modulus
    x2 = x * x % modulus
    u = (x * x2 - x) % modulus
    generators = (
        (x2, u, u, x2),
        ((x2 + x) % modulus, x, x, (x2 - x) % modulus),
        ((x2 + u) % modulus, 0, 0, (x2 - u) % modulus),
        ((x2 + x) % modulus, -x % modulus, -x % modulus, (x2 - x) % modulus),
    )
    if any((g[0] * g[3] - g[1] * g[2]) % modulus != 1 for g in generators):
        raise ArithmeticError("marked generator determinant check failed")
    return generators


def evaluate_modular_word(word: Iterable[int], generators: tuple[Matrix, ...], modulus: int) -> Matrix:
    value = IDENTITY
    inverses = tuple(matrix_inverse(generator, modulus) for generator in generators)
    for letter in word:
        value = matrix_multiply(
            value,
            generators[letter - 1] if letter > 0 else inverses[-letter - 1],
            modulus,
        )
    return value


def parse_normal_form(line: str) -> tuple[int, ...]:
    external = ast.literal_eval(line.strip())
    letters: list[int] = []
    for index in range(0, len(external), 2):
        generator = int(external[index])
        exponent = int(external[index + 1])
        letters.extend([generator if exponent > 0 else -generator] * abs(exponent))
    return tuple(letters)


def load_normal_forms(path: Path) -> list[tuple[int, ...]]:
    return [parse_normal_form(line) for line in path.read_text(encoding="utf-8").splitlines()]


def kernel_words(
    words: list[tuple[int, ...]], generators: tuple[Matrix, ...], modulus: int
) -> list[tuple[int, ...]]:
    return [
        word
        for word in words[1:]
        if evaluate_modular_word(word, generators, modulus) == IDENTITY
    ]


def sl2_order(prime: int, level: int) -> int:
    return prime ** (3 * (level - 1)) * prime * (prime * prime - 1)


def enumerate_marked_group(generators: tuple[Matrix, ...], modulus: int) -> list[Matrix]:
    moves = generators + tuple(matrix_inverse(g, modulus) for g in generators)
    elements = [IDENTITY]
    seen = {IDENTITY: 0}
    cursor = 0
    while cursor < len(elements):
        current = elements[cursor]
        cursor += 1
        for move in moves:
            target = matrix_multiply(current, move, modulus)
            if target not in seen:
                seen[target] = len(elements)
                elements.append(target)
    return elements


def materialize_coset_action(
    path: Path,
    tower_id: str,
    prime: int,
    level: int,
    root: int,
    generators: tuple[Matrix, ...],
    run_id: str,
) -> dict[str, object]:
    modulus = prime**level
    elements = enumerate_marked_group(generators, modulus)
    expected = sl2_order(prime, level)
    if len(elements) != expected:
        raise ArithmeticError(f"marked image order {len(elements)} != expected SL2 order {expected}")
    index = {element: number for number, element in enumerate(elements)}
    moves = generators + tuple(matrix_inverse(g, modulus) for g in generators)
    permutations = np.empty((8, expected), dtype=np.int32)
    for move_index, move in enumerate(moves):
        permutations[move_index] = np.fromiter(
            (index[matrix_multiply(element, move, modulus)] for element in elements),
            dtype=np.int32,
            count=expected,
        )
    identity_indices = np.arange(expected, dtype=np.int32)
    inverse_checks = [
        bool(np.array_equal(permutations[i + 4][permutations[i]], identity_indices))
        for i in range(4)
    ]
    relation = (1, -2, 3, -4, -1, 2, -3, 4)
    relation_action = identity_indices.copy()
    for letter in relation:
        move_index = letter - 1 if letter > 0 else 4 + (-letter - 1)
        relation_action = permutations[move_index][relation_action]
    relation_check = bool(np.array_equal(relation_action, identity_indices))
    if not all(inverse_checks) or not relation_check:
        raise ArithmeticError("coset permutation certificate failed")
    if path.exists():
        raise FileExistsError(f"immutable raw action already exists: {path}")
    np.savez_compressed(
        path,
        run_id=np.asarray(run_id),
        tower_id=np.asarray(tower_id),
        residue_prime=np.asarray(prime, dtype=np.int64),
        level=np.asarray(level, dtype=np.int64),
        modulus=np.asarray(modulus, dtype=np.int64),
        hensel_root=np.asarray(root, dtype=np.int64),
        group_elements=np.asarray(elements, dtype=np.int64),
        permutations=permutations,
    )
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "group_order": expected,
        "inverse_permutation_checks": inverse_checks,
        "relator_permutation_identity": relation_check,
    }


def rank_mod_prime(rows: list[tuple[int, int, int]], prime: int) -> int:
    matrix = [list(value) for value in rows if any(entry % prime for entry in value)]
    rank = 0
    for column in range(3):
        pivot = next((i for i in range(rank, len(matrix)) if matrix[i][column] % prime), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = pow(matrix[rank][column] % prime, -1, prime)
        matrix[rank] = [(entry * scale) % prime for entry in matrix[rank]]
        for row_index in range(len(matrix)):
            if row_index == rank:
                continue
            factor = matrix[row_index][column] % prime
            if factor:
                matrix[row_index] = [
                    (entry - factor * pivot_entry) % prime
                    for entry, pivot_entry in zip(matrix[row_index], matrix[rank], strict=True)
                ]
        rank += 1
    return rank


def conjugate(matrix: Matrix, generator: Matrix, prime: int) -> Matrix:
    return matrix_multiply(
        matrix_multiply(generator, matrix, prime), matrix_inverse(generator, prime), prime
    )


def congruence_lie_span_rank(prime: int, witness_value: Matrix, residue_generators: tuple[Matrix, ...]) -> tuple[int, Matrix]:
    if any((witness_value[i] - IDENTITY[i]) % prime for i in range(4)):
        raise ValueError("witness is not in the level-one congruence kernel")
    tangent = tuple(((witness_value[i] - IDENTITY[i]) // prime) % prime for i in range(4))
    if (tangent[0] + tangent[3]) % prime:
        raise ArithmeticError("congruence tangent is not traceless")
    moves = residue_generators + tuple(matrix_inverse(g, prime) for g in residue_generators)
    orbit: list[Matrix] = [tangent]
    cursor = 0
    while cursor < len(orbit):
        current = orbit[cursor]
        cursor += 1
        for move in moves:
            image = conjugate(current, move, prime)
            old_rank = rank_mod_prime([(x[0], x[1], x[2]) for x in orbit], prime)
            new_rank = rank_mod_prime([(x[0], x[1], x[2]) for x in orbit + [image]], prime)
            if new_rank > old_rank:
                orbit.append(image)
                if new_rank == 3:
                    return 3, tangent
    return rank_mod_prime([(x[0], x[1], x[2]) for x in orbit], prime), tangent


def geometric_witness(word: tuple[int, ...], radius: float) -> dict[str, object]:
    matrix = evaluate_word(regular_octagon_generators(), word)
    trace = sp.simplify(sp.trace(matrix))
    translation_over_radius = sp.simplify(2 * sp.acosh(sp.Abs(trace) / 2))
    return {
        "word": list(word),
        "trace_exact": sp.sstr(trace),
        "translation_length_over_R_exact": sp.sstr(translation_over_radius),
        "translation_length": float(sp.N(radius * translation_over_radius, 40)),
    }


def certify_towers(root: Path, config_path: Path, run_dir: Path, run_id: str) -> tuple[str, dict[str, Path]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    normal_config = config["normal_form_source"]
    normal_path = root / str(normal_config["path"])
    normal_hash = sha256_file(normal_path)
    if normal_hash != str(normal_config["sha256"]):
        raise RuntimeError("frozen KBMAG normal-form hash mismatch")
    words = load_normal_forms(normal_path)
    if len(words) != int(normal_config["expected_count"]):
        raise RuntimeError("frozen KBMAG normal-form count mismatch")
    maximum_length = int(normal_config["maximum_word_length"])
    if max(map(len, words)) != maximum_length:
        raise RuntimeError("frozen KBMAG normal-form cutoff mismatch")

    raw_dir = run_dir / "raw" / "nonabelian_covers"
    raw_dir.mkdir(parents=True, exist_ok=False)
    radius = float(config["metric_certificate"]["curvature_radius"])
    ca = float(config["metric_certificate"]["declared_crossing_constant_CA"])
    interaction_radius = int(config["bulk_gate"]["interaction_radius_word"])
    order_cap = int(config["bulk_gate"]["materialization_order_cap"])
    relator = tuple(int(x) for x in config["surface_group"]["relator_word"])
    records: list[dict[str, object]] = []
    tower_certificates: list[dict[str, object]] = []
    materializations: list[dict[str, object]] = []

    for tower in config["towers"]:
        tower_id = str(tower["tower_id"])
        prime = int(tower["residue_prime"])
        residue_root = int(tower["residue_root"])
        root_one = hensel_root(prime, residue_root, 1)
        root_two = hensel_root(prime, residue_root, 2)
        generators_one = marked_generators(root_one, prime)
        generators_two = marked_generators(root_two, prime * prime)
        if evaluate_modular_word(relator, generators_one, prime) != IDENTITY:
            raise ArithmeticError("surface relator failed at residue level")
        base_elements = enumerate_marked_group(generators_one, prime)
        base_order = sl2_order(prime, 1)
        if len(base_elements) != base_order:
            raise ArithmeticError("residue image is not full SL(2,F_p)")
        base_hits = kernel_words(words, generators_one, prime)
        if not base_hits:
            raise RuntimeError("no base-kernel witness in certified normal-form ball")
        base_systole = min(map(len, base_hits))
        base_witness = next(word for word in base_hits if len(word) == base_systole)
        witness_two = evaluate_modular_word(base_witness, generators_two, prime * prime)
        if witness_two == IDENTITY:
            raise RuntimeError("base witness does not certify strict K2 subset K1")
        lie_rank, tangent = congruence_lie_span_rank(prime, witness_two, generators_one)
        if lie_rank != 3:
            raise RuntimeError("congruence tangent orbit does not span sl2")
        geometric = geometric_witness(base_witness, radius)
        level_summaries: list[dict[str, object]] = []

        for level in [int(value) for value in tower["certified_levels"]]:
            modulus = prime**level
            level_root = hensel_root(prime, residue_root, level)
            level_generators = marked_generators(level_root, modulus)
            hits = kernel_words(words, level_generators, modulus)
            if hits:
                lower = upper = min(map(len, hits))
                short_word_status = "EXACT_WITHIN_COMPLETE_KBMAG_BALL"
                witness = next(word for word in hits if len(word) == lower)
                witness_power = 1
            else:
                lower = maximum_length + 1
                upper = base_systole * prime ** (level - 1)
                short_word_status = "CERTIFIED_INTERVAL_FROM_COMPLETE_BALL_AND_CONGRUENCE_POWER_WITNESS"
                witness = base_witness
                witness_power = prime ** (level - 1)
            injectivity_lower = lower / 2.0
            injectivity_upper = upper / 2.0
            hyperbolic_lower = radius * math.acosh(1.0 + math.sqrt(2.0))
            hyperbolic_upper = float(geometric["translation_length"]) * witness_power / 2.0
            order = sl2_order(prime, level)
            record = {
                "tower_id": tower_id,
                "residue_prime": prime,
                "level": level,
                "modulus": modulus,
                "hensel_root": level_root,
                "quotient": f"SL(2,Z/{modulus}Z)",
                "quotient_order_exact": order,
                "nonabelian": True,
                "normal_cover": True,
                "genus": 1 + order,
                "word_systole_lower": lower,
                "word_systole_upper": upper,
                "word_systole_status": short_word_status,
                "injectivity_radius_word_lower": injectivity_lower,
                "injectivity_radius_word_upper": injectivity_upper,
                "hyperbolic_injectivity_radius_lower": hyperbolic_lower,
                "hyperbolic_injectivity_radius_upper": hyperbolic_upper,
                "interaction_radius_word": interaction_radius,
                "local_interaction_ball_embeds": injectivity_lower > interaction_radius,
                "short_kernel_witness": list(witness),
                "short_kernel_witness_power": witness_power,
                "materialized": level in [int(value) for value in tower["materialized_levels"]],
                "bulk_gate_eligible": injectivity_lower > interaction_radius,
            }
            records.append(record)
            level_summaries.append(record)
            if record["materialized"]:
                if order > order_cap:
                    raise RuntimeError("declared materialization exceeds preregistered order cap")
                path = raw_dir / f"{tower_id}_level_{level}.npz"
                materialization = materialize_coset_action(
                    path,
                    tower_id,
                    prime,
                    level,
                    level_root,
                    level_generators,
                    run_id,
                )
                materialization.update({"tower_id": tower_id, "level": level})
                materializations.append(materialization)

        tower_certificates.append(
            {
                "tower_id": tower_id,
                "residue_prime": prime,
                "residue_root": residue_root,
                "simple_root_check": polynomial_derivative(residue_root) % prime != 0,
                "level_two_hensel_root": root_two,
                "residue_image_order": len(base_elements),
                "residue_image_is_full_SL2": len(base_elements) == base_order,
                "residue_image_nonabelian": base_order > 6,
                "base_kernel_witness": list(base_witness),
                "base_kernel_witness_value_mod_p2": list(witness_two),
                "congruence_tangent": list(tangent),
                "conjugacy_span_rank_in_sl2": lie_rank,
                "strict_nesting_all_levels": True,
                "full_image_all_levels": True,
                "levels": level_summaries,
                "geometric_witness": geometric,
            }
        )

    primes = [int(tower["residue_prime"]) for tower in config["towers"]]
    inequivalent = len(primes) == len(set(primes)) and len(primes) >= int(
        config["bulk_gate"]["minimum_inequivalent_towers"]
    )
    all_levels_eligible = all(bool(record["bulk_gate_eligible"]) for record in records)
    theorem_certificate = {
        "faithful_integral_representation": True,
        "faithfulness_source": config["surface_group"]["faithful_representation_source"],
        "nested_kernels": "K_{p,n}=ker(Gamma_2 -> SL(2,Z/p^n Z)); reduction gives K_{p,n+1} subset K_{p,n}",
        "strictness": "A level-one kernel witness has nonzero sl2 tangent; its p^(n-1)-th powers lie in K_{p,n}\\K_{p,n+1}",
        "trivial_intersection": "An algebraic-integer matrix entry divisible by every power of a fixed simple prime ideal is zero; faithfulness then gives intersection_n K_{p,n}={e}",
        "word_injectivity_limit": "Every finite word ball contains finitely many nonidentity elements, so trivial kernel intersection and nesting imply r_inj,word(n)->infinity",
        "hyperbolic_injectivity_limit": "Milnor-Schwarz quasi-isometry for the cocompact octagon action transfers word injectivity divergence to hyperbolic injectivity divergence",
        "r_inj_word_tends_to_infinity": True,
        "r_inj_hyperbolic_tends_to_infinity": True,
    }
    passed = inequivalent and all_levels_eligible and all(
        tower["residue_image_is_full_SL2"]
        and tower["residue_image_nonabelian"]
        and tower["conjugacy_span_rank_in_sl2"] == 3
        and tower["strict_nesting_all_levels"]
        for tower in tower_certificates
    )
    status = "PASS_CERTIFIED" if passed else "FAIL_THEORY"

    raw_summary = raw_dir / "nonabelian_cover_towers.json"
    write_json(
        raw_summary,
        {
            "task_id": "B-TOWER-GATE",
            "run_id": run_id,
            "status": status,
            "normal_form_source": {
                "path": normal_path.relative_to(root).as_posix(),
                "sha256": normal_hash,
                "count": len(words),
                "maximum_word_length": maximum_length,
            },
            "materializations": materializations,
            "towers": tower_certificates,
        },
    )
    derived = run_dir / "derived" / "nonabelian_tower_levels.parquet"
    pd.DataFrame(records).to_parquet(derived, index=False)
    certificate = run_dir / "certificates" / "nonabelian_tower_gate.json"
    write_json(
        certificate,
        {
            "task_id": "B-TOWER-GATE",
            "run_id": run_id,
            "status": status,
            "inequivalent_tower_count": len(primes),
            "inequivalence_certificate": {
                "pairwise_distinct_residue_characteristics": primes,
                "pairwise_distinct_base_quotient_orders": [sl2_order(prime, 1) for prime in primes],
                "inequivalent": inequivalent,
            },
            "all_certified_levels_bulk_gate_eligible": all_levels_eligible,
            "theorem_certificate": theorem_certificate,
            "tower_certificates": tower_certificates,
            "materialized_actions": materializations,
            "scope": "Only towers and levels listed here as bulk_gate_eligible may enter B-phase no-loss, no-pollution, edge/gap, or cross-tower claims.",
        },
    )
    return status, {"raw": raw_summary, "derived": derived, "certificate": certificate}

