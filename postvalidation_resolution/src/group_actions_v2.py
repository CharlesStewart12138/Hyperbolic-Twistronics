from __future__ import annotations

import numpy as np

from cover_towers import (
    RELATOR,
    AuxMatrix,
    MarkedGroup,
    aux_evaluate,
    aux_generators,
    aux_inverse,
    aux_multiply,
    pack,
)
from dyadic_ring import (
    RamifiedDyadicRing,
    RingMatrix,
    evaluate_word,
    expected_sl2_order,
    marked_generators,
    matrix_inverse,
    matrix_multiply,
)


def enumerate_marked_group_v2(
    tower_id: str,
    ring: RamifiedDyadicRing,
    *,
    with_auxiliary_p7: bool,
    maximum_order: int,
) -> MarkedGroup:
    """Enumerate the exact marked image, using the ambient order only as a bound."""
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
        ambient_upper_bound = expected_sl2_order(ring.depth) * 336
        auxiliary_base = 7
    else:
        auxiliary_moves = (None,) * 8
        identity_auxiliary = None
        ambient_upper_bound = expected_sl2_order(ring.depth)
        auxiliary_base = 1
    allocation_bound = min(ambient_upper_bound, maximum_order)

    dyadic_array = np.empty((allocation_bound, 4), dtype=np.uint16)
    auxiliary_array = np.empty((allocation_bound, 4), dtype=np.uint8) if with_auxiliary_p7 else None
    permutations = np.empty((8, allocation_bound), dtype=np.int32)
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
                if size >= allocation_bound:
                    raise MemoryError(
                        f"exact marked image reached the preregistered cap {maximum_order} "
                        f"before enumeration closed (ambient bound {ambient_upper_bound})"
                    )
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
    for generator_index in range(4):
        if not np.array_equal(
            permutations[generator_index + 4][permutations[generator_index]],
            identity_indices,
        ):
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
        upper_order_bound=ambient_upper_bound,
    )
