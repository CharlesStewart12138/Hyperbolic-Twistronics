from covers.nonabelian_congruence_towers import (
    IDENTITY,
    enumerate_marked_group,
    evaluate_modular_word,
    hensel_root,
    marked_generators,
    polynomial,
    sl2_order,
)


RELATOR = (1, -2, 3, -4, -1, 2, -3, 4)


def test_hensel_roots_are_nested_exact_roots():
    for prime, residue in ((7, 2), (23, 11), (31, 3)):
        root_one = hensel_root(prime, residue, 1)
        root_two = hensel_root(prime, residue, 2)
        assert root_two % prime == root_one
        assert polynomial(root_two) % (prime * prime) == 0


def test_marked_generators_satisfy_surface_relator():
    for prime, residue in ((7, 2), (23, 11), (31, 3)):
        generators = marked_generators(residue, prime)
        assert evaluate_modular_word(RELATOR, generators, prime) == IDENTITY


def test_p7_residue_image_is_full_nonabelian_sl2():
    generators = marked_generators(2, 7)
    elements = enumerate_marked_group(generators, 7)
    assert len(elements) == sl2_order(7, 1) == 336
    assert evaluate_modular_word((1, 2, -1, -2), generators, 7) != IDENTITY
