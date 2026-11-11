import numpy as np
import pytest

from advaita_virtualcell.transport import allocate, integer_totals, transport_counts


@pytest.mark.parametrize("seed", range(12))
def test_transport_attains_lower_bound_with_exact_margins(seed):
    rng = np.random.default_rng(seed)
    original = rng.poisson(4, (17, 23)).astype(np.int32)
    target = integer_totals(rng.random(23), int(original.sum()))
    result, info = transport_counts(original, target)
    result = result.toarray()
    np.testing.assert_array_equal(result.sum(1), original.sum(1))
    np.testing.assert_array_equal(result.sum(0), target)
    assert (result >= 0).all()
    assert np.abs(result - original).sum() == np.abs(target - original.sum(0)).sum()
    assert info["minimal_count_movement"]
    identity, _ = transport_counts(result, target)
    np.testing.assert_array_equal(identity.toarray(), result)


def test_transport_can_populate_previously_absent_genes():
    source = np.array([[2, 0, 0], [4, 0, 0]], np.int32)
    result, _ = transport_counts(source, [0, 3, 3])
    np.testing.assert_array_equal(result.sum(0), [[0, 3, 3]])
    np.testing.assert_array_equal(result.sum(1), [[2], [4]])


def test_allocation_capacity_and_zero_weight_fallback():
    np.testing.assert_array_equal(allocate([0, 0, 0], 8, [2, 3, 4]), [2, 3, 3])
    np.testing.assert_array_equal(allocate([100, 1, 1], 7, [1, 4, 4]), [1, 3, 3])
    with pytest.raises(ValueError, match="capacity"):
        allocate([1, 1], 10, [3, 3])
    with pytest.raises(ValueError, match="mass"):
        transport_counts([[1, 2]], [2, 2])


def test_integer_rounding_ties_are_stable():
    np.testing.assert_array_equal(integer_totals([1, 1, 1], 5), [2, 2, 1])
    with pytest.raises(ValueError):
        integer_totals([np.nan, 1], 4)
