"""Public cumulative-boundary histogram encoding contracts."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from zcu_tools.simulate.adc import cumulative_count_bins


@pytest.mark.parametrize(
    "counts",
    [
        [2**63 - 1, 1, 0],
        [2**63 - 1, 2**63 - 1, 3],  # Repeated wraparound ends at positive 1.
    ],
)
def test_unrepresentable_point_total_is_rejected(counts):
    histogram = np.array([[0, 0, 0], counts], dtype=np.int64)

    with pytest.raises(ValueError, match="total.*int64"):
        cumulative_count_bins((histogram,))


def test_representable_totals_preserve_exact_counts_and_duplicate_bins():
    first = np.array([[0, 0, 0], [3, 0, 2], [2**63 - 3, 0, 2]], dtype=np.int64)
    second = np.array([[0, 0], [4, 1], [2**63 - 2, 1]], dtype=np.int64)

    bins = cumulative_count_bins((first, second))

    assert bins.counts.dtype == np.int64
    assert_array_equal(bins.counts, [[0, 0, 0, 0], [3, 0, 1, 1], [2**63 - 3, 0, 1, 1]])
    assert len(bins.indices) == 2
    assert_array_equal(bins.indices[0], [[2, 2, 2, 2], [0, 2, 2, 2], [0, 2, 2, 2]])
    assert_array_equal(bins.indices[1], [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
