"""Nearest-even numeric conversion previously covered at the core raw seam."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from zcu_tools.simulate.rounding import rounded_sums, shot_layout


def test_nearest_even_owns_values_and_rejects_integer_overflow():
    values = np.array([[2.5, 3.5], [-2.5, -3.5]])
    actual = rounded_sums(values)
    assert_array_equal(actual, [[2, 4], [-2, -4]])
    assert actual.dtype == np.int64
    actual[:] = 99
    assert_array_equal(values, [[2.5, 3.5], [-2.5, -3.5]])
    for value in (2.0**63, np.nextafter(-(2.0**63), -np.inf), np.inf, np.nan):
        with pytest.raises(ValueError, match="int64"):
            rounded_sums(np.array([value]))
    assert_array_equal(rounded_sums(np.array([-(2.0**63)])), [-9223372036854775808])


@pytest.mark.parametrize("avg_level", [0, 1, 2])
def test_shot_layout_retains_point_read_order_and_owns_output(avg_level):
    values = np.arange(6 * 4 * 2 * 2, dtype=float).reshape((6, 4, 2, 2))
    actual = shot_layout(values, (2, 3), 4, avg_level)
    restored = np.moveaxis(actual, avg_level, 2)
    assert_array_equal(restored, values.reshape(2, 3, 4, 2, 2))
    assert not np.shares_memory(actual, values)
