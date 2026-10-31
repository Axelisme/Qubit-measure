"""Exact simulation time arithmetic at the shared numeric schema boundary."""

from fractions import Fraction

import numpy as np
import pytest

from zcu_tools.core import simulation


@pytest.mark.parametrize(
    ("clocks_mhz", "periods"),
    [
        ((430.08, 384.0, 307.2, 6144.0), (100, 112, 140, 7)),
        ((430.08, 384.0, 76.8, 6144.0), (100, 112, 560, 7)),
    ],
    ids=["mock", "ftf"],
)
def test_board_clocks_share_exact_integer_periods(
    clocks_mhz: tuple[float, ...], periods: tuple[int, ...]
) -> None:
    time_base = simulation.TimeBase.from_clocks(clocks_mhz)

    assert time_base.ticks_per_us == Fraction(43008)
    assert tuple(time_base.period_ticks(clock) for clock in clocks_mhz) == periods


@pytest.mark.parametrize("frequency", [Fraction(43008), Fraction(3, 2)])
def test_ns_rounds_exact_grid_once_without_changing_array_shape(
    frequency: Fraction,
) -> None:
    time_base = simulation.TimeBase(frequency)
    start, offset, period = 100, 112, 140
    composed = start + offset + np.arange(6, dtype=np.int64).reshape(2, 3) * period
    absolute = np.array([[212, 352, 492], [632, 772, 912]], dtype=np.int64)
    expected = np.array(
        [float(Fraction(int(tick) * 1000) / frequency) for tick in absolute.flat]
    ).reshape(2, 3)

    assert time_base.ns(composed).shape == (2, 3)
    np.testing.assert_array_equal(time_base.ns(composed).view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(time_base.ns(absolute).view(np.uint64), expected.view(np.uint64))
    assert time_base.ns(-212).hex() == (-expected[0, 0]).item().hex()
    assert time_base.ns(0) == 0.0
    assert time_base.ns(np.array([], dtype=np.int64)).shape == (0,)
    assert time_base.ns(np.array(212, dtype=np.int64)).shape == ()
    if frequency == Fraction(3, 2):
        assert time_base.ns(1) == 666.6666666666666
