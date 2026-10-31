"""Exact simulation time arithmetic at the shared numeric schema boundary."""

from fractions import Fraction

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
