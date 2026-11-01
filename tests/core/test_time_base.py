"""Exact simulation time arithmetic at the shared numeric schema boundary."""

from fractions import Fraction

import numpy as np
import pytest
from zcu_tools.core import simulation


@pytest.mark.parametrize(
    ("clocks_mhz", "frequency", "periods"),
    [
        ((430.08, 384.0, 307.2, 6144.0), Fraction(43008), (100, 112, 140, 7)),
        ((430.08, 384.0, 76.8, 6144.0), Fraction(43008), (100, 112, 560, 7)),
        ((430.08, 599.04, 307.2, 38.4), Fraction(419328, 5), (195, 140, 273, 2184)),
    ],
    ids=["mock", "ftf", "hardware"],
)
def test_board_clocks_share_exact_integer_periods(
    clocks_mhz: tuple[float, ...], frequency: Fraction, periods: tuple[int, ...]
) -> None:
    time_base = simulation.TimeBase.from_clocks(clocks_mhz)

    assert time_base.ticks_per_us == frequency
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
    np.testing.assert_array_equal(
        time_base.ns(composed).view(np.uint64), expected.view(np.uint64)
    )
    np.testing.assert_array_equal(
        time_base.ns(absolute).view(np.uint64), expected.view(np.uint64)
    )
    assert time_base.ns(-212).hex() == (-expected[0, 0]).item().hex()
    assert time_base.ns(0) == 0.0
    assert time_base.ns(np.array([], dtype=np.int64)).shape == (0,)
    if frequency == Fraction(3, 2):
        assert time_base.ns(1) == 666.6666666666666


@pytest.mark.parametrize("frequency", [Fraction(43008), Fraction(3, 2)])
def test_zero_dimensional_tick_arrays_return_float64_arrays(
    frequency: Fraction,
) -> None:
    time_base = simulation.TimeBase(frequency)
    ticks = np.array(212, dtype=np.int64)
    expected = np.array(float(Fraction(212000) / frequency), dtype=np.float64)

    result = time_base.ns(ticks)

    assert type(result) is np.ndarray
    assert result.dtype == np.dtype(np.float64)
    assert result.shape == ()
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("frequency", [Fraction(43008), Fraction(3, 2), Fraction(1000)])
def test_ns_quantizes_exact_rational_time_with_integer_ties_even(
    frequency: Fraction,
) -> None:
    base = simulation.TimeBase(frequency)
    ticks = np.array([-129, -43, -9, -3, 0, 3, 9, 43, 129], dtype=np.int64)
    expected = [
        float(2 * round(Fraction(int(t) * 1000, 1) / frequency / 2)) for t in ticks
    ]
    actual = base.ns(ticks, quantum_ns=2)
    np.testing.assert_array_equal(
        actual.view(np.uint64), np.array(expected).view(np.uint64)
    )
    for tick, value in zip(ticks, expected, strict=True):
        assert base.ns(tick, quantum_ns=2) == value


def test_fixture_times_must_be_exactly_on_the_grid() -> None:
    time_base = simulation.TimeBase.from_clocks((430.08, 384.0, 307.2, 6144.0))

    assert time_base.ticks(0.5) == 21504
    assert time_base.ticks(Fraction(1, 43008)) == 1
    assert simulation.TimeBase(Fraction(3, 2)).ticks(Fraction(2, 3)) == 1
    for time_us in (0.04, np.nextafter(0.5, 0.0).item()):
        with pytest.raises(ValueError, match="grid"):
            time_base.ticks(time_us)
    with pytest.raises(ValueError, match="grid"):
        time_base.period_ticks(5.0)


@pytest.mark.parametrize("frequency", [Fraction(43008), Fraction(3, 2)])
def test_ns_rejects_values_past_the_exact_numerator_boundary(
    frequency: Fraction,
) -> None:
    time_base = simulation.TimeBase(frequency)
    limit = 2**53 // (1000 * frequency.denominator)
    for tick in (-limit, limit):
        expected = float(Fraction(tick * 1000) / frequency)
        assert time_base.ns(tick).hex() == expected.hex()
    for tick in (-limit - 1, limit + 1, -(2**63), 2**63):
        with pytest.raises(OverflowError):
            time_base.ns(tick)
    for tick in (-limit - 1, limit + 1, -(2**63)):
        with pytest.raises(OverflowError):
            time_base.ns(np.array([[0, tick]], dtype=np.int64))


@pytest.mark.parametrize("frequency", [Fraction(43008), Fraction(3, 2)])
def test_fixture_ticks_obey_the_same_exact_range(frequency: Fraction) -> None:
    time_base = simulation.TimeBase(frequency)
    limit = 2**53 // (1000 * frequency.denominator)
    for tick in (-limit, limit):
        assert time_base.ticks(Fraction(tick) / frequency) == tick
    for tick in (-limit - 1, limit + 1, -(2**63), 2**63):
        with pytest.raises(OverflowError):
            time_base.ticks(Fraction(tick) / frequency)


@pytest.mark.parametrize("clock", [0.0, -1.0, float("inf"), float("nan")])
def test_clock_frequencies_must_be_positive_and_finite(clock: float) -> None:
    with pytest.raises(ValueError, match="positive|finite"):
        simulation.TimeBase.from_clocks((clock,))
    with pytest.raises(ValueError, match="positive|finite"):
        simulation.TimeBase(Fraction(1)).period_ticks(clock)


@pytest.mark.parametrize(
    "frequency",
    [Fraction(2**53 + 1), Fraction(2**63), Fraction(1, 2**63), Fraction(1, 10**16)],
)
def test_unrepresentable_time_bases_fail_at_construction(frequency: Fraction) -> None:
    with pytest.raises(OverflowError):
        simulation.TimeBase(frequency)


@pytest.mark.parametrize(
    ("clocks_mhz", "lcm_mhz", "limit_reason"),
    [
        (
            (430.0801001, 599.0401),
            "25763522617191401/10000",
            "not exact in float64",
        ),
        (
            (430.0801, 599.0401, 307.2001, 38.4),
            "15195948907225359974592/5",
            "exceeds int64 range",
        ),
    ],
    ids=["float64-exactness", "int64-range"],
)
def test_common_clock_overflow_reports_clocks_lcm_and_ratio_denominators(
    clocks_mhz: tuple[float, ...], lcm_mhz: str, limit_reason: str
) -> None:
    with pytest.raises(OverflowError) as raised:
        simulation.TimeBase.from_clocks(iter(clocks_mhz))

    message = str(raised.value)
    for clock in clocks_mhz:
        assert repr(clock) in message
    assert f"LCM={lcm_mhz} MHz" in message
    assert "clock ratios" in message
    assert "denominators" in message
    assert limit_reason in message


def test_common_clock_build_validates_input_and_intermediate_ranges() -> None:
    with pytest.raises(ValueError, match="at least one clock"):
        simulation.TimeBase.from_clocks(())
    with pytest.raises(OverflowError):
        simulation.TimeBase.from_clocks((1e20,))
    with pytest.raises(OverflowError):
        simulation.TimeBase.from_clocks((1e-20,))
    with pytest.raises(OverflowError):
        simulation.TimeBase.from_clocks((2147483647.0, 4294967291.0, 3.0))
    for frequency in (Fraction(0), Fraction(-1)):
        with pytest.raises(ValueError, match="positive"):
            simulation.TimeBase(frequency)


def test_fixture_times_must_be_finite() -> None:
    time_base = simulation.TimeBase(Fraction(1))
    for time_us in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            time_base.ticks(time_us)
