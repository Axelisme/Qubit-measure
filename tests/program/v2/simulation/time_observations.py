"""Test-only physical-time observations of the public integer request schema."""

from fractions import Fraction

from zcu_tools.core.simulation import PulseEnvelope, TimeBase


def time_us(base: TimeBase, ticks: int) -> float:
    """Project one tick observation to us using an independent rational oracle."""
    return float(Fraction(int(ticks)) / base.ticks_per_us)


def times_us(base: TimeBase, ticks: tuple[int, ...]) -> tuple[float, ...]:
    """Project trigger observations to us, preserving their emission order."""
    return tuple(time_us(base, tick) for tick in ticks)


def envelope_times_us(base: TimeBase, envelope: PulseEnvelope) -> tuple[float, ...]:
    """Observe the uniform envelope axis in us without production conversion."""
    return tuple(
        time_us(base, index * envelope.sample_period_ticks)
        for index in range(len(envelope.values))
    )
