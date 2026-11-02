"""Compile-time pulse timestamp transfers and pointwise sweep proofs.

These summaries validate Repeat backedges only. They never update native
channel timestamps, timeline ends, reference time or generated instructions.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from qick.asm_v2 import QickParam

from .table_param import TableParam


def bounds(value: float | QickParam) -> tuple[float, float]:
    """Return minimum/maximum us over every combination of linear sweep spans."""
    if isinstance(value, TableParam):
        return value.minval(), value.maxval()
    if isinstance(value, QickParam):
        return (
            value.start + sum(min(0.0, s) for s in value.spans.values()),
            value.start + sum(max(0.0, s) for s in value.spans.values()),
        )
    return value, value


def _guard_candidates(values: tuple[float | QickParam, ...]):
    """Keep the pointwise maximum envelope without choosing crossing sweeps."""
    result: list[float | QickParam] = []
    for value in values:
        if any(bounds(old - value)[0] >= -1e-12 for old in result):
            continue
        result = [old for old in result if bounds(value - old)[0] < -1e-12]
        result.append(value)
    return tuple(result)


@dataclass
class PulseGuard:
    """Validation-only timestamp transfer and managed-pulse entry limits.

    F(x) = max(x + shifts, floors). Multiple candidates retain crossing sweep
    envelopes exactly, rather than choosing one sweep endpoint for every point.
    Every limit requires the entry timestamp x <= limit. No scheduling state is
    changed here; native Pulse still owns timestamp updates.
    """

    channel: int  # Logical generator channel, used in overlap diagnostics.
    shifts: tuple[float | QickParam, ...] = (0.0,)  # Entry-dependent offsets in us.
    floors: tuple[float | QickParam, ...] = ()  # Entry-independent lower ends in us.
    limits: tuple[float | QickParam, ...] = ()  # Required entry upper bounds in us.

    unbounded: bool = False  # Author-owned segment cannot supply overlap proofs.

    def require(
        self, ends: Iterable[float | QickParam], limits: Iterable[float | QickParam]
    ) -> None:
        """Raise overlap if any end exceeds any limit at any sweep point (us)."""
        if self.unbounded:
            return
        for end in ends:
            for limit in limits:
                if bounds(limit - end)[0] < -1e-12:
                    raise ValueError(
                        f"repeat: channel {self.channel} pulse overlap; cannot prove "
                        "start >= previous end at every sweep point; use timeline_inc"
                    )

    def then(self, other: "PulseGuard") -> None:
        """Compose a following transfer, pulling its guards back to our entry."""
        if self.unbounded or other.unbounded:
            self.unbounded = True
            return
        self.require(self.floors, other.limits)
        limits = self.limits + tuple(
            limit - shift for limit in other.limits for shift in self.shifts
        )
        self.limits = tuple(-v for v in _guard_candidates(tuple(-v for v in limits)))
        self.floors = _guard_candidates(
            other.floors
            + tuple(end + shift for end in self.floors for shift in other.shifts)
        )
        self.shifts = _guard_candidates(
            tuple(a + b for a in self.shifts for b in other.shifts)
        )

    def repeated(self, minimum: int, maximum: int) -> "PulseGuard":
        """Compose every possible count without unrolling iterations.

        F^n(x) has shifts n*a and floors b + max(0, (n-1)*a).
        Guards at entries k=0..maximum-1 need x+k*a <= limit;
        their constant floors need b+max(0, (maximum-2)*a) <= limit.
        Each expression is affine in k, so its endpoints cover all interior
        integer counts at each sweep point, even when candidate sweeps cross.
        """
        if self.unbounded:
            return PulseGuard(self.channel, unbounded=True)
        if maximum == 0:
            return PulseGuard(self.channel)
        if maximum >= 2:
            self.require(
                self.floors
                + tuple(
                    end + (maximum - 2) * shift
                    for end in self.floors
                    for shift in self.shifts
                ),
                self.limits,
            )
        limits = self.limits + tuple(
            limit - (maximum - 1) * shift
            for limit in self.limits
            for shift in self.shifts
        )
        return PulseGuard(
            self.channel,
            _guard_candidates(
                tuple(n * a for n in (minimum, maximum) for a in self.shifts)
            ),
            _guard_candidates(
                self.floors
                + tuple(
                    end + (maximum - 1) * shift
                    for end in self.floors
                    for shift in self.shifts
                )
            ),
            tuple(-v for v in _guard_candidates(tuple(-v for v in limits))),
        )
