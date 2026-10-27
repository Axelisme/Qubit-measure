"""Environment-owned physical time, independent of acquisition pacing speed."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationTiming:
    """Pacing policy and physical clock, fixed when a binding is constructed.

    time_scale: finite nonnegative multiplier of the physical round duration
        used only for pacing sleep, default zero. One paces at physical speed.
    clock: environment-owned shared clock, or None to skip physical accounting.
        Its offset always uses unscaled duration, even when time_scale is zero.
        Binding never closes or resets it. No hardware acquisition uses it.
    """

    time_scale: float = 0.0
    clock: "SimulationClock | None" = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_scale) or self.time_scale < 0:
            raise ValueError("time_scale must be finite and nonnegative")


class SimulationClock:
    """Monotonic seconds plus a nondecreasing physical-duration offset.

    monotonic is a no-argument wall elapsed-seconds source, defaulting to
    time.monotonic. It must never decrease. One environment shares one clock.
    Construction resets the offset to zero; idle time advances naturally.
    """

    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._offset_s = 0.0

    def now(self) -> float:
        """Return current simulated seconds (wall monotonic plus offset)."""
        return self._monotonic() + self._offset_s

    def complete_round(self, physical_s: float, elapsed_s: float) -> None:
        """Credit max(0, physical_s - elapsed_s) after a completed round.

        Both inputs are nonnegative seconds. elapsed_s includes snapshot,
        execution and pacing sleep. Do not call for cancelled or failed rounds.
        """
        self._offset_s += max(0.0, physical_s - elapsed_s)
