from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qick.asm_v2 import QickParam

from .base import Module

if TYPE_CHECKING:
    from zcu_tools.program.v2.timeline import TimelineProgramV2


logger = logging.getLogger(__name__)


class Delay(Module):
    def __init__(
        self, name: str, delay: float | QickParam
    ) -> None:
        self.name = name
        self.delay = delay



    def declare(self, prog: TimelineProgramV2) -> None:
        """Delay needs no hardware resources; prog owns the emission timeline."""

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Ceil the selected end plus delay us; do not move reference."""
        prog.timeline_inc(
            self.delay,
            timeline=timeline if timeline is not None else "default",
            tag=tag,
        )



class SoftDelay(Module):
    def __init__(self, name: str, delay: float | QickParam) -> None:
        self.name = name
        self.delay = delay



    def declare(self, prog: TimelineProgramV2) -> None:
        """SoftDelay needs no hardware resources on prog."""

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Ceil the selected end plus delay us, exactly like Delay in this path."""
        prog.timeline_inc(
            self.delay,
            timeline=timeline if timeline is not None else "default",
            tag=tag,
        )



class DelayAuto(Module):
    """Wait for selected channel timelines and then advance the reference time.

    name identifies the module. t is extra time in microseconds (including QickParam
    sweeps), or a register name holding tProc cycles. gens and ros select generator
    and readout timelines. At least one must be True; otherwise construction raises
    ValueError. tag names a time parameter and cannot be used with a register name.
    """

    def __init__(
        self,
        name: str,
        t: float | QickParam | str = 0.0,
        *,
        gens: bool = True,
        ros: bool = True,
    ) -> None:
        if not gens and not ros:
            raise ValueError("DelayAuto requires at least one channel: gens or ros")

        self.name = name
        self.t = t
        self.gens = gens
        self.ros = ros




    def declare(self, prog: TimelineProgramV2) -> None:
        """DelayAuto consumes existing channel/register declarations on prog."""

    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Move native reference by selected channel ends plus t, not a duration.

        timeline must be None; explicit names (including default) raise ValueError
        at emission. Register t holds nonnegative tProc cycles and requires every
        timeline end <= static advance. Scalar/QickParam t is extra us.
        """
        if timeline is not None:
            raise ValueError(
                "DelayAuto cannot select a timeline; use native delay_auto or timeline_inc"
            )
        if isinstance(self.t, str):
            if tag is not None:
                raise ValueError("DelayAuto with a register time cannot have a tag")
            prog.delay_reg_auto(time_reg=self.t, gens=self.gens, ros=self.ros)
        else:
            prog.delay_auto(t=self.t, gens=self.gens, ros=self.ros, tag=tag)



