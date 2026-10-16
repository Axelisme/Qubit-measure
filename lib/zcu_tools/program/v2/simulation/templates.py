"""Flat-top lowering and detached slot-array selection for frontend templates."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    EventSlot,
    PulseEnvelope,
    TimelineTemplate,
)


@dataclass(frozen=True)
class _FlatTop:
    envelope: PulseEnvelope
    ramp_us: float
    period_us: float
    half: int

    def whole(self, duration: float) -> PulseEnvelope:
        """Rebuild the scalar interpreter's compact two-ramp sample axis."""
        axis = np.asarray(self.envelope.sample_us[: self.half])
        times = np.concatenate(
            (
                axis,
                [self.ramp_us, duration - self.ramp_us - self.period_us],
                axis + duration - self.ramp_us,
            )
        )
        return PulseEnvelope(tuple(float(t) for t in times), self.envelope.values)

    def ramps(self) -> tuple[PulseEnvelope, PulseEnvelope]:
        """Keep linear transitions to/from the half-gain plateau at boundaries."""
        axis = self.envelope.sample_us[: self.half]
        values = self.envelope.values
        up = PulseEnvelope((*axis, self.ramp_us), (*values[: self.half], 0.5 + 0j))
        down = PulseEnvelope(
            (0.0, *(t + self.period_us for t in axis)),
            (0.5 + 0j, *values[self.half + 2 :]),
        )
        return up, down


def take_slot(slot: EventSlot, points: NDArray[np.int64]) -> EventSlot:
    """Detach a slot's point arrays in the supplied sorted point order."""
    return replace(
        slot,
        start_us=slot.start_us[points].copy(),
        duration_us=slot.duration_us[points].copy(),
        frequency_mhz=slot.frequency_mhz[points].copy(),
        phase_deg=slot.phase_deg[points].copy(),
        gain=slot.gain[points].copy(),
        envelope_index=slot.envelope_index[points].copy(),
        active=slot.active[points].copy(),
    )


def select_candidates(
    slots: list[EventSlot], index: NDArray[np.int64], *, source: str
) -> EventSlot:
    """Gather selected pulse parameters into one slot with candidate envelopes.

    slots must share generator, phase origin and iteration position. index is an
    int64 candidate index per point. Source is the selecting semantic Module.
    """

    def select(values: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        return np.take_along_axis(np.stack(values), index[None, :], axis=0)[0]

    return replace(
        slots[0],
        source=source,
        start_us=select([s.start_us for s in slots]),
        duration_us=select([s.duration_us for s in slots]),
        frequency_mhz=select([s.frequency_mhz for s in slots]),
        phase_deg=select([s.phase_deg for s in slots]),
        gain=select([s.gain for s in slots]),
        envelopes=tuple(s.envelopes[0] for s in slots),
        envelope_index=index.copy(),
    )


class FlatTopEnvelopes:
    """Own compiled flat-top candidates and lower them without point loops.

    Long plateaus become ramp/constant/ramp slots. Whole short envelopes have
    fewer than 10 ns of plateau, bounding their duration candidates by the DAC
    clock, not the sweep size. Candidate shapes also supply short-plateau envelopes.
    """

    def __init__(self) -> None:
        self._candidates: dict[int, _FlatTop] = {}

    def record(self, envelope: PulseEnvelope) -> None:
        """Register a detached flat-top envelope produced by the scalar frontend."""
        half = (len(envelope.values) - 2) // 2
        period = envelope.sample_us[1] - envelope.sample_us[0]
        self._candidates[id(envelope)] = _FlatTop(
            envelope, envelope.sample_us[half], period, half
        )

    def _shapes(self, slot: EventSlot) -> list[_FlatTop]:
        return [self._candidates[id(e)] for e in slot.envelopes]

    def long_plateau(self, slot: EventSlot) -> NDArray[np.bool_] | None:
        """Return the >=10 ns structural mask, or None for a non-flat pulse."""
        if id(slot.envelopes[0]) not in self._candidates:
            return None
        ramps = np.asarray([s.ramp_us for s in self._shapes(slot)])[slot.envelope_index]
        return slot.duration_us - 2 * ramps >= 0.010

    def lower(self, template: TimelineTemplate) -> TimelineTemplate:
        """Lower flat-tops in a class already partitioned by long_plateau masks."""
        events = []
        for slot in template.events:
            long = self.long_plateau(slot)
            if long is None:
                events.append(slot)
            elif np.all(long):
                events.extend(self._long(slot))
            else:
                if np.any(long):
                    raise ValueError("flat-top structural class mixes plateau kinds")
                events.append(self._short(slot))
        return replace(template, events=tuple(events))

    def _long(self, slot: EventSlot) -> tuple[EventSlot, EventSlot, EventSlot]:
        shapes = self._shapes(slot)
        ramps = np.asarray([s.ramp_us for s in shapes])[slot.envelope_index]
        periods = np.asarray([s.period_us for s in shapes])[slot.envelope_index]
        envelopes = [s.ramps() for s in shapes]
        up = replace(slot, duration_us=ramps, envelopes=tuple(e[0] for e in envelopes))
        plateau = replace(
            slot,
            start_us=slot.start_us + ramps,
            duration_us=slot.duration_us - 2 * ramps - periods,
            envelopes=(None,),
            envelope_index=np.zeros_like(slot.envelope_index),
            gain=slot.gain * 0.5,
        )
        down = replace(
            slot,
            start_us=slot.start_us + slot.duration_us - ramps - periods,
            duration_us=ramps + periods,
            envelopes=tuple(e[1] for e in envelopes),
        )
        return up, plateau, down

    def _short(self, slot: EventSlot) -> EventSlot:
        envelopes = []
        indices = np.zeros_like(slot.envelope_index)
        for candidate, shape in enumerate(self._shapes(slot)):
            chosen = slot.envelope_index == candidate
            for duration in np.unique(slot.duration_us[chosen]):
                indices[chosen & (slot.duration_us == duration)] = len(envelopes)
                envelopes.append(shape.whole(float(duration)))
        return replace(slot, envelopes=tuple(envelopes), envelope_index=indices)
