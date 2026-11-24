"""Eligibility for reusing vacuum projective continuation rows."""

from fractions import Fraction
from typing import NamedTuple

from zcu_tools.simulate.adc import ReadoutPulse
from zcu_tools.simulate.evolution import TimelineBatch


class FlatCell(NamedTuple):
    """Exact cell key with pulse support relative to the cell start."""

    duration_ns: Fraction
    pulse_start_ns: Fraction
    pulse_end_ns: Fraction
    frequency_ghz: float
    amplitude: complex


def _single_incident(
    pulses: tuple[ReadoutPulse, ...], start: float, end: float
) -> ReadoutPulse | None:
    incident = [p for p in pulses if p.start_ns < end and start < p.end_ns]
    if len(incident) != 1:
        return None
    (pulse,) = incident
    if not start <= pulse.start_ns < pulse.end_ns <= end:
        return None
    return pulse if pulse.amplitude != 0 and not pulse.breakpoints_ns else None


def flat_cell_keys(
    batch: TimelineBatch,
    spans: tuple[tuple[float, float], ...],
    pulses: tuple[ReadoutPulse, ...],
    boundaries: tuple[int, ...],
) -> dict[int, FlatCell]:
    """Return exact local keys indexed by eligible interior projective event.

    The caller supplies a one-point batch, its final Engine spans and all its
    cavity inputs. Boundaries index the starts of its vacuum-reset projection
    cells; the last cell, which ends at terminal readout, is always excluded.
    """
    if (
        batch.flux_waveform is not None
        or len(batch.flux.start_ns) != 1
        or any(p.envelope is not None for p in pulses)
    ):
        return {}
    # Engine decides whether a drive is effective. Reuse excludes even a
    # zero-gain track when its declared nonempty support overlaps the cell.
    # Keep the declared endpoints exact so rounding cannot hide overlap.
    drives = []
    for track in batch.tracks:
        if track.active[0] and track.duration_ns[0] > 0:
            start = Fraction(float(track.start_ns[0]))
            drives.append((start, start + Fraction(float(track.duration_ns[0]))))
    keys = {}
    for event, (first, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        if stop - first != 1:
            continue
        start, end = spans[first]
        if any(a < end and start < b for a, b in drives):
            continue
        pulse = _single_incident(pulses, start, end)
        if pulse is None:
            continue
        # One tone can only change by a common phase. Vacuum input and a
        # single span mean no inherited or intermediate branch field to match.
        # Keep exact differences of binary times, not rounded relative times.
        origin = Fraction(float(start))
        keys[event] = FlatCell(
            Fraction(float(end)) - origin,
            Fraction(float(pulse.start_ns)) - origin,
            Fraction(float(pulse.end_ns)) - origin,
            pulse.frequency_ghz,
            pulse.amplitude,
        )
    return keys
