"""Batch source-line filtering and F1/C/F2 lattice sampling in engine units."""

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.evolution import (
    DriveTrack,
    EnvelopeSpec,
    FluxSchedule,
    FluxWaveform,
)
from zcu_tools.simulate.lines import FluxLine

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
DELTA = 1e-3


@dataclass(frozen=True)
class FluxTrack:
    """One flux-source slot across points, with owned read-only arrays.

    source: nonempty FluxLine.source name. start_ns/duration_ns: nonnegative
    (points,) pulse support; start and start+duration round independently to
    nearest 2 ns, ties even, matching FilteredFlux. frequency_ghz: signed
    carrier cycles/ns. phase_rad:
    phase at absolute source time zero (before line delay), in radians.
    amplitude: signed real source gain. These five arrays are finite float64.
    envelopes: candidate EnvelopeSpecs. envelope_index: int64 (points,) index
    into envelopes. active: bool (points,) source-activity mask. Sampled complex
    envelopes interpolate linearly and hold their last value through pulse end.
    The real lab-frame signal passes through delay/lowpass before loop mixing.
    Invalid support, shapes, finite values or indices raise ValueError.
    """

    source: str
    start_ns: RealArray
    duration_ns: RealArray
    frequency_ghz: RealArray
    phase_rad: RealArray
    amplitude: RealArray
    envelopes: tuple[EnvelopeSpec, ...]
    envelope_index: NDArray[np.int64]
    active: NDArray[np.bool_]

    def __post_init__(self) -> None:
        # The microwave and flux records share playback and array invariants.
        validated = DriveTrack(
            self.source,
            self.start_ns,
            self.duration_ns,
            self.frequency_ghz,
            self.phase_rad,
            self.amplitude,
            self.envelopes,
            self.envelope_index,
            self.active,
        )
        for item in fields(DriveTrack):
            if item.name != "port":
                object.__setattr__(self, item.name, getattr(validated, item.name))


def _validate(
    tracks: tuple[FluxTrack, ...],
    dc_flux: RealArray,
    lines: tuple[FluxLine, ...],
    sample_period_ns: float,
    points: int,
) -> RealArray:
    dc = np.asarray(dc_flux, dtype=float)
    if dc.shape != (3,) or not np.isfinite(dc).all():
        raise ValueError("DC flux must be finite F1/C/F2 shape (3,)")
    if not np.isfinite(sample_period_ns) or sample_period_ns <= 0:
        raise ValueError("DAC sample period must be positive finite ns")
    if type(points) is not int or points < 0:
        raise ValueError("points must be a nonnegative integer count")
    names = {line.source for line in lines}
    if len(names) != len(lines):
        raise ValueError("flux line source names must be unique")
    for i, track in enumerate(tracks):
        if track.source not in names:
            raise ValueError(f"unknown flux source {track.source}")
        if track.start_ns.shape != (points,):
            raise ValueError("flux track shape must match points")
        for other in tracks[:i]:
            if other.source == track.source:
                a0, a1 = _support(track)
                b0, b1 = _support(other)
                overlap = track.active & other.active & (a0 < b1) & (b0 < a1)
                if np.any(overlap & (a1 > a0) & (b1 > b0)):
                    raise ValueError("same-source flux pulses must not overlap")
    return dc


def _support(track: FluxTrack) -> tuple[RealArray, RealArray]:
    return 2 * np.rint(track.start_ns / 2), 2 * np.rint(
        (track.start_ns + track.duration_ns) / 2
    )


def _terminal_times(
    tracks: tuple[FluxTrack, ...],
    lines: tuple[FluxLine, ...],
    step: float,
    points: int,
) -> RealArray:
    """Conservative absolute per-pulse bound, valid despite signs/cancellation."""
    horizon = np.zeros(points)
    active = np.zeros(points, dtype=bool)
    by_source = {line.source: line for line in lines}
    for track in tracks:
        line = by_source[track.source]
        start, end = _support(track)
        mask = track.active & (end > start)
        active |= mask
        maxima = np.array(
            [
                1.0 if env.kind == "constant" else np.max(np.abs(env.samples))
                for env in track.envelopes
            ]
        )
        bound = (
            np.abs(track.amplitude)
            * maxima[track.envelope_index]
            * np.max(np.abs(line.coefficients))
        )
        # Allocate the absolute error budget to pulses, not their signed sum.
        extra = line.tau_ns * np.log(np.maximum(1.0, bound * len(tracks) / (DELTA / 2)))
        horizon = np.maximum(horizon, np.where(mask, end + line.delay_ns + extra, 0.0))
    return np.where(active, (np.floor(horizon / step) + 1) * step, 0.0)


def _moments(length: RealArray, lam: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    x = lam * length
    decay = np.exp(-x)
    first = -np.expm1(-x) / lam
    # The series avoids cancellation in the integral of u*exp(-lambda*u).
    numerator = np.where(
        np.abs(x) < 1e-3,
        x**2 * (0.5 - x / 3 + x**2 / 8 - x**3 / 30),
        -np.expm1(-x) - x * decay,
    )
    return first, numerator / lam**2


def _filtered_track(track: FluxTrack, line: FluxLine, times: RealArray) -> RealArray:
    """Exact convolution of linear complex envelope pieces times a carrier."""
    start, end = _support(track)
    duration = (end - start)[:, None]
    local = times[None, :] - line.delay_ns - start[:, None]
    omega = 2 * np.pi * track.frequency_ghz[:, None]
    phase = track.phase_rad[:, None] + omega * start[:, None]
    output: ComplexArray = np.zeros(local.shape, dtype=np.complex128)
    for index, env in enumerate(track.envelopes):
        selected = track.active & (track.envelope_index == index)
        if line.tau_ns == 0:
            values = env.values(np.maximum(local, 0))
            output += np.where(
                selected[:, None] & (local >= 0) & (local < duration),
                values * np.exp(1j * (phase + omega * local)),
                0,
            )
            continue
        knots = np.array([0.0]) if env.kind == "constant" else env.sample_ns
        values = np.array([1.0 + 0j]) if env.kind == "constant" else env.samples
        lam = np.asarray(1 / line.tau_ns + 1j * omega, dtype=np.complex128)
        for k, lo in enumerate(knots):
            hi = knots[k + 1] if k + 1 < len(knots) else np.inf
            slope = (values[k + 1] - values[k]) / (hi - lo) if np.isfinite(hi) else 0j
            upper = np.maximum(lo, np.minimum(np.minimum(local, duration), hi))
            length = upper - lo
            first, second = _moments(length, lam)
            amplitude = values[k] + slope * (upper - lo)
            # Upper-end factoring prevents overflow on long pulses/tails.
            contribution = (
                np.exp(1j * (phase + omega * upper))
                * np.exp(-np.maximum(local - upper, 0) / line.tau_ns)
                * (amplitude * first - slope * second)
                / line.tau_ns
            )
            output += np.where(selected[:, None], contribution, 0)
    return output.real * track.amplitude[:, None]


def _sample(
    tracks: tuple[FluxTrack, ...],
    dc: RealArray,
    lines: tuple[FluxLine, ...],
    times: RealArray,
    points: int,
) -> RealArray:
    flux = np.broadcast_to(dc, (points, len(times), 3)).copy()
    by_source = {line.source: line for line in lines}
    for track in tracks:
        line = by_source[track.source]
        flux += _filtered_track(track, line, times)[:, :, None] * np.array(
            line.coefficients
        )
    return flux


def flux_waveform(
    tracks: tuple[FluxTrack, ...],
    *,
    dc_flux: RealArray,
    lines: tuple[FluxLine, ...],
    sample_period_ns: float,
    points: int,
) -> FluxWaveform:
    """Filter a template to ragged absolute F1/C/F2 flux on DAC sample nodes.

    points is the integer count, in track-array order. Each point starts at zero;
    sample_period_ns is the positive DAC spacing. Source delay/lowpass precedes
    loop mixing. Samples retain unsnapped filtered values. A terminal DC sample
    is appended only after a conservative sum-of-absolute-pulse-tails bound is
    below delta/2 = 5e-4 Phi0 on every loop. Empty/inactive points contain DC at
    t=0 only. Invalid shapes, unknown sources or same-source overlaps raise
    ValueError. No state survives calls, and no per-point quadrature is used.
    """
    dc = _validate(tracks, dc_flux, lines, sample_period_ns, points)
    terminal = _terminal_times(tracks, lines, sample_period_ns, points)
    times = np.asarray(
        np.arange(int(np.max(terminal, initial=0) / sample_period_ns) + 1)
        * sample_period_ns,
        dtype=np.float64,
    )
    flux = _sample(tracks, dc, lines, times, points)
    keep = times[None, :] <= terminal[:, None]
    flux[times[None, :] == terminal[:, None]] = dc
    counts = keep.sum(axis=1)
    return FluxWaveform(
        np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
        np.broadcast_to(times, keep.shape)[keep],
        flux[keep],
    )


def schedule(
    tracks: tuple[FluxTrack, ...],
    *,
    dc_flux: RealArray,
    lines: tuple[FluxLine, ...],
    sample_period_ns: float,
    points: int,
) -> FluxSchedule:
    """Filter all points, quantize DAC-interval midpoints and merge equal levels.

    Inputs match flux_waveform. Levels are nearest integers (ties even) of
    (flux-dc_flux)/delta, delta=1e-3 Phi0. Offsets within delta/2 snap to DC.
    Signed source signals pass through their own delay/lowpass before summing
    loop flux. Tail truncation uses a conservative absolute pulse-tail sum below
    delta/2 = 5e-4 Phi0 on every loop, never a relative or signed criterion.
    First segment starts at zero, last level is DC; inactive points have only
    one DC segment. Raises ValueError for the same invalid inputs as waveform.
    """
    dc = _validate(tracks, dc_flux, lines, sample_period_ns, points)
    terminal = _terminal_times(tracks, lines, sample_period_ns, points)
    times = np.asarray(
        np.arange(int(np.max(terminal, initial=0) / sample_period_ns) + 1)
        * sample_period_ns,
        dtype=np.float64,
    )
    flux = _sample(tracks, dc, lines, times + sample_period_ns / 2, points)
    levels = np.rint((flux - dc) / DELTA).astype(np.int64)
    levels[times[None, :] >= terminal[:, None]] = 0
    changes = np.ones((points, len(times)), dtype=bool)
    changes[:, 1:] = np.any(levels[:, 1:] != levels[:, :-1], axis=2)
    return FluxSchedule(
        np.concatenate(([0], np.cumsum(changes.sum(axis=1)))).astype(np.int64),
        np.broadcast_to(times, changes.shape)[changes],
        levels[changes],
    )
