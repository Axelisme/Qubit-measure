"""Compiled-template planning for one fluxonium without density propagation."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import EnvironmentSnapshot, TimelineTemplate
from zcu_tools.experiment.readout_plan import (
    ReadoutContents,
    ReadoutDrive,
    classify_readout,
    drive_track,
    readout_contents,
)
from zcu_tools.resources.single_fluxonium.schema import SingleFluxoniumSettings
from zcu_tools.simulate.adc import ReadoutPulse
from zcu_tools.simulate.evolution import (
    ControlPulse,
    ControlSequence,
    DriveTrack,
    Envelope,
    EnvelopeSpec,
    FluxSchedule,
    FluxWaveform,
    TimelineBatch,
)
from zcu_tools.simulate.lines import DistortionTerm, FilteredFlux, FluxLine

from .numeric import dc_flux


@dataclass(frozen=True)
class SingleFluxoniumPlan:
    """Detached numeric inputs for all points of one structural template.

    batch: one complete ns/GHz/radian TimelineBatch, ending at terminal readout.
        Only microwave events enter tracks. One-loop flux includes filtered tails
        until readout; the engine alone segments/evolves it.
    readout: distinct terminal pulse/grid contents, common-origin phases, without
        a physical ADC model; apply local_readout_pulses before ADC use.
    cavity_drives: earlier readout-generator pulses per point, absolute ns/GHz,
        pulse-local radians with zero-GHz reference, retained but not solved.
    No model, density or random samples are built. Consumers must not modify
    these values. Execution supports terminal readout only and rejects earlier
    cavity drives; their Stark/backaction physics is not modeled here.
    """

    batch: TimelineBatch
    readout: ReadoutContents
    cavity_drives: tuple[tuple[ReadoutPulse, ...], ...]


def _controls(track: DriveTrack, point: int) -> ControlPulse:
    spec = track.envelopes[track.envelope_index[point]]
    envelope = (
        Envelope.flat(complex(track.amplitude[point]))
        if spec.kind == "constant"
        else Envelope.sampled(
            spec.sample_ns, spec.samples * track.amplitude[point], outside="hold"
        )
    )
    start = float(track.start_ns[point])
    frequency = float(track.frequency_ghz[point])
    return ControlPulse(
        track.port,
        start,
        start + float(track.duration_ns[point]),
        envelope,
        frequency,
        float(track.phase_rad[point]) + 2 * np.pi * frequency * start,
        breakpoints_ns=tuple(
            start + t for t in spec.sample_ns if 0 < t < track.duration_ns[point]
        ),
    )


def _point_flux(
    settings: SingleFluxoniumSettings,
    dc: float,
    tracks: tuple[DriveTrack, ...],
    point: int,
    end: float,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]
]:
    line = settings.flux_line
    if line is None:
        raise ValueError("no fast-flux line is bound")
    controls = ControlSequence(
        tuple(_controls(t, point) for t in tracks if t.active[point])
    )
    filtered = FilteredFlux(
        np.array([dc]),
        controls,
        (
            FluxLine(
                line.name,
                (line.flux_phi0_per_gain,),
                line.delay_ns,
                line.tau_ns,
                tuple(
                    DistortionTerm(t.amplitude, t.tau_ns) for t in line.distortion_terms
                ),
            ),
        ),
        rtol=settings.numerics.solver.rtol,
        atol=settings.numerics.solver.atol,
    )
    # Sample the complete history to readout, never impose a tail cutoff.
    # Include event sides so instantaneous feedthrough is not spread over 2 ns.
    events = [t for t in filtered.event_times_ns if 0 < t <= end]
    times = np.unique(
        np.r_[
            np.arange(0, end, 2.0),
            end,
            events,
            [np.nextafter(t, -np.inf) for t in events],
        ]
    )
    values = np.array([filtered.at(float(t))[0] for t in times])
    starts = np.unique(
        np.r_[np.arange(0, end, settings.numerics.solver.bath_segment_ns), 0.0, events]
    )
    starts = starts[(starts < end) | (starts == 0)]
    stops = np.r_[starts[1:], end]
    levels = np.rint(
        (np.array([filtered.at(float(t))[0] for t in (starts + stops) / 2]) - dc) / 1e-3
    ).astype(np.int64)
    changed = np.r_[True, levels[1:] != levels[:-1]]
    return times, values, starts[changed], levels[changed]


def _flux_inputs(
    settings: SingleFluxoniumSettings,
    dc: float,
    tracks: tuple[DriveTrack, ...],
    ends: NDArray[np.float64],
) -> tuple[FluxSchedule, FluxWaveform | None]:
    if settings.flux_line is None:
        return FluxSchedule(
            np.arange(len(ends) + 1, dtype=np.int64),
            np.zeros(len(ends)),
            np.zeros((len(ends), 1), dtype=np.int64),
        ), None
    offsets, waveform_offsets = [0], [0]
    starts, levels, times, values = [], [], [], []
    for point, end in enumerate(ends):
        sample_times, samples, point_starts, point_levels = _point_flux(
            settings, dc, tracks, point, float(end)
        )
        starts.extend(point_starts)
        levels.extend(point_levels)
        times.extend(sample_times)
        values.extend(samples)
        offsets.append(len(starts))
        waveform_offsets.append(len(times))
    flux = FluxSchedule(
        np.array(offsets, dtype=np.int64),
        np.array(starts),
        np.array(levels, dtype=np.int64).reshape(-1, 1),
    )
    waveform = FluxWaveform(
        np.array(waveform_offsets, dtype=np.int64),
        np.array(times),
        np.array(values).reshape(-1, 1),
    )
    line = settings.flux_line
    continuous = bool(line.tau_ns or line.distortion_terms) or any(
        np.any(t.frequency_ghz[t.active] != 0)
        or any(
            e.kind == "sampled" and np.any(e.samples != e.samples[0])
            for e in t.envelopes
        )
        for t in tracks
    )
    return flux, waveform if tracks and continuous else None


def plan_template(
    settings: SingleFluxoniumSettings,
    snapshot: EnvironmentSnapshot,
    template: TimelineTemplate,
    *,
    readout_kind: Literal["pulse", "mux"] = "pulse",
) -> SingleFluxoniumPlan:
    """Convert one complete QICK template to evolution and terminal readout inputs.

    settings is validated physical/wiring data; snapshot supplies exactly the live
    DC source value in its native A/V unit. template arrays use us/MHz/degrees.
    Support uses the shared nearest-2-ns, ties-even conversion. Only the bound
    microwave generator enters tracks. Fast-flux controls pass through
    FilteredFlux including delay, signed complex envelopes and all exponential
    tails. Absolute Phi0 waveform samples use a 2 ns grid plus event sides and
    readout endpoint; interpolation is linear. Quantized bath levels use
    midpoint flux at solver.bath_segment_ns intervals plus delayed events,
    rounded ties-even to 1e-3 Phi0 relative to DC; equal adjacent levels merge.
    Ideal constant platforms omit waveform to retain abrupt level changes.
    No forced return to DC or physics calculation occurs. All original points
    occupy one batch, even when their readout times differ. Engine owns segmenting.

    Exactly one configured receiver and terminal readout per point are required.
    readout_kind="pulse" requires one tone; "mux" accepts 1..8 coherent tones
    with common start. A None flux_line permits only DC flux and microwaves.
    Earlier readout pulses are retained separately, not treated as microwaves.
    Unbound generators/receivers, invalid DC, crossing readout blocks or invalid
    numeric records raise ValueError. No ADC solver, density or RNG is used.
    Returned contents are detached; input buffers may subsequently be reused.
    """
    if (
        len(template.readouts) != 1
        or template.readouts[0].channel != settings.readout.adc.ro_ch
    ):
        raise ValueError("single fluxonium requires its one bound receiver")
    dc = dc_flux(settings, snapshot)
    memo: dict[int, EnvelopeSpec] = {}
    drive = ReadoutDrive(
        settings.readout.drive.gen_ch,
        settings.readout.drive.input_per_gain,
        readout_kind,
    )
    partition = classify_readout(template, drive, memo)
    contents = readout_contents(
        replace(template, events=partition.terminal_slots),
        drive,
        template.readouts[0],
        memo,
    )
    microwave, fast_flux = [], []
    for event in template.events:
        if event.gen_ch == drive.gen_ch:
            continue
        if event.gen_ch == settings.drive_line.gen_ch:
            microwave.append(drive_track(event, settings.drive_line.name, memo))
        elif (
            settings.flux_line is not None and event.gen_ch == settings.flux_line.gen_ch
        ):
            fast_flux.append(drive_track(event, settings.flux_line.name, memo))
        else:
            raise ValueError(f"unbound single-fluxonium generator {event.gen_ch}")
    flux, waveform = _flux_inputs(settings, dc, tuple(fast_flux), contents.end_ns)
    return SingleFluxoniumPlan(
        TimelineBatch(
            template.points, tuple(microwave), flux, contents.end_ns, waveform
        ),
        contents,
        partition.cavity_drives,
    )
