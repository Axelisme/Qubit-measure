"""Compiled-template planning for one fluxonium without density propagation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    ChargeDriveRequirement,
    EnvironmentSnapshot,
    TimeBase,
    TimelineTemplate,
)
from zcu_tools.experiment.readout_plan import (
    ProjectiveRead,
    ReadoutContents,
    ReadoutDrive,
    classify_readout,
    drive_track,
    logical_event_contents,
)
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
from zcu_tools.simulators.single_fluxonium.config.schema import SingleFluxoniumSettings
from zcu_tools.simulators.single_fluxonium.environment.numeric import dc_flux

# Sample filtered fast-flux waveforms on the existing 2 ns DAC grid.
_WAVEFORM_SAMPLE_NS = 2.0
# Quantize bath midpoint offsets relative to DC in reduced-flux units.
_FLUX_LATTICE_PHI0 = 1e-3


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
    these values. Triggered earlier blocks project the DC density at their
    starts. Execution rejects untriggered cavity drives; their Stark/backaction
    physics is not modeled here.
    """

    batch: TimelineBatch
    readout: ReadoutContents
    cavity_drives: tuple[tuple[ReadoutPulse, ...], ...]
    read_events: tuple[ReadoutContents, ...]
    """Ordered projective blocks and terminal, including unobserved grids."""
    projective_reads: tuple[tuple[ProjectiveRead, ...], ...]
    """Per-point DC rank-1 projection times and associated physical blocks."""


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
            np.arange(0, end, _WAVEFORM_SAMPLE_NS),
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
        (np.array([filtered.at(float(t))[0] for t in (starts + stops) / 2]) - dc)
        / _FLUX_LATTICE_PHI0
    ).astype(np.int64)
    changed = np.r_[True, levels[1:] != levels[:-1]]
    return times, values, starts[changed], levels[changed]


def _flux_inputs(
    settings: SingleFluxoniumSettings,
    dc: float,
    tracks: tuple[DriveTrack, ...],
    ends: NDArray[np.float64],
) -> tuple[FluxSchedule, FluxWaveform | None]:
    # No active controls means exact DC, regardless of line delay or tails.
    if settings.flux_line is None or not any(np.any(t.active) for t in tracks):
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
    time_base: TimeBase,
    readout_kind: Literal["pulse", "mux"] = "pulse",
    charge_requirements: tuple[ChargeDriveRequirement, ...] = (),
) -> SingleFluxoniumPlan:
    """Convert one complete QICK template to evolution and logical read events.

    settings is validated physical/wiring data; snapshot supplies exactly the live
    DC source value in its native A/V unit. template arrays use ticks/MHz/degrees with the request's time_base.
    Support uses the shared nearest-2-ns, ties-even conversion. Only the bound
    microwave generators enter tracks under their own source names. Fast-flux
    controls pass through FilteredFlux including delay, signed complex envelopes and all exponential
    tails. Absolute Phi0 waveform samples use a 2 ns grid plus event sides and
    readout endpoint; interpolation is linear. Quantized bath levels use
    midpoint flux at solver.bath_segment_ns intervals plus delayed events,
    rounded ties-even to 1e-3 Phi0 relative to DC; equal adjacent levels merge.
    Ideal constant platforms omit waveform to retain abrupt level changes.
    No forced return to DC or physics calculation occurs. All original points
    occupy one batch, even when their readout times differ. Engine owns segmenting.

    The configured receiver must occur once in the complete observation set.
    Other receivers retain their triggers for shared projective classification.
    One terminal pulse block per point is required.
    Triggered nonterminal blocks become ordered projective reads. Unobserved
    terminal blocks retain pulses and empty grids. Projective starts enter batch
    segment boundaries; runtime alone projects densities and continues evolution.
    readout_kind="pulse" requires one tone; "mux" accepts 1..8 coherent tones
    with common start. A None flux_line permits only DC flux and microwaves.
    Earlier readout pulses are retained separately, not treated as microwaves.
    Unbound generators/receivers, invalid DC, crossing readout blocks or invalid
    numeric records raise ValueError. No ADC solver, density or RNG is used.
    charge_requirements names generators that must bind to charge drive_lines,
    even if already bound to flux or readout. Mismatches raise ValueError naming
    the request and channel before waveform construction or evolution.
    Returned contents are detached; input buffers may subsequently be reused.
    """
    drive_names = {line.gen_ch: line.name for line in settings.drive_lines.values()}
    for requirement in charge_requirements:
        if requirement.gen_ch not in drive_names:
            raise ValueError(
                f"{requirement.name}: generator {requirement.gen_ch} requires a "
                "declared charge drive line"
            )
    windows = tuple(
        window
        for window in template.readouts
        if window.channel == settings.readout.adc.ro_ch
    )
    if len(windows) != 1:
        raise ValueError("single fluxonium requires its configured receiver once")
    dc = dc_flux(settings, snapshot)
    memo: dict[int, EnvelopeSpec] = {}
    drive = ReadoutDrive(
        settings.readout.drive.gen_ch,
        settings.readout.drive.input_per_gain,
        readout_kind,
    )
    partition = classify_readout(template, drive, memo, time_base=time_base)
    read_events = logical_event_contents(
        template, drive, windows[0], memo, time_base=time_base
    )
    contents = read_events[-1]
    microwave, fast_flux = [], []
    for index, event in enumerate(template.events):
        if event.gen_ch == drive.gen_ch:
            continue
        if event.gen_ch in drive_names:
            microwave.append(
                drive_track(event, drive_names[event.gen_ch], memo, time_base=time_base)
            )
        elif (
            settings.flux_line is not None and event.gen_ch == settings.flux_line.gen_ch
        ):
            fast_flux.append(
                drive_track(event, settings.flux_line.name, memo, time_base=time_base)
            )
        else:
            raise ValueError(
                f"template.events.{index}.gen_ch: unbound single-fluxonium "
                f"generator {event.gen_ch}; declare a drive_lines or flux_line binding"
            )
    flux, waveform = _flux_inputs(settings, dc, tuple(fast_flux), contents.end_ns)
    return SingleFluxoniumPlan(
        TimelineBatch(
            template.points,
            tuple(microwave),
            flux,
            contents.end_ns,
            waveform,
            segment_boundaries=tuple(
                tuple(read.start_ns for read in reads)
                for reads in partition.projective_reads
            ),
        ),
        contents,
        partition.cavity_drives,
        read_events,
        partition.projective_reads,
    )
