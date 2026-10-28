"""Array QICK-to-engine conversion and distinct DC readout contents."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    EnvironmentSnapshot,
    TimelineTemplate,
)
from zcu_tools.experiment.readout_plan import (
    ReadoutDrive,
    classify_readout,
    drive_track,
    local_readout_pulses,
    readout_contents,
)
from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.simulate.adc import (
    ReadoutBatch,
    ReadoutPulse,
    Receiver,
)
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulate.ftf import (
    FluxTrack,
    JointModel,
    flux_waveform,
    model_source,
    readout_model,
    schedule,
)
from zcu_tools.simulate.lines import DistortionTerm, FluxLine

from . import replay
from .numeric import dc_flux, drive_sources, joint_parameters, resonators
from .readout import FTFReadoutConfig, FTFReceiverConfig, readout_configuration


class FTFModelFacts:
    """Physical inputs for one complete settings/snapshot/numeric revision.

    settings is a validated FTF resource and snapshot supplies live DC values.
    joint owns the fixed-DC projection and its bounded eigen/bath memo. source
    owns Gibbs density, drive quadratures, flux expansion and secular models.
    readout is the neutral DC ADC model, without pulse or receiver calibration.
    These objects must remain unchanged for their lifetime. Invalid DC or bath
    inputs propagate their numeric errors. No density evolution or sampling runs.
    Used by FTFSource across acquires and by standalone plan_template calls.
    """

    def __init__(self, settings: FTFSettings, snapshot: EnvironmentSnapshot) -> None:
        self._settings_json = settings.model_dump_json()
        self._snapshot = snapshot
        self._revision = (replay.BACKEND_VERSION, replay.SEGMENT_NUMERIC_VERSION)
        self.joint = JointModel(
            joint_parameters(settings),
            dc_flux(settings, snapshot),
            temperature_k=settings.chip.temperature_k,
        )
        resonator_pair = resonators(settings)
        self.source = model_source(
            self.joint, drive_sources(settings, self.joint), resonators=resonator_pair
        )
        self.readout = readout_model(self.joint, resonator_pair)

    def matches(self, settings: FTFSettings, snapshot: EnvironmentSnapshot) -> bool:
        """Whether the complete settings, snapshot and current revisions are equal."""
        return (
            self._snapshot == snapshot
            and self._revision
            == (replay.BACKEND_VERSION, replay.SEGMENT_NUMERIC_VERSION)
            and self._settings_json == settings.model_dump_json()
        )


@dataclass(frozen=True)
class FTFReadoutPlan:
    """One ADC's detached batch in program declaration order.

    channel is the logical ADC index. batch contains all incident carriers and
    this receiver's calibrated model, grids and point-to-content mapping.
    """

    channel: int
    batch: ReadoutBatch


@dataclass(frozen=True)
class FTFTemplatePlan:
    """Detached numeric inputs for all points of one structural template.

    source: round-DC secular model source, including the Gibbs initial density.
    batch: ns/GHz/rad microwave tracks and three-loop fast-flux schedule,
        with continuous samples for filtered/varying flux, ending at readout start.
    readouts: per-ADC distinct terminal pulse/grid contents at DC, in declaration order.
    cavity_drives: tuple in batch.points order, each containing the earlier active
        readout-generator ReadoutPulse inputs in template slot order. Supports are
        absolute ns, carriers GHz, phases pulse-local radians in the neutral DC
        readout model's zero-GHz reference. Amplitudes include input_per_gain and
        constant envelope values; nonconstant envelopes retain their local shape.
        Empty point tuples mean no cavity drive. Earlier pulses must end no later
        than terminal start; crossing pulses raise ValueError during planning.
    batch includes delayed flux events and cavity pulse boundaries, but not cavity shifts. Runtime
    queries final spans and advances densities and the dominant cavity branch
    sequentially. Planning never solves cavity fields, propagates a density or
    draws random samples.
    These inputs are author diagnostics, not public measurement metadata.
    """

    source: ModelSource
    batch: TimelineBatch
    readouts: tuple[FTFReadoutPlan, ...]
    cavity_drives: tuple[tuple[ReadoutPulse, ...], ...]


@dataclass(frozen=True)
class _ReadoutPath:
    """Selected source, calibration and ADC capabilities for one template."""

    gen_ch: int
    input_per_gain: float
    receivers: tuple[FTFReceiverConfig, ...]
    multiplexed: bool


def _readout_path(
    settings: FTFSettings, config: FTFReadoutConfig, template: TimelineTemplate
) -> _ReadoutPath:
    """Select capabilities from active event generators, rejecting mixed paths."""
    pulse_gen = settings.readout.drive.gen_ch
    generators = {event.gen_ch for event in template.events if np.any(event.active)}
    config.validate_generators(pulse_gen, generators)
    mux = config.mux
    if mux is not None and mux.gen_ch in generators:
        return _ReadoutPath(mux.gen_ch, mux.input_per_gain, mux.receivers, True)
    return _ReadoutPath(
        settings.readout.drive.gen_ch,
        settings.readout.drive.input_per_gain,
        (config.pulse,),
        False,
    )


def _flux_lines(settings: FTFSettings) -> dict[int, FluxLine]:
    """Adapt one-loop resource calibrations into F1/C/F2 numeric source lines."""
    lines = {}
    for axis, (loop, line) in enumerate(
        (
            ("f1", settings.flux_lines.f1),
            ("coupler", settings.flux_lines.coupler),
            ("f2", settings.flux_lines.f2),
        )
    ):
        if line is not None:
            # Loop keys identify numeric sources; display names need not be unique.
            lines[line.gen_ch] = FluxLine(
                loop,
                tuple(line.flux_phi0_per_gain if i == axis else 0.0 for i in range(3)),
                line.delay_ns,
                line.tau_ns,
                tuple(
                    DistortionTerm(t.amplitude, t.tau_ns) for t in line.distortion_terms
                ),
            )
    return lines


def _flux_boundaries(
    tracks: tuple[FluxTrack, ...],
    lines: dict[str, FluxLine],
    ends: NDArray[np.float64],
) -> tuple[tuple[float, ...], ...]:
    """Keep delayed support and envelope events inside each evolution window."""
    if not tracks:
        return ()
    events: list[set[float]] = [set() for _ in ends]
    for track in tracks:
        for point in np.flatnonzero(track.active):
            start, duration = track.start_ns[point], track.duration_ns[point]
            knots = track.envelopes[track.envelope_index[point]].sample_ns
            interior = knots[(knots > 0) & (knots < duration)]
            times = np.r_[start, start + duration, 2 * np.rint((start + interior) / 2)]
            times += lines[track.source].delay_ns
            events[point].update(float(t) for t in times if 0 <= t <= ends[point])
    return tuple(tuple(sorted(point)) for point in events)


def _timeline_batch(
    settings: FTFSettings,
    joint: JointModel,
    template: TimelineTemplate,
    end_ns: NDArray[np.float64],
    path: _ReadoutPath,
    memo: dict[int, EnvelopeSpec],
) -> TimelineBatch:
    """Separate microwave and fast-flux slots using the round's DC projection."""
    microwave = {s.gen_ch: name for name, s in settings.drive_lines.items()}
    lines = _flux_lines(settings)
    drives, flux_tracks = [], []
    for event in template.events:
        if event.gen_ch == path.gen_ch or not np.any(event.active):
            continue
        if event.gen_ch in lines:
            track = drive_track(event, lines[event.gen_ch].source, memo)
            flux_tracks.append(
                FluxTrack(
                    track.port,
                    track.start_ns,
                    track.duration_ns,
                    track.frequency_ghz,
                    track.phase_rad,
                    track.amplitude,
                    track.envelopes,
                    track.envelope_index,
                    track.active,
                )
            )
        elif event.gen_ch in microwave:
            drives.append(drive_track(event, microwave[event.gen_ch], memo))
        else:
            raise ValueError(f"unbound FTF control generator {event.gen_ch}")
    flux = schedule(
        tuple(flux_tracks),
        dc_flux=joint.dc_flux,
        lines=tuple(lines.values()),
        sample_period_ns=2.0,
        points=len(template.points),
    )
    by_source = {line.source: line for line in lines.values()}
    continuous = any(
        by_source[track.source].tau_ns
        or by_source[track.source].distortion_terms
        or np.any(track.frequency_ghz[track.active] != 0)
        or any(
            env.kind == "sampled" and np.any(env.samples != env.samples[0])
            for env in track.envelopes
        )
        for track in flux_tracks
    )
    waveform = (
        flux_waveform(
            tuple(flux_tracks),
            dc_flux=joint.dc_flux,
            lines=tuple(lines.values()),
            sample_period_ns=2.0,
            points=len(template.points),
        )
        if continuous
        else None
    )
    boundaries = _flux_boundaries(tuple(flux_tracks), by_source, end_ns)
    return TimelineBatch(
        template.points,
        tuple(drives),
        flux,
        end_ns,
        flux_waveform=waveform,
        segment_boundaries=boundaries,
    )


def _cavity_batch(
    batch: TimelineBatch,
    pulses: tuple[tuple[ReadoutPulse, ...], ...],
) -> TimelineBatch:
    """Add cavity support boundaries without evolving densities or fields."""
    if not any(pulses):
        return batch
    boundaries = tuple(
        tuple(
            sorted(
                {float(t) for pulse in point for t in (pulse.start_ns, pulse.end_ns)}
                | set(
                    batch.segment_boundaries[index] if batch.segment_boundaries else ()
                )
            )
        )
        for index, point in enumerate(pulses)
    )
    return replace(batch, segment_boundaries=boundaries)


def plan_template(
    settings: FTFSettings,
    snapshot: EnvironmentSnapshot,
    template: TimelineTemplate,
    *,
    readout: FTFReadoutConfig | None = None,
    facts: FTFModelFacts | None = None,
) -> FTFTemplatePlan:
    """Convert one C1 template to C3 evolution and C6 readout inputs.

    settings is the immutable FTF resource, snapshot supplies live fixed-unit DC
    values, and template has point arrays in us/MHz/degrees. Support rounds to
    nearest 2 ns, ties even, without changing the common carrier origin. The
    latest active readout-generator block per point ends evolution. Earlier
    readout pulses are cavity drives and must end no later than that start.
    The first ADC trigger must not precede it; triggers after pulse end are valid.
    Bound fast-flux generators pass through per-loop delay and multi-exponential
    filtering, with 2 ns waveform samples and 1e-3 Phi0 midpoint bath levels.
    Ideal constant platforms omit continuous samples to retain sudden switching.
    Filtered or varying envelopes provide continuous samples for engine routing.
    Flux endpoints and envelope knots join cavity boundaries. Snapshot DC still
    defines the fixed projection, Gibbs state and terminal readout model.
    Overlapping/later controls and unsettled flux do not change that DC ADC model.
    Constant complex readout envelopes fold into amplitude, enabling the exact field solution.
    readout supplies detached author capabilities; None adapts current settings.
    facts optionally shares physical inputs across templates/acquires. It must
    match the complete settings, snapshot and numeric revision or raises ValueError.
    None builds physical inputs for this call. Receiver/carrier/content data are
    always constructed from the current template and readout configuration.
    Event generators select pulse or mux capabilities. Using both configured
    readout generators in one template raises ValueError naming both channels.
    Readout windows must belong to that path's receivers and generator.
    Mux permits simultaneous coherent tones with independent ADC grids. Program
    common-origin carrier phases convert to pulse-local phases in the first
    carrier's model reference frame; ADC demodulation remains absolute-time.
    Missing routes, invalid DC values or invalid readout pulse blocks raise
    ValueError. Terminal contents deduplicate point rows; earlier cavity pulses
    detach once per active point/slot. No per-point evolution planner is called.
    """
    config = readout_configuration(settings) if readout is None else readout
    path = _readout_path(settings, config, template)
    receivers = {receiver.ro_ch: receiver for receiver in path.receivers}
    if not template.readouts:
        raise ValueError("FTF requires at least one bound receiver")
    if len({window.channel for window in template.readouts}) != len(template.readouts):
        raise ValueError("duplicate FTF receiver")
    # Identity lookup is safe: template keeps every input envelope alive for
    # this call. No memo survives a plan or retains compiled program buffers.
    memo: dict[int, EnvelopeSpec] = {}
    drive = ReadoutDrive(
        path.gen_ch,
        path.input_per_gain,
        "mux" if path.multiplexed else "pulse",
    )
    partition = classify_readout(template, drive, memo)
    cavity_drives = partition.cavity_drives
    terminal_template = replace(template, events=partition.terminal_slots)
    contents = []
    for window in template.readouts:
        if window.channel not in receivers:
            raise ValueError(f"unbound FTF receiver {window.channel}")
        contents.append(readout_contents(terminal_template, drive, window, memo))
    end_ns = contents[0].end_ns
    if facts is None:
        facts = FTFModelFacts(settings, snapshot)
    elif not facts.matches(settings, snapshot):
        raise ValueError("FTF physical facts do not match settings/snapshot/revision")
    joint, source, physical = facts.joint, facts.source, facts.readout
    reference_frequency_ghz = contents[0].pulses[0][0].frequency_ghz
    readouts = []
    for window, content in zip(template.readouts, contents, strict=True):
        adc = receivers[window.channel]
        dt = 1000 / window.sample_rate_mhz
        receiver = Receiver(
            adc.gain,
            float(np.deg2rad(adc.phase_deg)),
            np.array(adc.offset_adc),
            np.array(adc.noise_psd_adc2_ns) / dt,
            time_of_flight_ns=adc.time_of_flight_ns,
            ripple_depth=adc.ripple_depth,
            ripple_period_ghz=adc.ripple_period_mhz / 1000,
            ripple_phase_rad=float(np.deg2rad(adc.ripple_phase_deg)),
            slope_db_per_ghz=adc.slope_db_per_ghz,
            pivot_ghz=adc.pivot_ghz,
            extra_ripples=adc.extra_ripples,
            roughness=adc.roughness,
            band_edges=adc.band_edges,
        )
        model = replace(
            physical,
            receiver=receiver,
            reference_frequency_ghz=reference_frequency_ghz,
            mismatch_phase_rad=np.array(
                [
                    r.mismatch_phase_rad
                    for r in (
                        settings.readout.resonators.r1,
                        settings.readout.resonators.r2,
                    )
                ]
            ),
        )
        readouts.append(
            FTFReadoutPlan(
                window.channel,
                ReadoutBatch(
                    model,
                    local_readout_pulses(content.pulses, reference_frequency_ghz),
                    content.grids,
                    content.inverse,
                ),
            )
        )
    return FTFTemplatePlan(
        source,
        _cavity_batch(
            _timeline_batch(settings, joint, template, end_ns, path, memo),
            cavity_drives,
        ),
        tuple(readouts),
        cavity_drives,
    )
