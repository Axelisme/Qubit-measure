"""Array QICK-to-engine conversion and distinct DC readout contents."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig

from zcu_tools.core.simulation import (
    EnvironmentSnapshot,
    TimeBase,
    TimelineTemplate,
)
from zcu_tools.experiment.readout_plan import (
    ProjectiveRead,
    ReadoutDrive,
    ReadoutMapping,
    classify_readout,
    drive_track,
    logical_event_contents,
    map_readout_events,
    readout_contents,
)
from zcu_tools.program.v2.mocksoc import make_mux_mock_soccfg, pfb_channel
from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.simulate.adc import (
    ReadoutBatch,
    ReadoutPulse,
    Receiver,
)
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    FluxWaveform,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulate.ftf import (
    FluxTrack,
    JointModel,
    flux_waveform,
    model_source,
    plateau_boundaries,
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

    channel is the logical ADC index. batch contains this receiver's calibrated
    model, grids and point-to-content mapping. Observed mux events retain only
    incident carriers in the ADC's profile PFB channel; same-channel carriers
    superpose. Observed empty channels emit receiver offset and noise only.
    Unobserved events keep their pulses and empty grids.
    """

    channel: int
    batch: ReadoutBatch


@dataclass(frozen=True)
class FTFTemplatePlan:
    """Detached numeric inputs for all points of one structural template.

    source: round-DC secular model source, including the Gibbs initial density.
    batch: ns/GHz/rad microwave tracks and three-loop fast-flux schedule,
        with continuous samples for filtered/varying flux, ending at terminal start.
    readouts: per-ADC pulse/grid contents at DC, in declaration order. Every ADC
        retains the same ordered logical event axis, with empty grids for events
        it does not observe. An event contains only its own block's pulses.
    cavity_drives: tuple in batch.points order, each containing the earlier active
        untriggered readout-generator ReadoutPulse inputs in template slot order. Supports are
        absolute ns, carriers GHz, phases pulse-local radians in the neutral DC
        readout model's zero-GHz reference. Amplitudes include input_per_gain and
        constant envelope values; nonconstant envelopes retain their local shape.
        Empty point tuples mean no cavity drive. Earlier pulses must end no later
        than the next block start; crossing pulses raise ValueError during planning.
    batch includes delayed flux events, cavity pulse boundaries and projection
    starts, but not cavity shifts. Runtime
    queries final spans and advances densities and the dominant cavity branch
    sequentially. Planning never solves cavity fields, propagates a density or
    draws random samples.
    projective_reads: per-point nonterminal read blocks in chronological order.
        Starts are projection times in ns; pulses use zero-GHz pulse-local phases
        and triggers retain channel-local indices. These starts are batch segment
        boundaries. Runtime owns projection and continuation; planning does neither.
    These inputs are author diagnostics, not public measurement metadata.
    """

    source: ModelSource
    batch: TimelineBatch
    readouts: tuple[FTFReadoutPlan, ...]
    cavity_drives: tuple[tuple[ReadoutPulse, ...], ...]
    projective_reads: tuple[tuple[ProjectiveRead, ...], ...]


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
    waveform: FluxWaveform | None,
) -> tuple[tuple[float, ...], ...]:
    """Keep nominal events and actual sampled plateau edges inside each window."""
    if not tracks:
        return ()
    events: list[set[float]] = [set() for _ in ends]
    if waveform is not None:
        for point, nodes in enumerate(plateau_boundaries(waveform)):
            events[point].update(t for t in nodes if 0 <= t <= ends[point])
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
    *,
    time_base: TimeBase,
) -> TimelineBatch:
    """Separate microwave and fast-flux slots using the round's DC projection."""
    memo: dict[int, EnvelopeSpec] = {}
    microwave = {s.gen_ch: name for name, s in settings.drive_lines.items()}
    lines = _flux_lines(settings)
    drives, flux_tracks = [], []
    for event in template.events:
        if event.gen_ch == path.gen_ch or not np.any(event.active):
            continue
        if event.gen_ch in lines:
            track = drive_track(
                event, lines[event.gen_ch].source, memo, time_base=time_base
            )
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
            drives.append(
                drive_track(event, microwave[event.gen_ch], memo, time_base=time_base)
            )
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
    boundaries = _flux_boundaries(tuple(flux_tracks), by_source, end_ns, waveform)
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
    projective_reads: tuple[tuple[ProjectiveRead, ...], ...],
) -> TimelineBatch:
    """Add cavity and projection boundaries without evolving densities or fields."""
    if not any(pulses) and not any(projective_reads):
        return batch
    boundaries = tuple(
        tuple(
            sorted(
                {float(t) for pulse in point for t in (pulse.start_ns, pulse.end_ns)}
                | {read.start_ns for read in reads}
                | set(
                    batch.segment_boundaries[index] if batch.segment_boundaries else ()
                )
            )
        )
        for index, (point, reads) in enumerate(
            zip(pulses, projective_reads, strict=True)
        )
    )
    return replace(batch, segment_boundaries=boundaries)


def _projective_contents(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    memo: dict[int, EnvelopeSpec],
    time_base: TimeBase,
) -> tuple[tuple[ReadoutMapping, ...], float]:
    """Align actual trigger contents on a common chronological block axis."""
    channels = tuple(
        logical_event_contents(template, drive, window, memo, time_base=time_base)
        for window in template.readouts
    )
    reference = channels[0][-1].pulses[0][0].frequency_ghz
    return tuple(
        map_readout_events(contents, reference) for contents in channels
    ), reference


def _mux_channel_contents(
    content: ReadoutMapping, path: _ReadoutPath, channel: int
) -> ReadoutMapping:
    """Select incident carriers using the fixed compile profile's PFB channels."""
    channels = tuple(receiver.ro_ch for receiver in path.receivers)
    profile = make_mux_mock_soccfg(
        path.gen_ch + 1, max(channels) + 1, gen_ch=path.gen_ch, ro_chs=channels
    )
    selected = []
    for events in content.events:
        row = []
        for event in events:
            addressed = {
                pfb_channel(profile, channel, grid.demodulation_frequency_ghz * 1000)
                for grid in event.grids
            }
            if len(addressed) > 1:
                raise ValueError(
                    f"FTF mux ADC {channel} cannot select different PFB channels "
                    "within one readout event"
                )
            pulses = tuple(
                pulse
                for pulse in event.pulses
                if not addressed
                or pfb_channel(profile, channel, pulse.frequency_ghz * 1000)
                in addressed
            )
            row.append(replace(event, pulses=pulses))
        selected.append(tuple(row))
    return replace(content, events=tuple(selected))


def plan_template(
    settings: FTFSettings,
    snapshot: EnvironmentSnapshot,
    template: TimelineTemplate,
    *,
    time_base: TimeBase,
    soccfg: QickConfig,
    readout: FTFReadoutConfig | None = None,
    facts: FTFModelFacts | None = None,
) -> FTFTemplatePlan:
    """Convert one C1 template to C3 evolution and C6 readout inputs.

    settings is the immutable FTF resource, snapshot supplies live fixed-unit DC
    values, and template has point arrays in ticks/MHz/degrees, interpreted with time_base. Support rounds to
    nearest 2 ns, ties even, without changing the common carrier origin. The
    latest active readout-generator block per point ends evolution. Earlier
    readout blocks with their own triggers are projective reads; untriggered
    blocks are cavity drives. A block owns triggers until the next block starts,
    including ringdown. Earlier pulses may end at the next start, but their last
    ADC sample must be strictly earlier. Invalid windows name the block/trigger.
    Bound fast-flux generators pass through per-loop delay and multi-exponential
    filtering, with 2 ns waveform samples and 1e-3 Phi0 midpoint bath levels.
    Ideal constant platforms omit continuous samples to retain sudden switching.
    Filtered or varying envelopes provide continuous samples for engine routing.
    Flux endpoints and envelope knots join cavity boundaries and projection starts.
    Snapshot DC still defines the fixed projection, Gibbs state and readout model.
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
    Mux retains only carriers in each ADC's fixed-profile PFB channel, including
    same-channel superposition. Each logical event selects independently. Grids
    in one event spanning different PFB channels raise ValueError. An addressed
    single carrier is unchanged; off-channel carriers are excluded even alone.
    An observed mux event without an in-channel carrier retains empty pulses
    and emits receiver background offset and noise only. Unobserved events retain
    their pulses and empty grids without emitting data.
    Program common-origin carrier phases convert to pulse-local phases in the first
    terminal carrier's model reference frame; ADC demodulation remains absolute-time.
    All ADCs retain the same ordered event axis, even if none observes terminal.
    Unobserved events have empty grids. Each ADC must have at least one trigger.
    Projective block counts and each ADC's per-event grid counts must match across
    points; differences raise ValueError naming the point and differing counts.
    Missing routes, invalid DC values or invalid readout pulse blocks raise
    ValueError. Terminal-only contents deduplicate point rows; projective plans
    retain point-local contents. Earlier cavity pulses detach once per active
    point/slot. No per-point evolution planner is called.
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
    partition = classify_readout(template, drive, memo, time_base=time_base)
    cavity_drives = partition.cavity_drives
    terminal_template = replace(template, events=partition.terminal_slots)
    for window in template.readouts:
        if window.channel not in receivers:
            raise ValueError(f"unbound FTF receiver {window.channel}")
    end_ns = np.max(
        [
            np.where(e.active, time_base.ns(e.start_ticks), -np.inf)
            for e in partition.terminal_slots
        ],
        axis=0,
    )
    if any(partition.projective_reads):
        event_contents, reference_frequency_ghz = _projective_contents(
            template, drive, memo, time_base
        )
    else:
        contents = tuple(
            readout_contents(
                terminal_template, drive, window, memo, time_base=time_base
            )
            for window in template.readouts
        )
        reference_frequency_ghz = contents[0].pulses[0][0].frequency_ghz
        event_contents = tuple(
            map_readout_events((content,), reference_frequency_ghz)
            for content in contents
        )
    if facts is None:
        facts = FTFModelFacts(settings, snapshot)
    elif not facts.matches(settings, snapshot):
        raise ValueError("FTF physical facts do not match settings/snapshot/revision")
    joint, source, physical = facts.joint, facts.source, facts.readout
    readouts = []
    for window, content in zip(template.readouts, event_contents, strict=True):
        if path.multiplexed:
            content = _mux_channel_contents(content, path, window.channel)
        adc = receivers[window.channel]
        dt = time_base.ns(window.sample_period_ticks)
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
                    content.events,
                    content.inverse,
                    output_order=content.output_order,
                ),
            )
        )
    return FTFTemplatePlan(
        source,
        _cavity_batch(
            _timeline_batch(
                settings, joint, template, end_ns, path, time_base=time_base
            ),
            cavity_drives,
            partition.projective_reads,
        ),
        tuple(readouts),
        cavity_drives,
        partition.projective_reads,
    )
