"""Array QICK-to-engine conversion and distinct DC readout contents."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    EnvironmentSnapshot,
    TimelineTemplate,
)
from zcu_tools.experiment.readout_plan import (
    ProjectiveRead,
    ReadoutDrive,
    ReadoutPartition,
    classify_readout,
    drive_track,
    local_readout_pulses,
    read_event_contents,
    readout_contents,
)
from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.simulate.adc import (
    ReadoutBatch,
    ReadoutEvent,
    ReadoutPulse,
    Receiver,
)
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulate.ftf import (
    JointModel,
    model_source,
    readout_model,
    schedule,
)

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
    batch: ns/GHz/rad charge/flux drives at fixed DC, ending at readout start.
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
    batch includes cavity pulse boundaries, but not cavity shifts. Runtime
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


def _timeline_batch(
    settings: FTFSettings,
    joint: JointModel,
    template: TimelineTemplate,
    end_ns: NDArray[np.float64],
    path: _ReadoutPath,
    memo: dict[int, EnvelopeSpec],
) -> TimelineBatch:
    """Map charge/flux drive tracks onto the engine's fixed-DC schedule."""
    microwave = {s.gen_ch: name for name, s in settings.drive_lines.items()}
    drives = []
    for event in template.events:
        if event.gen_ch == path.gen_ch or not np.any(event.active):
            continue
        if event.gen_ch not in microwave:
            raise ValueError(f"unbound FTF control generator {event.gen_ch}")
        drives.append(drive_track(event, microwave[event.gen_ch], memo))
    fixed_dc = schedule(
        (),
        dc_flux=joint.dc_flux,
        lines=(),
        sample_period_ns=2.0,
        points=len(template.points),
    )
    return TimelineBatch(template.points, tuple(drives), fixed_dc, end_ns)


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
            )
        )
        for point, reads in zip(pulses, projective_reads, strict=True)
    )
    return replace(batch, segment_boundaries=boundaries)


@dataclass(frozen=True)
class _ReadoutEvents:
    """One ADC's detached logical events and point-to-content mapping."""

    events: tuple[tuple[ReadoutEvent, ...], ...]
    inverse: NDArray[np.int64]


def _projective_contents(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    partition: ReadoutPartition,
    terminal_start_ns: NDArray[np.float64],
    memo: dict[int, EnvelopeSpec],
) -> tuple[tuple[_ReadoutEvents, ...], float]:
    """Align actual trigger contents on a common chronological block axis."""
    channels = tuple(
        read_event_contents(template, drive, window, memo)
        for window in template.readouts
    )
    reference = partition.terminal_pulses[0][0].frequency_ghz
    # Projective plans retain point-local contents, including unobserved terminal
    # pulses. Observed-trigger keys alone cannot identify those incident fields.
    inverse = np.arange(len(template.points), dtype=np.int64)
    events_by_channel: list[list[tuple[ReadoutEvent, ...]]] = [[] for _ in channels]
    for point in range(len(template.points)):
        starts = (
            *(read.start_ns for read in partition.projective_reads[point]),
            float(terminal_start_ns[point]),
        )
        blocks = (
            *(read.pulses for read in partition.projective_reads[point]),
            partition.terminal_pulses[point],
        )
        # Classification phases are already pulse-local at zero GHz. Changing
        # the rotating reference adds only its phase at each pulse's start.
        pulses = tuple(
            tuple(
                replace(
                    pulse,
                    phase_rad=pulse.phase_rad + 2 * np.pi * reference * pulse.start_ns,
                )
                for pulse in block
            )
            for block in blocks
        )
        for output, channel in zip(events_by_channel, channels, strict=True):
            output.append(
                tuple(
                    ReadoutEvent(
                        incident,
                        tuple(
                            grid
                            for content in channel
                            if content.end_ns[point] == start
                            for grid in content.grids[content.inverse[point]]
                        ),
                    )
                    for start, incident in zip(starts, pulses, strict=True)
                )
            )
    return (
        tuple(_ReadoutEvents(tuple(events), inverse) for events in events_by_channel),
        reference,
    )


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
    readout blocks with their own triggers are projective reads; untriggered
    blocks are cavity drives. A block owns triggers until the next block starts,
    including ringdown. Earlier pulses may end at the next start, but their last
    ADC sample must be strictly earlier. Invalid windows name the block/trigger.
    Overlapping/later drive-line controls do not affect the DC readout model.
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
    partition = classify_readout(template, drive, memo)
    cavity_drives = partition.cavity_drives
    terminal_template = replace(template, events=partition.terminal_slots)
    for window in template.readouts:
        if window.channel not in receivers:
            raise ValueError(f"unbound FTF receiver {window.channel}")
    end_ns = np.max(
        [
            np.where(e.active, e.start_us * 1000, -np.inf)
            for e in partition.terminal_slots
        ],
        axis=0,
    )
    if any(partition.projective_reads):
        event_contents, reference_frequency_ghz = _projective_contents(
            template, drive, partition, end_ns, memo
        )
    else:
        contents = tuple(
            readout_contents(terminal_template, drive, window, memo)
            for window in template.readouts
        )
        reference_frequency_ghz = contents[0].pulses[0][0].frequency_ghz
        event_contents = tuple(
            _ReadoutEvents(
                tuple(
                    (ReadoutEvent(pulses, grids),)
                    for pulses, grids in zip(
                        local_readout_pulses(content.pulses, reference_frequency_ghz),
                        content.grids,
                        strict=True,
                    )
                ),
                content.inverse,
            )
            for content in contents
        )
    if facts is None:
        facts = FTFModelFacts(settings, snapshot)
    elif not facts.matches(settings, snapshot):
        raise ValueError("FTF physical facts do not match settings/snapshot/revision")
    joint, source, physical = facts.joint, facts.source, facts.readout
    readouts = []
    for window, content in zip(template.readouts, event_contents, strict=True):
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
                    content.events,
                    content.inverse,
                ),
            )
        )
    return FTFTemplatePlan(
        source,
        _cavity_batch(
            _timeline_batch(settings, joint, template, end_ns, path, memo),
            cavity_drives,
            partition.projective_reads,
        ),
        tuple(readouts),
        cavity_drives,
        partition.projective_reads,
    )
