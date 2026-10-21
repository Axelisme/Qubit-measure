"""Array QICK-to-engine conversion and distinct DC readout contents."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    EnvironmentSnapshot,
    EventSlot,
    ReadoutSlot,
    TimelineTemplate,
)
from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.simulate.adc import (
    PiecewiseLinearEnvelope,
    ReadoutBatch,
    ReadoutPulse,
    Receiver,
    SampleGrid,
)
from zcu_tools.simulate.evolution import (
    DriveTrack,
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
    readouts: per-ADC distinct terminal pulse/grid contents at DC, in declaration order.
    cavity_drives: tuple in batch.points order, each containing the earlier active
        readout-generator ReadoutPulse inputs in template slot order. Supports are
        absolute ns, carriers GHz, phases pulse-local radians in the neutral DC
        readout model's zero-GHz reference. Amplitudes include input_per_gain and
        constant envelope values; nonconstant envelopes retain their local shape.
        Empty point tuples mean no cavity drive. Earlier pulses must end no later
        than terminal start; crossing pulses raise ValueError during planning.
    batch includes cavity pulse boundaries, but not cavity shifts. Runtime
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


def _envelopes(
    event: EventSlot, memo: dict[int, EnvelopeSpec]
) -> tuple[EnvelopeSpec, ...]:
    """Reuse immutable specs by identity while the template retains all inputs."""
    specs = []
    for envelope in event.envelopes:
        key = id(envelope)
        if key not in memo:
            memo[key] = (
                EnvelopeSpec("constant")
                if envelope is None
                else EnvelopeSpec(
                    "sampled",
                    samples=np.array(envelope.values, dtype=np.complex128),
                    sample_ns=np.array(envelope.sample_us) * 1000,
                )
            )
        specs.append(memo[key])
    return tuple(specs)


def _track(event: EventSlot, port: str, memo: dict[int, EnvelopeSpec]) -> DriveTrack:
    start = event.start_us * 1000
    # Match the old numeric owner's absolute nearest-2-ns support, ties even.
    quantized = 2 * np.rint(start / 2)
    end = 2 * np.rint((start + event.duration_us * 1000) / 2)
    return DriveTrack(
        port,
        quantized,
        end - quantized,
        event.frequency_mhz / 1000,
        np.deg2rad(event.phase_deg)
        - 2 * np.pi * event.frequency_mhz * event.phase_reference_us,
        event.gain,
        _envelopes(event, memo),
        event.envelope_index,
        event.active,
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
        drives.append(_track(event, microwave[event.gen_ch], memo))
    fixed_dc = schedule(
        (),
        dc_flux=joint.dc_flux,
        lines=(),
        sample_period_ns=2.0,
        points=len(template.points),
    )
    return TimelineBatch(template.points, tuple(drives), fixed_dc, end_ns)


def _classify_readout(
    path: _ReadoutPath,
    template: TimelineTemplate,
    memo: dict[int, EnvelopeSpec],
) -> tuple[tuple[EventSlot, ...], tuple[tuple[ReadoutPulse, ...], ...]]:
    """Partition active readout-generator slots at each point's latest start."""
    slots = tuple(event for event in template.events if event.gen_ch == path.gen_ch)
    if not slots:
        raise ValueError("FTF requires one terminal readout pulse per point")
    latest = np.max([np.where(e.active, e.start_us, -np.inf) for e in slots], axis=0)
    if not np.isfinite(latest).all():
        raise ValueError("FTF requires one terminal readout pulse per point")
    terminals = []
    drives: list[list[ReadoutPulse]] = [[] for _ in template.points]
    for event in slots:
        terminal = event.active & (event.start_us == latest)
        earlier = event.active & ~terminal
        if np.any(earlier & (event.start_us + event.duration_us > latest)):
            if path.multiplexed:
                raise ValueError(
                    "FTF mux tones require a common pulse start; blocks overlap"
                )
            raise ValueError("FTF cavity drive crosses terminal readout start")
        terminals.append(replace(event, active=terminal))
        envelopes = list(_envelopes(event, memo))
        rows = np.column_stack(
            (
                event.start_us,
                event.duration_us,
                event.frequency_mhz,
                event.phase_deg - 360 * event.frequency_mhz * event.phase_reference_us,
                event.gain,
                event.envelope_index,
            )
        )
        for index in np.flatnonzero(earlier):
            drives[index].append(
                _readout_pulse(path.input_per_gain, rows[index], envelopes)
            )
    return tuple(terminals), _local_readout_pulses(tuple(tuple(p) for p in drives), 0.0)


def _readout_contents(
    path: _ReadoutPath,
    template: TimelineTemplate,
    window: ReadoutSlot,
    memo: dict[int, EnvelopeSpec],
) -> tuple[
    NDArray[np.float64],
    tuple[tuple[ReadoutPulse, ...], ...],
    tuple[tuple[SampleGrid, ...], ...],
    NDArray[np.int64],
]:
    """Merge mutually exclusive readout slots, then deduplicate their point rows."""
    slots = tuple(e for e in template.events if e.gen_ch == path.gen_ch)
    count = np.sum([e.active for e in slots], axis=0)
    if not slots or np.any(count == 0):
        raise ValueError("FTF requires one terminal readout pulse per point")
    if not path.multiplexed and np.any(count != 1):
        raise ValueError("FTF requires one terminal readout pulse per point")
    tones = int(count[0])
    if tones > 8 or np.any(count != tones):
        raise ValueError("FTF mux requires a fixed 1..8 tone count per point")
    if not window.trigger_us.shape[1] or np.any(
        window.generator_channels != path.gen_ch
    ):
        raise ValueError("FTF requires ADC timing and its bound readout generator")
    fields = ("start_us", "duration_us", "frequency_mhz", "phase_deg", "gain")
    values = np.zeros((len(template.points), tones, 6))
    rows = np.arange(len(template.points))
    ranks = np.cumsum([e.active for e in slots], axis=0) - 1
    envelopes = []
    for event, rank in zip(slots, ranks, strict=True):
        for col, name in enumerate(fields):
            values[rows, rank, col] += np.where(event.active, getattr(event, name), 0)
        # Retain the program's common-origin phase until the model reference
        # frequency is known; ADC pulse-local conversion happens below.
        values[rows, rank, 3] -= np.where(
            event.active, 360 * event.frequency_mhz * event.phase_reference_us, 0
        )
        values[rows, rank, 5] += np.where(
            event.active, event.envelope_index + len(envelopes), 0
        )
        envelopes.extend(_envelopes(event, memo))
    if np.any(values[:, :, 0] != values[:, :1, 0]):
        raise ValueError("FTF mux tones require a common pulse start per point")
    if np.any(window.trigger_us[:, 0] < values[:, 0, 0]):
        raise ValueError(
            "FTF first ADC trigger must not precede terminal readout start"
        )
    features = np.column_stack(
        (
            values.reshape(len(template.points), -1),
            window.trigger_us,
            window.frequency_mhz,
            window.phase_deg,
        )
    )
    _, representatives, inverse = np.unique(
        features, axis=0, return_index=True, return_inverse=True
    )
    pulses = []
    grids = []
    dt = 1000 / window.sample_rate_mhz
    for index in representatives:
        pulses.append(
            tuple(
                _readout_pulse(path.input_per_gain, row, envelopes)
                for row in values[index]
            )
        )
        grids.append(
            tuple(
                SampleGrid(
                    trigger * 1000 + dt * np.arange(window.samples),
                    frequency / 1000,
                    float(np.deg2rad(phase)),
                )
                for trigger, frequency, phase in zip(
                    window.trigger_us[index],
                    window.frequency_mhz[index],
                    window.phase_deg[index],
                    strict=True,
                )
            )
        )
    return values[:, 0, 0] * 1000, tuple(pulses), tuple(grids), inverse.astype(np.int64)


def _readout_pulse(
    input_per_gain: float, row: NDArray[np.float64], envelopes: list[EnvelopeSpec]
) -> ReadoutPulse:
    start, duration, freq, phase, gain, envelope_index = row
    envelope = envelopes[int(envelope_index)]
    amplitude = complex(input_per_gain * gain)
    callback = None
    corners = ()
    if envelope.kind == "sampled":
        if np.all(envelope.samples == envelope.samples[0]):
            amplitude *= envelope.samples[0]
        else:
            callback = PiecewiseLinearEnvelope(envelope.sample_ns, envelope.samples)
            corners = tuple(
                start * 1000 + t for t in envelope.sample_ns if 0 < t < duration * 1000
            )
    return ReadoutPulse(
        start * 1000,
        (start + duration) * 1000,
        amplitude,
        freq / 1000,
        float(np.deg2rad(phase)),
        callback,
        corners,
    )


def _local_readout_pulses(
    contents: tuple[tuple[ReadoutPulse, ...], ...], reference_frequency_ghz: float
) -> tuple[tuple[ReadoutPulse, ...], ...]:
    """Convert common-origin phases to ADC's negative pulse-local carrier.

    Demodulation keeps absolute time, so the incident field needs the phase
    accumulated from zero to pulse start in the model's rotating frame.
    Leave the reference carrier untouched to retain single-tone output bits.
    """
    local_contents = []
    for pulses in contents:
        local_pulses = []
        for pulse in pulses:
            detuning_ghz = pulse.frequency_ghz - reference_frequency_ghz
            if detuning_ghz != 0:
                pulse = replace(
                    pulse,
                    phase_rad=pulse.phase_rad
                    - 2 * np.pi * detuning_ghz * pulse.start_ns,
                )
            local_pulses.append(pulse)
        local_contents.append(tuple(local_pulses))
    return tuple(local_contents)


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
            )
        )
        for point in pulses
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
    terminal_slots, cavity_drives = _classify_readout(path, template, memo)
    terminal_template = replace(template, events=terminal_slots)
    contents = []
    for window in template.readouts:
        if window.channel not in receivers:
            raise ValueError(f"unbound FTF receiver {window.channel}")
        contents.append(_readout_contents(path, terminal_template, window, memo))
    end_ns = contents[0][0]
    if facts is None:
        facts = FTFModelFacts(settings, snapshot)
    elif not facts.matches(settings, snapshot):
        raise ValueError("FTF physical facts do not match settings/snapshot/revision")
    joint, source, physical = facts.joint, facts.source, facts.readout
    reference_frequency_ghz = contents[0][1][0][0].frequency_ghz
    readouts = []
    for window, (_, pulses, grids, inverse) in zip(
        template.readouts, contents, strict=True
    ):
        adc = receivers[window.channel]
        dt = 1000 / window.sample_rate_mhz
        receiver = Receiver(
            adc.gain,
            float(np.deg2rad(adc.phase_deg)),
            np.array(adc.offset_adc),
            np.array(adc.noise_psd_adc2_ns) / dt,
            time_of_flight_ns=adc.time_of_flight_ns,
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
                    _local_readout_pulses(pulses, reference_frequency_ghz),
                    grids,
                    inverse,
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
