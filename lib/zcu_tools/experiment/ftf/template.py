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
    SegmentShift,
    TimelineBatch,
    segment_spans,
)
from zcu_tools.simulate.ftf import (
    CavityDrive,
    JointModel,
    model_source,
    readout_model,
    schedule,
)

from . import replay
from .numeric import dc_flux, drive_sources, joint_parameters, resonators
from .readout import FTFReadoutConfig, readout_configuration


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
    batch includes final-span averaged cavity shifts and dephasing, including
    ringdown after drive switch-off. Planning solves conditional cavity fields,
    but never propagates a density or draws random samples.
    These inputs are author diagnostics, not public measurement metadata.
    """

    source: ModelSource
    batch: TimelineBatch
    readouts: tuple[FTFReadoutPlan, ...]
    cavity_drives: tuple[tuple[ReadoutPulse, ...], ...]


def _timeline_batch(
    settings: FTFSettings,
    joint: JointModel,
    template: TimelineTemplate,
    end_ns: NDArray[np.float64],
    memo: dict[int, EnvelopeSpec],
) -> TimelineBatch:
    """Map charge/flux drive tracks onto the engine's fixed-DC schedule."""
    microwave = {s.gen_ch: name for name, s in settings.drive_lines.items()}
    drives = []
    for event in template.events:
        if event.gen_ch == settings.readout.drive.gen_ch:
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
    joint: JointModel,
    settings: FTFSettings,
    pulses: tuple[tuple[ReadoutPulse, ...], ...],
) -> TimelineBatch:
    """Compose cavity averages over evolution-owned final spans, once per point."""
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
    batch = replace(batch, segment_boundaries=boundaries)
    spans = segment_spans(batch)
    pair = resonators(settings)
    shifts = []
    for point_pulses, point_spans in zip(pulses, spans, strict=True):
        if not point_pulses:
            shifts.append(())
            continue
        cavity = CavityDrive(joint, pair, point_pulses)
        corrections = []
        for start, end in point_spans:
            average = cavity.average(start, end)
            corrections.append(
                SegmentShift(
                    start, end, average.stark_shift_ghz, average.dephasing_per_ns
                )
            )
        shifts.append(tuple(corrections))
    return replace(batch, segment_shifts=tuple(shifts))


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
    Mux permits simultaneous coherent tones with independent ADC grids. Program
    common-origin carrier phases convert to pulse-local phases in the first
    carrier's model reference frame; ADC demodulation remains absolute-time.
    Missing routes, invalid DC values or invalid readout pulse blocks raise
    ValueError. Terminal contents deduplicate point rows; earlier cavity pulses
    detach once per active point/slot. No per-point evolution planner is called.
    """
    config = readout_configuration(settings) if readout is None else readout
    receivers = {receiver.ro_ch: receiver for receiver in config.receivers}
    if not template.readouts:
        raise ValueError("FTF requires at least one bound receiver")
    if len({window.channel for window in template.readouts}) != len(template.readouts):
        raise ValueError("duplicate FTF receiver")
    # Identity lookup is safe: template keeps every input envelope alive for
    # this call. No memo survives a plan or retains compiled program buffers.
    memo: dict[int, EnvelopeSpec] = {}
    drive = ReadoutDrive(
        settings.readout.drive.gen_ch,
        settings.readout.drive.input_per_gain,
        config.kind,
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
            _timeline_batch(settings, joint, template, end_ns, memo),
            joint,
            settings,
            cavity_drives,
        ),
        tuple(readouts),
        cavity_drives,
    )
