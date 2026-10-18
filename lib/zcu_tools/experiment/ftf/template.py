"""Array QICK-to-engine conversion and distinct DC readout contents."""

from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import EnvironmentSnapshot, EventSlot, TimelineTemplate
from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.simulate.adc import ReadoutBatch, ReadoutPulse, Receiver, SampleGrid
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

from .numeric import dc_flux, drive_sources, joint_parameters, resonators


@dataclass(frozen=True)
class FTFTemplatePlan:
    """Detached numeric inputs for all points of one structural template.

    source: round-DC secular model source, including the Gibbs initial density.
    batch: ns/GHz/rad charge drives at fixed DC flux, ending at readout start.
    readout: distinct pulse/grid contents at DC, indexed in batch point order.
    No propagation, conditional field solve or random draws happen in planning.
    These inputs are author diagnostics, not public measurement metadata.
    """

    source: ModelSource
    batch: TimelineBatch
    readout: ReadoutBatch


def _envelopes(event: EventSlot) -> tuple[EnvelopeSpec, ...]:
    return tuple(
        EnvelopeSpec("constant")
        if envelope is None
        else EnvelopeSpec(
            "sampled",
            samples=np.array(envelope.values, dtype=np.complex128),
            sample_ns=np.array(envelope.sample_us) * 1000,
        )
        for envelope in event.envelopes
    )


def _track(event: EventSlot, port: str) -> DriveTrack:
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
        _envelopes(event),
        event.envelope_index,
        event.active,
    )


def _timeline_batch(
    settings: FTFSettings,
    joint: JointModel,
    template: TimelineTemplate,
    end_ns: NDArray[np.float64],
) -> TimelineBatch:
    """Map charge controls onto the engine's fixed-DC schedule."""
    microwave = {s.gen_ch: name for name, s in settings.drive_lines.items()}
    drives = []
    for event in template.events:
        if event.gen_ch == settings.readout.drive.gen_ch:
            continue
        if event.gen_ch not in microwave:
            raise ValueError(f"unbound FTF control generator {event.gen_ch}")
        drives.append(_track(event, microwave[event.gen_ch]))
    fixed_dc = schedule(
        (),
        dc_flux=joint.dc_flux,
        lines=(),
        sample_period_ns=2.0,
        points=len(template.points),
    )
    return TimelineBatch(template.points, tuple(drives), fixed_dc, end_ns)


def _readout_contents(
    settings: FTFSettings,
    template: TimelineTemplate,
) -> tuple[
    NDArray[np.float64],
    tuple[tuple[ReadoutPulse, ...], ...],
    tuple[tuple[SampleGrid, ...], ...],
    NDArray[np.int64],
]:
    """Merge mutually exclusive readout slots, then deduplicate their point rows."""
    slots = tuple(
        e for e in template.events if e.gen_ch == settings.readout.drive.gen_ch
    )
    count = np.sum([e.active for e in slots], axis=0)
    if not slots or np.any(count != 1):
        raise ValueError("FTF requires one terminal readout pulse per point")
    if (
        len(template.readouts) != 1
        or template.readouts[0].channel != settings.readout.adc.ro_ch
    ):
        raise ValueError("FTF supports its single bound receiver only")
    window = template.readouts[0]
    if not window.trigger_us.shape[1] or np.any(
        window.generator_channels != settings.readout.drive.gen_ch
    ):
        raise ValueError("FTF requires ADC timing and its bound readout generator")
    fields = ("start_us", "duration_us", "frequency_mhz", "phase_deg", "gain")
    values = np.zeros((len(template.points), 6))
    envelopes = []
    for event in slots:
        for col, name in enumerate(fields):
            values[:, col] += np.where(event.active, getattr(event, name), 0)
        # ReadoutPulse's phase is at pulse start, with a negative local carrier.
        values[:, 3] -= np.where(
            event.active, 360 * event.frequency_mhz * event.phase_reference_us, 0
        )
        values[:, 5] += np.where(event.active, event.envelope_index + len(envelopes), 0)
        envelopes.extend(_envelopes(event))
    features = np.column_stack(
        (values, window.trigger_us, window.frequency_mhz, window.phase_deg)
    )
    _, representatives, inverse = np.unique(
        features, axis=0, return_index=True, return_inverse=True
    )
    pulses = []
    grids = []
    dt = 1000 / window.sample_rate_mhz
    for index in representatives:
        start, duration, freq, phase, gain, envelope_index = values[index]
        envelope = envelopes[int(envelope_index)]
        amplitude = complex(settings.readout.drive.input_per_gain * gain)
        callback = None
        corners = ()
        if envelope.kind == "sampled":
            if np.all(envelope.samples == envelope.samples[0]):
                amplitude *= envelope.samples[0]
            else:
                callback = partial(_envelope_value, envelope)
                corners = tuple(
                    start * 1000 + t
                    for t in envelope.sample_ns
                    if 0 < t < duration * 1000
                )
        pulses.append(
            (
                ReadoutPulse(
                    start * 1000,
                    (start + duration) * 1000,
                    amplitude,
                    freq / 1000,
                    float(np.deg2rad(phase)),
                    callback,
                    corners,
                ),
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
    return values[:, 0] * 1000, tuple(pulses), tuple(grids), inverse.astype(np.int64)


def _envelope_value(envelope: EnvelopeSpec, local_ns: float) -> complex:
    return complex(envelope.values(np.array(local_ns)))


def plan_template(
    settings: FTFSettings,
    snapshot: EnvironmentSnapshot,
    template: TimelineTemplate,
) -> FTFTemplatePlan:
    """Convert one C1 template to C3 evolution and C6 readout inputs.

    settings is the immutable FTF resource, snapshot supplies live fixed-unit DC
    values, and template has point arrays in us/MHz/degrees. Support rounds to
    nearest 2 ns, ties even, without changing the common carrier origin. The
    single active readout pulse per point ends evolution; overlapping/later
    charge controls do not affect the DC readout model. Constant complex
    readout envelopes fold into amplitude, enabling the exact field solution.
    Missing routes, invalid DC values or multiple/missing readout pulses raise
    ValueError. Loops scale with slots, candidates and unique readout contents,
    not with points or reps. No old per-point planner is called.
    """
    end_ns, pulses, grids, inverse = _readout_contents(settings, template)
    joint = JointModel(
        joint_parameters(settings),
        dc_flux(settings, snapshot),
        temperature_k=settings.chip.temperature_k,
    )
    resonator_pair = resonators(settings)
    source = model_source(
        joint, drive_sources(settings, joint), resonators=resonator_pair
    )
    window = template.readouts[0]
    dt = 1000 / window.sample_rate_mhz
    receiver = Receiver(
        settings.readout.adc.gain,
        float(np.deg2rad(settings.readout.adc.phase_deg)),
        np.array(settings.readout.adc.offset_adc),
        np.array(settings.readout.adc.noise_psd_adc2_ns) / dt,
        time_of_flight_ns=settings.readout.adc.time_of_flight_ns,
    )
    model = replace(
        readout_model(joint, resonator_pair),
        receiver=receiver,
        reference_frequency_ghz=pulses[0][0].frequency_ghz,
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
    return FTFTemplatePlan(
        source,
        _timeline_batch(settings, joint, template, end_ns),
        ReadoutBatch(model, pulses, grids, inverse),
    )
