"""Shared QICK drive/readout waveform conversion without environment physics."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import EventSlot, ReadoutSlot, TimelineTemplate
from zcu_tools.simulate.adc import PiecewiseLinearEnvelope, ReadoutPulse, SampleGrid
from zcu_tools.simulate.evolution import DriveTrack, EnvelopeSpec


@dataclass(frozen=True)
class ReadoutDrive:
    """Detached generator calibration for readout waveform planning.

    gen_ch: nonnegative logical generator channel selected from EventSlot.gen_ch.
    input_per_gain: finite signed incident field amplitude per normalized DAC gain.
        It uses the amplitude units required by the caller's ADC physical model.
    kind: "pulse" requires one terminal tone; "mux" permits 1..8 coherent tones.
    Invalid values raise ValueError at construction. No settings are retained.
    """

    gen_ch: int
    input_per_gain: float
    kind: Literal["pulse", "mux"] = "pulse"

    def __post_init__(self) -> None:
        if self.gen_ch < 0 or not np.isfinite(self.input_per_gain):
            raise ValueError("readout generator and finite field calibration required")
        if self.kind not in ("pulse", "mux"):
            raise ValueError("readout kind must be pulse or mux")


@dataclass(frozen=True)
class ReadoutPartition:
    """Terminal masks and earlier incident drives for one template.

    terminal_slots: readout-generator EventSlots in input slot order with active
        masks selecting only each point's latest block; other arrays are borrowed.
    cavity_drives: per-point tuples of earlier ADC ReadoutPulse values in input
        slot order, with absolute ns, GHz and pulse-local radians referenced to
        zero GHz. Empty point tuples mean no earlier readout drive. Pulse sample
        arrays are detached; field amplitudes include signed gain/calibration.
    """

    terminal_slots: tuple[EventSlot, ...]
    cavity_drives: tuple[tuple[ReadoutPulse, ...], ...]


@dataclass(frozen=True)
class ReadoutContents:
    """Distinct terminal incident fields and one ADC's sampling grids.

    end_ns: owned float64 (points,) evolution endpoints at terminal pulse start.
    pulses: per-content tuples of coherent ReadoutPulse values with absolute ns,
        GHz and common-origin radians, including calibrated signed amplitudes.
        Constant envelopes fold into amplitude; shaped ones retain linear samples.
    grids: matching per-content tuples of SampleGrid values in trigger order,
        using absolute ns, GHz and radians without moving the demodulation origin.
    inverse: owned int64 (points,) indices selecting pulses/grids for each point.
        Distinct contents are sorted by numeric pulse and ADC feature rows.
    All arrays/specs must remain unchanged while a consumer uses this plan.
    """

    end_ns: NDArray[np.float64]
    pulses: tuple[tuple[ReadoutPulse, ...], ...]
    grids: tuple[tuple[SampleGrid, ...], ...]
    inverse: NDArray[np.int64]


def envelope_specs(
    event: EventSlot, memo: dict[int, EnvelopeSpec]
) -> tuple[EnvelopeSpec, ...]:
    """Detach an EventSlot's candidate envelopes in their original order.

    Convert local us to ns without resampling complex values; None means unity.
    memo is a caller-owned, initially empty identity cache shared by every event
    of one live template. Keep all input envelopes alive while reusing it, do not
    modify cached specs, and discard it with that template. Returned immutable
    EnvelopeSpec values own their sample arrays. Invalid samples raise ValueError.
    """
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


def drive_track(
    event: EventSlot, port: str, memo: dict[int, EnvelopeSpec]
) -> DriveTrack:
    """Detach one QICK drive slot into an evolution track in ns/GHz/radians.

    event has the aligned per-point arrays defined by core.simulation.EventSlot
    in us/MHz/degrees. port is a nonempty numeric SecularDrive name, not a
    generator index. memo follows envelope_specs' template-local lifetime.
    Absolute support endpoints round independently to nearest 2 ns, ties even.
    Phase retains the common t=0 carrier origin, including phase_reference_us.
    Signed gains and complex samples are unchanged; inactive rows remain masked.
    Returned arrays own read-only copies. No calibration or physics is applied.
    Invalid supports, shapes, sample values or indices raise ValueError through
    DriveTrack/EnvelopeSpec. Inputs are not modified.
    """
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
        envelope_specs(event, memo),
        event.envelope_index,
        event.active,
    )


def classify_readout(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    memo: dict[int, EnvelopeSpec],
) -> ReadoutPartition:
    """Partition readout-generator events at each point's latest active start.

    template uses core.simulation units and aligned per-point arrays; drive
    selects the generator and signed field calibration. memo follows
    envelope_specs' lifetime contract. Return terminal masks in slot order and
    earlier cavity pulses in point order, in ns/GHz/pulse-local radians with
    zero-GHz reference. Earlier pulses may end exactly at terminal start.
    Missing terminal pulses or earlier pulses crossing that start raise ValueError.
    Inputs are not modified; terminal slots borrow unchanged input arrays.
    """
    slots = tuple(event for event in template.events if event.gen_ch == drive.gen_ch)
    if not slots:
        raise ValueError("requires one terminal readout pulse per point")
    latest = np.max([np.where(e.active, e.start_us, -np.inf) for e in slots], axis=0)
    if not np.isfinite(latest).all():
        raise ValueError("requires one terminal readout pulse per point")
    terminals = []
    drives: list[list[ReadoutPulse]] = [[] for _ in template.points]
    for event in slots:
        terminal = event.active & (event.start_us == latest)
        earlier = event.active & ~terminal
        if np.any(earlier & (event.start_us + event.duration_us > latest)):
            if drive.kind == "mux":
                raise ValueError(
                    "mux tones require a common pulse start; blocks overlap"
                )
            raise ValueError("cavity drive crosses terminal readout start")
        terminals.append(replace(event, active=terminal))
        envelopes = list(envelope_specs(event, memo))
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
            drives[index].append(_readout_pulse(drive, rows[index], envelopes))
    return ReadoutPartition(
        tuple(terminals), local_readout_pulses(tuple(tuple(p) for p in drives), 0.0)
    )


def readout_contents(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    window: ReadoutSlot,
    memo: dict[int, EnvelopeSpec],
) -> ReadoutContents:
    """Deduplicate terminal pulses and one receiver's grids by complete content.

    template contains only terminal readout events on drive.gen_ch; other
    generators are ignored. window has aligned core.simulation ADC trigger arrays.
    memo follows envelope_specs' lifetime contract. Signed gains and complex
    envelopes are preserved. Pulses use ns/GHz and common-origin radians; call
    local_readout_pulses once the ADC reference frequency is selected. Grids use
    absolute ns/GHz/radians and dt=1000/window.sample_rate_mhz.
    Return owned arrays with sorted distinct content and a point-to-content map.
    Missing pulses, varying tone count, pulse mode with multiple tones, mux with
    more than eight tones or different starts, wrong ADC generator routes, empty
    trigger axes or a first trigger before terminal start raise ValueError.
    Core template/window arrays must satisfy their documented shapes and units.
    No physical model, density propagation or sampling is constructed.
    """
    slots = tuple(e for e in template.events if e.gen_ch == drive.gen_ch)
    count = np.sum([e.active for e in slots], axis=0)
    if not slots or np.any(count == 0):
        raise ValueError("requires one terminal readout pulse per point")
    if drive.kind == "pulse" and np.any(count != 1):
        raise ValueError("requires one terminal readout pulse per point")
    tones = int(count[0])
    if tones > 8 or np.any(count != tones):
        raise ValueError("mux requires a fixed 1..8 tone count per point")
    if not window.trigger_us.shape[1] or np.any(
        window.generator_channels != drive.gen_ch
    ):
        raise ValueError("requires ADC timing and its bound readout generator")
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
        envelopes.extend(envelope_specs(event, memo))
    if np.any(values[:, :, 0] != values[:, :1, 0]):
        raise ValueError("mux tones require a common pulse start per point")
    if np.any(window.trigger_us[:, 0] < values[:, 0, 0]):
        raise ValueError("first ADC trigger must not precede terminal readout start")
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
            tuple(_readout_pulse(drive, row, envelopes) for row in values[index])
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
    return ReadoutContents(
        values[:, 0, 0] * 1000, tuple(pulses), tuple(grids), inverse.astype(np.int64)
    )


def _readout_pulse(
    drive: ReadoutDrive, row: NDArray[np.float64], envelopes: list[EnvelopeSpec]
) -> ReadoutPulse:
    start, duration, freq, phase, gain, envelope_index = row
    envelope = envelopes[int(envelope_index)]
    amplitude = complex(drive.input_per_gain * gain)
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


def local_readout_pulses(
    contents: tuple[tuple[ReadoutPulse, ...], ...], reference_frequency_ghz: float
) -> tuple[tuple[ReadoutPulse, ...], ...]:
    """Convert common-origin phases to ADC's negative pulse-local carrier.

    contents contains point/content tuples of ADC ReadoutPulse values in ns/GHz
    with common-origin phase_rad. reference_frequency_ghz is the model's finite
    rotating-frame reference. Return the same layout with each phase reduced by
    2*pi*(frequency_ghz-reference_frequency_ghz)*start_ns; the reference carrier
    stays unchanged bitwise. Input pulses are not modified. Nonfinite reference
    raises ValueError. ADC demodulation grids must retain their absolute origin.
    """
    if not np.isfinite(reference_frequency_ghz):
        raise ValueError("readout reference frequency must be finite")
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
