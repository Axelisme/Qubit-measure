"""Shared QICK drive/readout waveform conversion without environment physics."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    EventSlot,
    PulseEnvelope,
    ReadoutSlot,
    TimeBase,
    TimelineTemplate,
)
from zcu_tools.simulate.adc import (
    PiecewiseLinearEnvelope,
    ReadoutEvent,
    ReadoutPulse,
    SampleGrid,
)
from zcu_tools.simulate.evolution import DriveTrack, EnvelopeSpec


@dataclass(frozen=True)
class ReadoutDrive:
    """Detached generator calibration for readout waveform planning.

    gen_ch: nonnegative logical generator channel selected from EventSlot.gen_ch.
    input_per_gain: finite signed incident field amplitude per normalized DAC gain.
        It uses the amplitude units required by the caller's ADC physical model.
    kind: "pulse" requires one tone per read; "mux" permits 1..8 coherent tones.
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
class ReadoutTrigger:
    """One receiver trigger paired with a readout pulse block.

    channel: zero-based ADC index from ReadoutSlot.channel.
    read_index: zero-based column in that ADC's compiled trigger emission order.
    trigger_ns: absolute sampling start in ns from execution zero.
    frequency_ghz: demodulation frequency in GHz.
    phase_rad: demodulation phase in radians at execution zero.
    """

    channel: int
    read_index: int
    trigger_ns: float
    frequency_ghz: float
    phase_rad: float


@dataclass(frozen=True)
class ProjectiveRead:
    """One nonterminal readout block that projects a point's quantum state.

    start_ns: absolute common pulse start in ns, the projection time.
    pulses: coherent incident tones, in input slot order, using ns/GHz and
        pulse-local radians at zero-GHz reference, with calibrated amplitudes.
    triggers: bound ADC triggers in receiver declaration/trigger emission order.
        Multiple receivers or triggers on the same block cause one projection.
    """

    start_ns: float
    pulses: tuple[ReadoutPulse, ...]
    triggers: tuple[ReadoutTrigger, ...]


@dataclass(frozen=True)
class ReadoutPartition:
    """Terminal masks, projective reads and untriggered cavity drives.

    terminal_slots: readout-generator EventSlots in input slot order with active
        masks selecting only each point's latest block; other arrays are borrowed.
    cavity_drives: per-point tuples of untriggered earlier ReadoutPulse values in input
        slot order, with absolute ns, GHz and pulse-local radians referenced to
        zero GHz. Empty point tuples mean no earlier readout drive. Pulse sample
        arrays are detached; field amplitudes include signed gain/calibration.
    projective_reads: per-point tuples of triggered nonterminal blocks, sorted
        by start time. Empty point tuples mean no projective read.
    terminal_pulses: per-point terminal coherent tones, including when no ADC
        observes that block. Units and phase reference match cavity_drives.
    """

    terminal_slots: tuple[EventSlot, ...]
    cavity_drives: tuple[tuple[ReadoutPulse, ...], ...]
    projective_reads: tuple[tuple[ProjectiveRead, ...], ...]
    terminal_pulses: tuple[tuple[ReadoutPulse, ...], ...]


@dataclass(frozen=True)
class ReadoutContents:
    """Incident fields and one ADC's sampling grids for a read block.

    end_ns: owned float64 (points,) evolution endpoints at this pulse start.
    pulses: per-content tuples of coherent ReadoutPulse values with absolute ns,
        GHz and common-origin radians, including calibrated signed amplitudes.
        Constant envelopes fold into amplitude; shaped ones retain linear samples.
    grids: matching per-content tuples of SampleGrid values in trigger order,
        using absolute ns, GHz and radians without moving the demodulation origin.
    inverse: owned int64 (points,) indices selecting pulses/grids for each point.
        Terminal-only readout_contents deduplicates contents and sorts numeric
        pulse/ADC feature rows. Projective logical_event_contents retains one
        content per point, without deduplication, and uses an identity inverse.
    read_indices: per-content original channel-local trigger columns, aligned
        with grids. Empty tuples mark unobserved blocks. Across logical events,
        each point's indices cover its complete ADC emission order exactly once.
    All arrays/specs must remain unchanged while a consumer uses this plan.
    """

    end_ns: NDArray[np.float64]
    pulses: tuple[tuple[ReadoutPulse, ...], ...]
    grids: tuple[tuple[SampleGrid, ...], ...]
    inverse: NDArray[np.int64]
    read_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ReadoutMapping:
    """Localized numeric events and their channel-local output column mapping.

    events: per-content chronological ReadoutEvent tuples, in ns/GHz and
        pulse-local radians at the caller's rotating reference. Empty grids
        retain unobserved event positions without emitting columns.
    inverse: int64 (points,) selecting one events content per point. A single
        event retains its deduplication; multiple events use point-local
        contents and identity indices.
    output_order: int64 (contents, reads), where [c, r] selects the event-major
        column to emit as output column r for content c. Every row is a complete
        permutation. Use inverse to select a point's permutation, just as events.
    Keep the returned arrays and event records unchanged while in use.
    """

    events: tuple[tuple[ReadoutEvent, ...], ...]
    inverse: NDArray[np.int64]
    output_order: NDArray[np.int64]


def map_readout_events(
    contents: tuple[ReadoutContents, ...], reference_frequency_ghz: float
) -> ReadoutMapping:
    """Localize planned logical events and restore original ADC emission columns.

    contents is a nonempty chronological tuple from logical_event_contents, or
    one terminal-only readout_contents result. Its points and per-event grid
    counts must align. reference_frequency_ghz is the finite ADC model reference;
    phases follow local_readout_pulses. No physical model or sampling is built.
    Invalid event layout or incomplete/duplicate trigger indices raise ValueError.
    """
    if not contents:
        raise ValueError("readout mapping requires logical event contents")
    localized = tuple(
        local_readout_pulses(content.pulses, reference_frequency_ghz)
        for content in contents
    )
    if any(content.inverse.shape != contents[0].inverse.shape for content in contents):
        raise ValueError("readout event point layouts must align")
    if len(contents) == 1:
        content = contents[0]
        events = tuple(
            (ReadoutEvent(pulses, grids),)
            for pulses, grids in zip(localized[0], content.grids, strict=True)
        )
        inverse = content.inverse
        indices = content.read_indices
    else:
        events = tuple(
            tuple(
                ReadoutEvent(
                    pulses[content.inverse[p]], content.grids[content.inverse[p]]
                )
                for content, pulses in zip(contents, localized, strict=True)
            )
            for p in range(len(contents[0].inverse))
        )
        inverse = np.arange(len(events), dtype=np.int64)
        indices = tuple(
            tuple(
                index
                for content in contents
                for index in content.read_indices[content.inverse[p]]
            )
            for p in range(len(events))
        )
    reads = sum(len(event.grids) for event in events[0])
    if any(sorted(row) != list(range(reads)) for row in indices):
        raise ValueError("readout trigger indices must form a complete permutation")
    return ReadoutMapping(events, inverse, np.argsort(indices, axis=1).astype(np.int64))


def envelope_specs(
    event: EventSlot, memo: dict[int, EnvelopeSpec], *, time_base: TimeBase
) -> tuple[EnvelopeSpec, ...]:
    """Detach an EventSlot's candidate envelopes in their original order.

    Convert local ticks through time_base to ns without resampling; None means unity.
    memo is a caller-owned, initially empty identity cache shared by every event
    of one live template and time_base. Keep input envelopes alive while reusing it, do not
    modify cached specs, and discard it with that template. Returned immutable
    EnvelopeSpec values own their sample arrays. Invalid samples raise ValueError.
    """
    return tuple(
        _envelope_spec(envelope, memo, time_base) for envelope in event.envelopes
    )


def _envelope_spec(
    envelope: PulseEnvelope | None,
    memo: dict[int, EnvelopeSpec],
    time_base: TimeBase,
) -> EnvelopeSpec:
    key = id(envelope)
    if key not in memo:
        memo[key] = (
            EnvelopeSpec("constant")
            if envelope is None
            else EnvelopeSpec(
                "sampled",
                samples=np.array(envelope.values, dtype=np.complex128),
                sample_ns=time_base.ns(
                    np.arange(len(envelope.values), dtype=np.int64)
                    * envelope.sample_period_ticks
                ),
            )
        )
    return memo[key]


def drive_track(
    event: EventSlot, port: str, memo: dict[int, EnvelopeSpec], *, time_base: TimeBase
) -> DriveTrack:
    """Detach one QICK drive slot into an evolution track in ns/GHz/radians.

    event has the aligned per-point arrays defined by core.simulation.EventSlot
    in ticks/MHz/degrees, interpreted with the request's time_base. port is a nonempty numeric SecularDrive name, not a
    generator index. memo follows envelope_specs' template-local lifetime.
    Absolute support endpoints round independently to nearest 2 ns, ties even.
    Phase retains the common t=0 carrier origin, including phase_reference_ticks.
    Signed gains and complex samples are unchanged; inactive rows remain masked.
    Returned arrays own read-only copies. No calibration or physics is applied.
    Invalid supports, shapes, sample values or indices raise ValueError through
    DriveTrack/EnvelopeSpec. Inputs are not modified.
    """
    quantized = time_base.ns(event.start_ticks, quantum_ns=2)
    end = time_base.ns(event.start_ticks + event.duration_ticks, quantum_ns=2)
    return DriveTrack(
        port,
        quantized,
        end - quantized,
        event.frequency_mhz / 1000,
        np.deg2rad(event.phase_deg)
        - 2
        * np.pi
        * event.frequency_mhz
        * (time_base.ns(event.phase_reference_ticks) / 1000),
        event.gain,
        envelope_specs(event, memo, time_base=time_base),
        event.envelope_index,
        event.active,
    )


def classify_readout(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    memo: dict[int, EnvelopeSpec],
    *,
    time_base: TimeBase,
) -> ReadoutPartition:
    """Partition readout blocks into terminal, projective and cavity drives.

    time_base is the originating request's clock for every integer tick field.
    template uses core.simulation units and aligned per-point arrays; drive
    selects the generator and signed field calibration. memo follows
    envelope_specs' lifetime contract. The latest active start is terminal.
    Earlier blocks with their own ADC triggers are projective reads; other
    earlier pulses remain cavity drives. Simultaneous mux tones form one block.
    Pair triggers in [block start, next block start), including pulse ringdown.
    Triggers at the next start belong to that next block, not the preceding one.
    Return terminal masks in slot order, cavity pulses in point/slot order and
    projective reads in point/start order, with ns/GHz/pulse-local radians at
    zero-GHz reference. Earlier pulses may end exactly at the next block start.
    Missing terminal pulses, triggers before the first block, pulses crossing
    the next start or nonterminal samples ending at/after that start raise
    ValueError. Sampling-window errors name the block, ADC and trigger index.
    Projective block counts and each ADC's per-event grid counts must match
    across points; differences raise ValueError naming the point and counts.
    Inputs are not modified; terminal slots borrow unchanged input arrays.
    """
    slots = tuple(event for event in template.events if event.gen_ch == drive.gen_ch)
    if not slots:
        raise ValueError("requires one terminal readout pulse per point")
    latest = np.max([np.where(e.active, e.start_ticks, -np.inf) for e in slots], axis=0)
    if not np.isfinite(latest).all():
        raise ValueError("requires one terminal readout pulse per point")
    terminals = []
    terminal_pulses: list[list[ReadoutPulse]] = [[] for _ in template.points]
    drives: list[list[ReadoutPulse]] = [[] for _ in template.points]
    for event in slots:
        terminal = event.active & (event.start_ticks == latest)
        earlier = event.active & ~terminal
        next_start = np.min(
            [
                np.where(
                    other.active & (other.start_ticks > event.start_ticks),
                    other.start_ticks,
                    np.inf,
                )
                for other in slots
            ],
            axis=0,
        )
        if np.any(earlier & (event.start_ticks + event.duration_ticks > next_start)):
            if drive.kind == "mux":
                raise ValueError(
                    "mux tones require a common pulse start; blocks overlap"
                )
            if np.any(
                earlier
                & (next_start == latest)
                & (event.start_ticks + event.duration_ticks > latest)
            ):
                raise ValueError("cavity drive crosses terminal readout start")
            raise ValueError("readout pulse crosses next readout start")
        terminals.append(replace(event, active=terminal))
        envelopes = list(event.envelopes)
        rows = np.column_stack(
            (
                event.start_ticks,
                event.duration_ticks,
                event.frequency_mhz,
                event.phase_deg
                - 360
                * event.frequency_mhz
                * (time_base.ns(event.phase_reference_ticks) / 1000),
                event.gain,
                event.envelope_index,
            )
        )
        for index in np.flatnonzero(earlier):
            drives[index].append(
                _readout_pulse(drive, rows[index], envelopes, time_base, memo)
            )
        for index in np.flatnonzero(terminal):
            terminal_pulses[index].append(
                _readout_pulse(drive, rows[index], envelopes, time_base, memo)
            )
    cavity_drives = []
    projective_reads = []
    for point, pulses in enumerate(
        local_readout_pulses(tuple(tuple(p) for p in drives), 0.0)
    ):
        cavity, reads = _partition_point(template, drive, point, pulses, time_base)
        cavity_drives.append(cavity)
        projective_reads.append(reads)
    _validate_topology(template, tuple(projective_reads))
    return ReadoutPartition(
        tuple(terminals),
        tuple(cavity_drives),
        tuple(projective_reads),
        _terminal_pulses(terminal_pulses, drive),
    )


def _terminal_pulses(
    pulses: list[list[ReadoutPulse]], drive: ReadoutDrive
) -> tuple[tuple[ReadoutPulse, ...], ...]:
    """Validate terminal tone counts even without ADC observations and localize phases."""
    counts = tuple(len(point) for point in pulses)
    if drive.kind == "pulse" and any(count != 1 for count in counts):
        raise ValueError("requires one terminal readout pulse per point")
    if max(counts) > 8 or len(set(counts)) != 1:
        raise ValueError("mux requires a fixed 1..8 tone count per point")
    return local_readout_pulses(tuple(tuple(point) for point in pulses), 0.0)


def _validate_topology(
    template: TimelineTemplate, projective_reads: tuple[tuple[ProjectiveRead, ...], ...]
) -> None:
    expected = len(projective_reads[0])
    for point, reads in enumerate(projective_reads):
        if len(reads) != expected:
            raise ValueError(
                f"point {template.points[point]} projective block count {len(reads)} "
                f"differs from point {template.points[0]} count {expected}"
            )
    for window in template.readouts:
        layouts = []
        for reads in projective_reads:
            counts = tuple(
                sum(t.channel == window.channel for t in read.triggers)
                for read in reads
            )
            layouts.append((*counts, window.trigger_ticks.shape[1] - sum(counts)))
        for point, layout in enumerate(layouts):
            for event, (actual, expected_grids) in enumerate(
                zip(layout, layouts[0], strict=True)
            ):
                if actual != expected_grids:
                    raise ValueError(
                        f"point {template.points[point]} ADC {window.channel} event {event} "
                        f"grid count {actual} differs from point {template.points[0]} "
                        f"count {expected_grids}"
                    )


def _partition_point(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    point: int,
    pulses: tuple[ReadoutPulse, ...],
    time_base: TimeBase,
) -> tuple[tuple[ReadoutPulse, ...], tuple[ProjectiveRead, ...]]:
    first_start = min(
        event.start_ticks[point]
        for event in template.events
        if event.gen_ch == drive.gen_ch and event.active[point]
    )
    for window in template.readouts:
        if np.any(
            (window.generator_channels[point] == drive.gen_ch)
            & (window.trigger_ticks[point] < first_start)
        ):
            raise ValueError(
                "first ADC trigger must not precede terminal readout start"
            )
    reads = []
    triggered_starts = set()
    for start in sorted({pulse.start_ns for pulse in pulses}):
        block = tuple(pulse for pulse in pulses if pulse.start_ns == start)
        end = min(
            time_base.ns(event.start_ticks[point])
            for event in template.events
            if event.gen_ch == drive.gen_ch
            and event.active[point]
            and time_base.ns(event.start_ticks[point]) > start
        )
        triggers = _block_triggers(template, drive.gen_ch, point, start, end, time_base)
        if triggers:
            reads.append(ProjectiveRead(float(start), block, tuple(triggers)))
            triggered_starts.add(start)
    return (
        tuple(pulse for pulse in pulses if pulse.start_ns not in triggered_starts),
        tuple(reads),
    )


def _block_triggers(
    template: TimelineTemplate,
    gen_ch: int,
    point: int,
    start: float,
    end: float,
    time_base: TimeBase,
) -> tuple[ReadoutTrigger, ...]:
    triggers = []
    for window in template.readouts:
        for read, trigger in enumerate(time_base.ns(window.trigger_ticks[point])):
            if (
                window.generator_channels[point, read] == gen_ch
                and start <= trigger < end
            ):
                last_sample = time_base.ns(
                    int(window.trigger_ticks[point, read])
                    + (window.samples - 1) * window.sample_period_ticks
                )
                if last_sample >= end:
                    raise ValueError(
                        f"readout block start={start:g} ns, ADC {window.channel} "
                        f"trigger index={read} at {trigger:g} ns: last sample={last_sample:g} ns "
                        f"must precede next block start={end:g} ns"
                    )
                triggers.append(
                    ReadoutTrigger(
                        window.channel,
                        read,
                        float(trigger),
                        float(window.frequency_mhz[point, read] / 1000),
                        float(np.deg2rad(window.phase_deg[point, read])),
                    )
                )
    return tuple(triggers)


def logical_event_contents(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    window: ReadoutSlot,
    memo: dict[int, EnvelopeSpec],
    *,
    time_base: TimeBase,
) -> tuple[ReadoutContents, ...]:
    """Return projective blocks followed by terminal, including unobserved blocks.

    Inputs and phase/units follow readout_contents. window must belong to
    template.readouts and contain at least one trigger. Each returned content
    holds only that block's pulses and zero or more grids in channel-local
    trigger order. All ADCs therefore share one chronological logical event
    axis. read_indices retains the original emission columns across events;
    map_readout_events converts these to the final output permutation.
    Empty grids emit no read column. Cross-point topology differences,
    invalid pulse counts/routes and crossing windows raise ValueError through
    classify_readout. Terminal-only plans retain readout_contents' exact
    deduplication and phases. Projective plans use point-local contents with
    identity inverse maps. No densities or ADC responses are constructed.
    """
    partition = classify_readout(template, drive, memo, time_base=time_base)
    if not any(partition.projective_reads):
        return (
            readout_contents(
                replace(template, events=partition.terminal_slots),
                drive,
                window,
                memo,
                time_base=time_base,
            ),
        )
    observed = read_event_contents(template, drive, window, memo, time_base=time_base)
    blocks = tuple(
        (*(read.pulses for read in reads), terminal)
        for reads, terminal in zip(
            partition.projective_reads, partition.terminal_pulses, strict=True
        )
    )
    contents = []
    for event in range(len(blocks[0])):
        pulses = tuple(
            tuple(
                replace(
                    pulse,
                    phase_rad=pulse.phase_rad
                    + 2 * np.pi * pulse.frequency_ghz * pulse.start_ns,
                )
                for pulse in point[event]
            )
            for point in blocks
        )
        starts = np.array([point[event][0].start_ns for point in blocks])
        grids = tuple(
            tuple(
                grid
                for content in observed
                if content.end_ns[point] == start
                for grid in content.grids[content.inverse[point]]
            )
            for point, start in enumerate(starts)
        )
        indices = tuple(
            tuple(
                read
                for read, content in enumerate(observed)
                if content.end_ns[point] == start
            )
            for point, start in enumerate(starts)
        )
        contents.append(
            ReadoutContents(
                starts,
                pulses,
                grids,
                np.arange(len(template.points), dtype=np.int64),
                indices,
            )
        )
    return tuple(contents)


def read_event_contents(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    window: ReadoutSlot,
    memo: dict[int, EnvelopeSpec],
    *,
    time_base: TimeBase,
) -> tuple[ReadoutContents, ...]:
    """Plan each ADC trigger with its own incident pulse block.

    template, drive and memo follow readout_contents' units and lifetime; window
    must be one of template.readouts so its channel/trigger indices match the
    classification. Triggers pair in [block start, next block start).
    Classify the complete template, then return one ReadoutContents per column
    of window, in trigger emission order. Each content has exactly one grid;
    end_ns is that read's pulse start at each point, not the terminal endpoint.
    Pulses retain common-origin phases for later local_readout_pulses conversion.
    Repeated triggers on one block use the same pulses but independent grids.
    Missing pulse blocks, crossing blocks, invalid tone counts and wrong ADC
    routes raise ValueError. This function builds no physical model or samples.
    """
    partition = classify_readout(template, drive, memo, time_base=time_base)
    terminal_start = np.max(
        [
            np.where(e.active, time_base.ns(e.start_ticks), -np.inf)
            for e in partition.terminal_slots
        ],
        axis=0,
    )
    contents = []
    for read in range(window.trigger_ticks.shape[1]):
        starts = terminal_start.copy()
        for point, projections in enumerate(partition.projective_reads):
            for projection in projections:
                if any(
                    t.channel == window.channel and t.read_index == read
                    for t in projection.triggers
                ):
                    starts[point] = projection.start_ns
        events = tuple(
            replace(e, active=e.active & (time_base.ns(e.start_ticks) == starts))
            for e in template.events
            if e.gen_ch == drive.gen_ch
        )
        selected_window = replace(
            window,
            trigger_ticks=window.trigger_ticks[:, read : read + 1],
            frequency_mhz=window.frequency_mhz[:, read : read + 1],
            phase_deg=window.phase_deg[:, read : read + 1],
            generator_channels=window.generator_channels[:, read : read + 1],
        )
        content = readout_contents(
            replace(template, events=events),
            drive,
            selected_window,
            memo,
            time_base=time_base,
        )
        contents.append(
            replace(content, read_indices=tuple((read,) for _ in content.grids))
        )
    if not contents:
        raise ValueError("requires ADC timing and its bound readout generator")
    return tuple(contents)


def readout_contents(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    window: ReadoutSlot,
    memo: dict[int, EnvelopeSpec],
    *,
    time_base: TimeBase,
) -> ReadoutContents:
    """Deduplicate terminal pulses and one receiver's grids by complete content.

    time_base is the originating request's clock. Its ns range/type errors
    propagate for invalid tick fields. template contains only terminal readout
    events on drive.gen_ch; other
    generators are ignored. window has aligned core.simulation ADC trigger arrays.
    memo follows envelope_specs' lifetime contract. Signed gains and complex
    envelopes are preserved. Pulses use ns/GHz and common-origin radians; call
    local_readout_pulses once the ADC reference frequency is selected. Grids use
    absolute ns/GHz/radians. time_base interprets all tick fields; grids combine
    trigger and integer sample offsets before one ns conversion.
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
    if not window.trigger_ticks.shape[1] or np.any(
        window.generator_channels != drive.gen_ch
    ):
        raise ValueError("requires ADC timing and its bound readout generator")
    fields = ("start_ticks", "duration_ticks", "frequency_mhz", "phase_deg", "gain")
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
            event.active,
            360
            * event.frequency_mhz
            * (time_base.ns(event.phase_reference_ticks) / 1000),
            0,
        )
        values[rows, rank, 5] += np.where(
            event.active, event.envelope_index + len(envelopes), 0
        )
        envelopes.extend(event.envelopes)
    if np.any(values[:, :, 0] != values[:, :1, 0]):
        raise ValueError("mux tones require a common pulse start per point")
    if np.any(window.trigger_ticks[:, 0] < values[:, 0, 0]):
        raise ValueError("first ADC trigger must not precede terminal readout start")
    features = np.column_stack(
        (
            values.reshape(len(template.points), -1),
            window.trigger_ticks,
            window.frequency_mhz,
            window.phase_deg,
        )
    )
    _, representatives, inverse = np.unique(
        features, axis=0, return_index=True, return_inverse=True
    )
    pulses = []
    grids = []
    offsets = np.arange(window.samples, dtype=np.int64) * window.sample_period_ticks
    for index in representatives:
        pulses.append(
            tuple(
                _readout_pulse(drive, row, envelopes, time_base, memo)
                for row in values[index]
            )
        )
        grids.append(
            tuple(
                SampleGrid(
                    time_base.ns(trigger + offsets),
                    frequency / 1000,
                    float(np.deg2rad(phase)),
                )
                for trigger, frequency, phase in zip(
                    window.trigger_ticks[index],
                    window.frequency_mhz[index],
                    window.phase_deg[index],
                    strict=True,
                )
            )
        )
    return ReadoutContents(
        time_base.ns(values[:, 0, 0].astype(np.int64)),
        tuple(pulses),
        tuple(grids),
        inverse.astype(np.int64),
        (tuple(range(window.trigger_ticks.shape[1])),) * len(grids),
    )


def _readout_pulse(
    drive: ReadoutDrive,
    row: NDArray[np.float64],
    envelopes: list[PulseEnvelope | None],
    time_base: TimeBase,
    memo: dict[int, EnvelopeSpec],
) -> ReadoutPulse:
    start, duration, freq, phase, gain, envelope_index = row
    start_ticks, duration_ticks = int(start), int(duration)
    envelope = envelopes[int(envelope_index)]
    amplitude = complex(drive.input_per_gain * gain)
    callback = None
    corners = ()
    if envelope is not None:
        spec = _envelope_spec(envelope, memo, time_base)
        samples = spec.samples
        if np.all(samples == samples[0]):
            amplitude *= samples[0]
        else:
            axis = (
                np.arange(len(samples), dtype=np.int64) * envelope.sample_period_ticks
            )
            callback = PiecewiseLinearEnvelope(spec.sample_ns, samples)
            corners = tuple(
                time_base.ns(start_ticks + axis[(axis > 0) & (axis < duration_ticks)])
            )
    return ReadoutPulse(
        time_base.ns(start_ticks),
        time_base.ns(start_ticks + duration_ticks),
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
