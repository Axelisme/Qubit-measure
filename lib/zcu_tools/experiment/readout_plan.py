"""Shared QICK drive/readout waveform conversion without environment physics."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.adc import (
    PiecewiseLinearEnvelope,
    ReadoutEvent,
    ReadoutPulse,
    SampleGrid,
    adc_to_device_time_ns,
)
from zcu_tools.simulate.boundary import (
    EventSlot,
    PulseEnvelope,
    ReadoutSlot,
    TimeBase,
    TimelineTemplate,
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
    incident_starts_ns: absolute emission starts of the blocks intersecting this
        trigger's device-time window, in chronological order. Empty denotes a
        receiver-only window anchored to an otherwise unobserved event.
    """

    channel: int
    read_index: int
    trigger_ns: float
    frequency_ghz: float
    phase_rad: float
    incident_starts_ns: tuple[float, ...]


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
    terminal_triggers: per-point triggers assigned to terminal in receiver
        declaration/trigger emission order. Empty tuples mean no observation.
    """

    terminal_slots: tuple[EventSlot, ...]
    cavity_drives: tuple[tuple[ReadoutPulse, ...], ...]
    projective_reads: tuple[tuple[ProjectiveRead, ...], ...]
    terminal_pulses: tuple[tuple[ReadoutPulse, ...], ...]
    terminal_triggers: tuple[tuple[ReadoutTrigger, ...], ...]


@dataclass(frozen=True)
class ReadoutContents:
    """Incident fields and one ADC's sampling grids for a read block.

    end_ns: owned float64 (points,) evolution endpoints at the pulse start or,
        for receiver-only contents, the first trigger.
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

    event has the aligned per-point arrays defined by simulate.boundary.EventSlot
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
    receiver_delays_ns: Mapping[int, float] | None = None,
    allow_receiver_only: bool = False,
) -> ReadoutPartition:
    """Partition readout blocks into terminal, projective and cavity drives.

    time_base is the originating request's clock for every integer tick field.
    template uses simulate.boundary units and aligned per-point arrays; drive
    selects the generator and signed field calibration. memo follows
    envelope_specs' lifetime contract. The latest active start is terminal.
    Earlier blocks with their own ADC triggers are projective reads; other
    earlier pulses remain cavity drives. Simultaneous mux tones form one block.
    receiver_delays_ns maps each ADC channel to its flight time in ns. A missing
    channel raises KeyError; None means zero delay for every ADC. The ADC
    conversion rejects invalid delays. Pair each device-time window with the
    blocks it intersects, from each block's start up to (not including) the next
    start or point end.
    This includes ringdown. The latest intersecting block owns the trigger.
    Windows entirely before the first block use an otherwise unobserved event for
    receiver-only output, or raise ValueError if no event is available without a
    new projection. A window wholly beyond point end also raises ValueError.
    Return terminal masks in slot order, cavity pulses in point/slot order and
    projective reads in point/start order, with ns/GHz/pulse-local radians at
    zero-GHz reference. Earlier pulses may end exactly at the next block start.
    With allow_receiver_only, a generator with no active pulses anywhere in the
    template returns no terminal slots and empty tuples per point. Otherwise,
    missing terminal pulses and pulses crossing the next start raise ValueError.
    Projective block counts and each ADC's per-event grid counts must match
    across points; differences raise ValueError naming the point and counts.
    Inputs are not modified; terminal slots borrow unchanged input arrays.
    """
    slots = tuple(event for event in template.events if event.gen_ch == drive.gen_ch)
    latest = (
        np.max([np.where(e.active, e.start_ticks, -np.inf) for e in slots], axis=0)
        if slots
        else np.full(len(template.points), -np.inf)
    )
    if not np.isfinite(latest).all():
        if allow_receiver_only and not np.isfinite(latest).any():
            empty = ((),) * len(template.points)
            return ReadoutPartition((), empty, empty, empty, empty)
        raise ValueError("requires one terminal readout pulse per point")
    terminals = []
    terminal_pulses: list[list[ReadoutPulse]] = [[] for _ in template.points]
    drives: list[list[ReadoutPulse]] = [[] for _ in template.points]
    for event in slots:
        terminal = event.active & (event.start_ticks == latest)
        earlier = event.active & ~terminal
        _validate_block_support(event, slots, latest, earlier, drive.kind)
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
    terminal_blocks = _terminal_pulses(terminal_pulses, drive)
    cavity_drives, projective_reads, terminal_triggers = [], [], []
    for point, pulses in enumerate(
        local_readout_pulses(tuple(tuple(p) for p in drives), 0.0)
    ):
        terminal_start = terminal_blocks[point][0].start_ns
        starts = (*sorted({pulse.start_ns for pulse in pulses}), terminal_start)
        triggers = _point_triggers(
            template, drive, point, starts, time_base, receiver_delays_ns
        )
        reads = _projective_blocks(pulses, starts[:-1], triggers)
        cavity_drives.append(
            tuple(pulse for pulse in pulses if pulse.start_ns not in triggers)
        )
        projective_reads.append(reads)
        terminal_triggers.append(triggers.get(terminal_start, ()))
    partition = ReadoutPartition(
        tuple(terminals),
        tuple(cavity_drives),
        tuple(projective_reads),
        terminal_blocks,
        tuple(terminal_triggers),
    )
    _validate_topology(template, partition)
    return partition


def _projective_blocks(
    pulses: tuple[ReadoutPulse, ...],
    starts: tuple[float, ...],
    triggers: Mapping[float, tuple[ReadoutTrigger, ...]],
) -> tuple[ProjectiveRead, ...]:
    return tuple(
        ProjectiveRead(
            start,
            tuple(pulse for pulse in pulses if pulse.start_ns == start),
            triggers[start],
        )
        for start in starts
        if start in triggers
    )


def _validate_block_support(
    event: EventSlot,
    slots: tuple[EventSlot, ...],
    terminal_starts: NDArray[np.float64],
    earlier: NDArray[np.bool_],
    kind: Literal["pulse", "mux"],
) -> None:
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
    crossing = earlier & (event.start_ticks + event.duration_ticks > next_start)
    if not np.any(crossing):
        return
    if kind == "mux":
        raise ValueError("mux tones require a common pulse start; blocks overlap")
    if np.any(crossing & (next_start == terminal_starts)):
        raise ValueError("cavity drive crosses terminal readout start")
    raise ValueError("readout pulse crosses next readout start")


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


def _validate_topology(template: TimelineTemplate, partition: ReadoutPartition) -> None:
    expected = len(partition.projective_reads[0])
    for point, reads in enumerate(partition.projective_reads):
        if len(reads) != expected:
            raise ValueError(
                f"point {template.points[point]} projective block count {len(reads)} "
                f"differs from point {template.points[0]} count {expected}"
            )
    for window in template.readouts:
        layouts = []
        for reads, terminal in zip(
            partition.projective_reads, partition.terminal_triggers, strict=True
        ):
            layouts.append(
                tuple(
                    sum(t.channel == window.channel for t in triggers)
                    for triggers in (*(read.triggers for read in reads), terminal)
                )
            )
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


def _point_triggers(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    point: int,
    starts: tuple[float, ...],
    time_base: TimeBase,
    receiver_delays_ns: Mapping[int, float] | None,
) -> dict[float, tuple[ReadoutTrigger, ...]]:
    ends = (*starts[1:], time_base.ns(template.duration_ticks[point]))
    assigned: dict[float, list[ReadoutTrigger]] = {}
    receiver_only = []
    for window in template.readouts:
        delay = (
            0.0 if receiver_delays_ns is None else receiver_delays_ns[window.channel]
        )
        ticks = window.trigger_ticks[point]
        first = time_base.ns(ticks)
        last = time_base.ns(ticks + (window.samples - 1) * window.sample_period_ticks)
        device_first = adc_to_device_time_ns(first, time_of_flight_ns=delay)
        device_last = adc_to_device_time_ns(last, time_of_flight_ns=delay)
        for read in np.flatnonzero(window.generator_channels[point] == drive.gen_ch):
            members = tuple(
                start
                for start, end in zip(starts, ends, strict=True)
                if start < end
                and start <= device_last[read]
                and device_first[read] < end
            )
            trigger = ReadoutTrigger(
                window.channel,
                int(read),
                float(first[read]),
                float(window.frequency_mhz[point, read] / 1000),
                float(np.deg2rad(window.phase_deg[point, read])),
                members,
            )
            if members:
                assigned.setdefault(members[-1], []).append(trigger)
            elif device_last[read] < starts[0]:
                receiver_only.append(trigger)
            else:
                raise ValueError(
                    f"ADC {window.channel} trigger index={read} at point "
                    f"{template.points[point]} has no readout block before point end"
                )

    # A receiver-only window can use an existing projection or terminal event
    # that its ADC does not observe. It never creates a projection by itself.
    anchors = (starts[-1], *(start for start in starts[:-1] if start in assigned))
    for trigger in receiver_only:
        for anchor in anchors:
            seen = assigned.setdefault(anchor, [])
            if all(
                t.channel != trigger.channel or not t.incident_starts_ns for t in seen
            ):
                seen.append(trigger)
                break
        else:
            raise ValueError(
                f"ADC {trigger.channel} trigger index={trigger.read_index} at point "
                f"{template.points[point]} precedes all readout blocks and needs an "
                "unobserved event for receiver-only output without a new projection"
            )

    channel_order = {window.channel: i for i, window in enumerate(template.readouts)}
    return {
        owner: tuple(
            sorted(triggers, key=lambda t: (channel_order[t.channel], t.read_index))
        )
        for owner, triggers in assigned.items()
    }


def _selected_pulses(
    blocks: Mapping[float, tuple[ReadoutPulse, ...]],
    owner: float,
    triggers: tuple[ReadoutTrigger, ...],
) -> tuple[ReadoutPulse, ...]:
    """Union the windows' incident blocks, keeping the owner's tones first."""
    if not triggers:
        return blocks[owner]
    starts = sorted(
        {start for trigger in triggers for start in trigger.incident_starts_ns}
    )
    if not starts:
        return ()
    return (
        *blocks[owner],
        *(p for start in starts if start != owner for p in blocks[start]),
    )


def _readout_blocks(
    partition: ReadoutPartition, point: int
) -> dict[float, tuple[ReadoutPulse, ...]]:
    blocks = {read.start_ns: read.pulses for read in partition.projective_reads[point]}
    terminal = partition.terminal_pulses[point]
    blocks[terminal[0].start_ns] = terminal
    for pulse in partition.cavity_drives[point]:
        blocks[pulse.start_ns] = (*blocks.get(pulse.start_ns, ()), pulse)
    return blocks


def _common_readout_phase(pulse: ReadoutPulse) -> ReadoutPulse:
    return replace(
        pulse,
        phase_rad=pulse.phase_rad + 2 * np.pi * pulse.frequency_ghz * pulse.start_ns,
    )


@dataclass(frozen=True)
class _ReceiverEvent:
    owner_ns: float
    pulses: tuple[ReadoutPulse, ...]
    read_indices: tuple[int, ...]


def _receiver_events(
    partition: ReadoutPartition, channel: int, *, by_trigger: bool = False
) -> tuple[tuple[_ReceiverEvent, ...], ...]:
    rows = []
    for point, reads in enumerate(partition.projective_reads):
        blocks = _readout_blocks(partition, point)
        terminal = partition.terminal_pulses[point][0].start_ns
        owners = {read.start_ns: read.triggers for read in reads}
        owners[terminal] = partition.terminal_triggers[point]
        events = []
        for owner, observed in owners.items():
            triggers = tuple(t for t in observed if t.channel == channel)
            groups = (
                tuple((trigger,) for trigger in triggers) if by_trigger else (triggers,)
            )
            events.extend(
                _ReceiverEvent(
                    owner,
                    _selected_pulses(blocks, owner, group),
                    tuple(t.read_index for t in group),
                )
                for group in groups
            )
        if by_trigger:
            events.sort(key=lambda event: event.read_indices)
        rows.append(tuple(events))
    return tuple(rows)


def logical_event_contents(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    window: ReadoutSlot,
    memo: dict[int, EnvelopeSpec],
    *,
    time_base: TimeBase,
    partition: ReadoutPartition | None = None,
) -> tuple[ReadoutContents, ...]:
    """Return each projective block then terminal on a common logical event axis.

    Inputs and phase/units follow readout_contents. window must belong to
    template.readouts and contain at least one trigger. An observed event holds
    the union of blocks matched to its grids; a receiver-only event has no
    pulses. Unobserved events retain the owner's pulse block with no grids.
    Empty grids emit no read column. read_indices retains ADC emission order.
    Terminal-only plans with the same one-block topology retain deduplication;
    projective or mixed membership uses point-local contents with identity maps.
    Pass a partition for the same template, drive and time base, or classify
    with zero delay. Partitions with no pulse anywhere must instead use
    readout_contents with allow_receiver_only. Invalid layouts or routes raise
    ValueError. No density or ADC response is constructed.
    """
    if partition is None:
        partition = classify_readout(template, drive, memo, time_base=time_base)
    if not partition.terminal_slots:
        raise ValueError("logical readout events require a terminal pulse block")
    terminal = readout_contents(
        replace(template, events=partition.terminal_slots),
        drive,
        window,
        memo,
        time_base=time_base,
    )
    rows = _receiver_events(partition, window.channel)
    if not any(partition.projective_reads):
        if all(_owner_pulses_only(row[0]) for row in rows):
            return (terminal,)
        if all(not row[0].pulses for row in rows):
            return (replace(terminal, pulses=((),) * len(terminal.grids)),)
    return tuple(
        _materialize_event(
            terminal,
            events,
            keep_owner_phase=index == len(rows[0]) - 1
            and not any(partition.projective_reads),
        )
        for index, events in enumerate(zip(*rows, strict=True))
    )


def _owner_pulses_only(event: _ReceiverEvent) -> bool:
    return bool(event.pulses) and all(
        p.start_ns == event.owner_ns for p in event.pulses
    )


def _materialize_event(
    source: ReadoutContents,
    events: tuple[_ReceiverEvent, ...],
    *,
    by_trigger: bool = False,
    keep_owner_phase: bool = False,
) -> ReadoutContents:
    pulses = []
    grids = []
    for row, event in zip(source.inverse, events, strict=True):
        pulses.append(
            source.pulses[row]
            + tuple(
                _common_readout_phase(pulse)
                for pulse in event.pulses
                if pulse.start_ns != event.owner_ns
            )
            if keep_owner_phase and event.pulses
            else tuple(_common_readout_phase(pulse) for pulse in event.pulses)
        )
        grids.append(
            source.grids[row]
            if by_trigger
            else tuple(source.grids[row][read] for read in event.read_indices)
        )
    return ReadoutContents(
        np.array([event.owner_ns for event in events]),
        tuple(pulses),
        tuple(grids),
        np.arange(len(events), dtype=np.int64),
        tuple(event.read_indices for event in events),
    )


def read_event_contents(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    window: ReadoutSlot,
    memo: dict[int, EnvelopeSpec],
    *,
    time_base: TimeBase,
) -> tuple[ReadoutContents, ...]:
    """Plan each trigger's incident pulses with zero receiver delay.

    The window must be one of template.readouts. Return one ReadoutContents
    per trigger in ADC emission order. Each has one grid, all matched pulses
    with common-origin phases, and end_ns at the owning block's start. A
    receiver-only window has no pulses and uses an unobserved block's start.
    Single-block contents are deduplicated, others use a point-local map.
    Invalid blocks, windows or ADC routes raise ValueError. No physical model
    or samples are constructed.
    """
    if not window.trigger_ticks.shape[1] or np.any(
        window.generator_channels != drive.gen_ch
    ):
        raise ValueError("requires ADC timing and its bound readout generator")
    partition = classify_readout(template, drive, memo, time_base=time_base)
    contents = []
    for read, events in enumerate(
        zip(*_receiver_events(partition, window.channel, by_trigger=True), strict=True)
    ):
        starts = np.array([event.owner_ns for event in events])
        slots = tuple(
            replace(e, active=e.active & (time_base.ns(e.start_ticks) == starts))
            for e in template.events
            if e.gen_ch == drive.gen_ch
        )
        content = readout_contents(
            replace(template, events=slots),
            drive,
            window.subset(reads=slice(read, read + 1)),
            memo,
            time_base=time_base,
        )
        if all(_owner_pulses_only(event) for event in events):
            contents.append(
                replace(content, read_indices=((read,),) * len(content.grids))
            )
        else:
            contents.append(
                _materialize_event(
                    content, events, by_trigger=True, keep_owner_phase=True
                )
            )
    return tuple(contents)


def readout_contents(
    template: TimelineTemplate,
    drive: ReadoutDrive,
    window: ReadoutSlot,
    memo: dict[int, EnvelopeSpec],
    *,
    time_base: TimeBase,
    allow_receiver_only: bool = False,
) -> ReadoutContents:
    """Deduplicate terminal pulses and one receiver's grids by complete content.

    time_base is the originating request's clock. Its ns range/type errors
    propagate for invalid tick fields. template contains only terminal readout
    events on drive.gen_ch; other
    generators are ignored. window has aligned simulate.boundary ADC trigger arrays.
    memo follows envelope_specs' lifetime contract. Signed gains and complex
    envelopes are preserved. Pulses use ns/GHz and common-origin radians; call
    local_readout_pulses once the ADC reference frequency is selected. Grids use
    absolute ns/GHz/radians. time_base interprets all tick fields; grids combine
    trigger and integer sample offsets before one ns conversion.
    Return owned arrays with sorted distinct content and a point-to-content map.
    Wrong ADC generator routes and empty trigger axes raise ValueError.
    With allow_receiver_only, no active drive.gen_ch pulse anywhere in the template
    yields empty pulses with endpoints at the first trigger for each point.
    Otherwise, missing pulses, varying tone count, pulse mode with multiple tones,
    or mux with more than eight tones or different starts raise ValueError.
    This function detaches the selected terminal block and grids without pairing.
    Core template/window arrays must satisfy their documented shapes and units.
    No physical model, density propagation or sampling is constructed.
    """
    slots = tuple(e for e in template.events if e.gen_ch == drive.gen_ch)
    count = (
        np.sum([e.active for e in slots], axis=0)
        if slots
        else np.zeros(len(template.points), dtype=np.int64)
    )
    if allow_receiver_only and not np.any(count):
        slots = ()
        tones = 0
    else:
        if np.any(count == 0) or (drive.kind == "pulse" and np.any(count != 1)):
            raise ValueError("requires one terminal readout pulse per point")
        tones = int(count[0])
        if tones > 8 or np.any(count != tones):
            raise ValueError("mux requires a fixed 1..8 tone count per point")
    if not window.trigger_ticks.shape[1] or np.any(
        window.generator_channels != drive.gen_ch
    ):
        raise ValueError("requires ADC timing and its bound readout generator")
    values, envelopes = _terminal_pulse_rows(
        slots, len(template.points), tones, time_base
    )
    end_ticks = (
        values[:, 0, 0].astype(np.int64) if tones else window.trigger_ticks[:, 0]
    )
    representatives, inverse = _content_rows(values, window)
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
        time_base.ns(end_ticks),
        tuple(pulses),
        tuple(grids),
        inverse.astype(np.int64),
        (tuple(range(window.trigger_ticks.shape[1])),) * len(grids),
    )


def _terminal_pulse_rows(
    slots: tuple[EventSlot, ...], points: int, tones: int, time_base: TimeBase
) -> tuple[NDArray[np.float64], list[PulseEnvelope | None]]:
    """Pack active terminal tones per point, retaining their common-origin phases."""
    fields = ("start_ticks", "duration_ticks", "frequency_mhz", "phase_deg", "gain")
    values = np.zeros((points, tones, 6))
    rows = np.arange(points)
    ranks = np.cumsum([e.active for e in slots], axis=0) - 1
    envelopes: list[PulseEnvelope | None] = []
    for event, rank in zip(slots, ranks, strict=True):
        for col, name in enumerate(fields):
            values[rows, rank, col] += np.where(event.active, getattr(event, name), 0)
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
    return values, envelopes


def _content_rows(
    pulses: NDArray[np.float64], window: ReadoutSlot
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Deduplicate rows by pulse, timing, and complete oscillator contents."""
    key = (
        pulses.reshape(len(pulses), -1),
        window.trigger_ticks,
        window.frequency_mhz,
        window.phase_deg,
    )
    _, representatives, inverse = np.unique(
        np.column_stack(key), axis=0, return_index=True, return_inverse=True
    )
    return representatives, inverse


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
