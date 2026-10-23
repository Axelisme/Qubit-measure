"""Immutable numeric acquisition boundary shared by frontend and environments.

Times are us, frequencies MHz, phases degrees, gain normalized DAC amplitude.
Execution is C-order over loop_dims (including reps); read order is trigger order.
Arrays in RoundSignals are float64 ADC sample units, last axis [I, Q]. Backend
sums are not normalized or rounded. No driver, Module or callback crosses here.
The optional round cancel query is read-only control, not a callback.
Frozen records do not freeze borrowed arrays. SamplingIdentity validates entropy;
AcquireRequest validates template coordinate names. Frontend and adapter checks
validate the remaining field ranges, layouts and contents against the request.
"""

from dataclasses import dataclass, field
from math import prod
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

AcquisitionMode = Literal["integrated", "decimated"]


class CancellationQuery(Protocol):
    """Read-only cancellation query shared by acquisition and numeric workers.

    threading.Event and program cancel flags satisfy this interface directly.
    The owner may set cancellation concurrently; consumers only query it.
    """

    def is_set(self) -> bool:
        """Return whether the owner has requested cancellation."""
        ...


@dataclass(frozen=True)
class SamplingIdentity:
    """Explicit deterministic shot/noise stream identity, without global RNG.

    seed: nonnegative integer root entropy.
    stream: tuple of nonnegative integer identity components chosen by the
        environment (e.g. acquire,round,execution,read). Equal full identity and
        equal calls reproduce arrays; changing identity selects a new stream.
        NumPy SeedSequence((seed,*stream)) with PCG64 defines this implementation's
        realization; cross-NumPy-version bitwise replay is not promised here.
    No acquire/round identity is guessed, and nothing is persisted by this tool.
    """

    seed: int
    stream: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if any(type(x) is not int or x < 0 for x in (self.seed, *self.stream)):
            raise ValueError(
                "sampling identity components must be nonnegative integers"
            )


@dataclass(frozen=True)
class PulseEnvelope:
    """Owned immutable signed complex envelope, continuously linearly interpolated.

    Attributes:
        sample_us: Strictly increasing local times in us from pulse start. These
            are the actual waveform sample axis, not solver integration steps.
        values: Dimensionless complex DAC amplitudes at sample_us, preserving
            original scale and sign. No peak normalization is applied. Interpolate
            real and imaginary parts linearly; hold the final sample through pulse
            end and use zero outside the event. Gain is applied separately.
    """

    sample_us: tuple[float, ...]
    values: tuple[complex, ...]


@dataclass(frozen=True)
class PulseEvent:
    """One generator pulse in a single execution's numeric timeline.

    Attributes:
        gen_ch: Zero-based generator index in the bound board configuration.
        start_us: Pulse start in us from this execution's common time origin,
            also used by ReadoutWindow.trigger_us. Not a wall-clock timestamp.
        duration_us: Pulse duration in us, quantized on the generator clock by
            the frontend; independent of the ADC duration.
        frequency_mhz: Generator frequency in MHz after QICK DDS quantization,
            matched to the readout channel by the current frontend.
        phase_deg: Generator phase in degrees after QICK phase-word conversion.
            This record imposes no additional wrapping or range check.
        gain: Signed normalized DAC amplitude after gain-word quantization,
            relative to the generator's maximum gain word. Nominal full scale
            is -1 to 1; this record does not enforce that range.
        envelope: Owned immutable continuous complex amplitude, or None for
            unity throughout duration_us. Never includes gain or carrier phase.
        phase_reference_us: Common carrier phase origin in execution us. Phase at
            time t is phase_deg + 360 * frequency_mhz * (t-phase_reference_us).
            Pulses do not independently reset their oscillator phase.
        source: Originating Module name for diagnostics, detached from the Module.
        iterations: Enclosing semantic loop (name, zero-based index) coordinates.
    """

    gen_ch: int
    start_us: float
    duration_us: float
    frequency_mhz: float
    phase_deg: float
    gain: float
    envelope: PulseEnvelope | None = None
    phase_reference_us: float = 0.0
    source: str = ""
    iterations: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class ReadoutWindow:
    """One ADC's fixed sample grid and read triggers for each execution.

    Attributes:
        channel: Zero-based ADC/readout index in the bound board configuration.
        gen_ch: Zero-based generator index routed to this ADC. The current
            frontend uses the last PulseReadout route declared for this ADC.
        samples: Compiled integer sample count per trigger, shared by all reads
            on this ADC. It is a count, not a duration or an integration weight.
        sample_rate_mhz: ADC output sample rate in MHz, or samples per us.
            The sample period is 1 / sample_rate_mhz us.
        trigger_us: Read start times in us from the same per-execution origin
            as PulseEvent.start_us. Tuple order is trigger emission order,
            not a separate chronological sort. Read r uses trigger_us[r] and
            samples equally spaced samples on this ADC's grid.
        frequency_mhz: Demodulation frequencies in MHz, one per trigger. Empty
            for legacy shape-only windows; populated in every PointTimeline.
        phase_deg: Demodulation phases in degrees, one per trigger; empty for
            shape-only windows. The common phase origin is execution time zero.
        generator_channels: Generator index per trigger in the same order as
            trigger_us. Empty in shape-only records, whose gen_ch is the fallback.
            PointTimeline windows populate this even when all reads use one route.
            gen_ch retains the final route for the fixed ADC shape descriptor.
    """

    channel: int
    gen_ch: int
    samples: int
    sample_rate_mhz: float
    trigger_us: tuple[float, ...]
    frequency_mhz: tuple[float, ...] = ()
    phase_deg: tuple[float, ...] = ()
    generator_channels: tuple[int, ...] = ()

    @property
    def reads(self) -> int:
        """Return the number of reads per execution, len(trigger_us)."""
        return len(self.trigger_us)


@dataclass(frozen=True)
class PointTimeline:
    """Detached numeric controls for one execution with time origin zero.

    Attributes:
        coordinates: Quantized hardware-loop (name, zero-based index) pairs in
            outermost-first order, excluding reps. Control values in events are
            already rounded by the board's clocks/DDS; indices are not voltages.
        events: Generator events in semantic emission order, not time-sorted.
        readouts: ADC declaration order; each window retains trigger emission
            order and this execution's trigger/demod values.
        duration_us: End of the body in execution us, including blocking waits
            and pending channel activity but excluding inter-shot relax_delay.
    """

    coordinates: tuple[tuple[str, int], ...]
    events: tuple[PulseEvent, ...]
    readouts: tuple[ReadoutWindow, ...]
    duration_us: float


@dataclass(frozen=True)
class EventSlot:
    """One generator position shared by the points of a TimelineTemplate.

    All arrays have shape (points,) in template.points order. Inactive entries
    retain finite placeholders. Arrays are borrowed, not frozen by this record.

    gen_ch: Fixed zero-based generator index.
    source: Originating Module name, for diagnostics only.
    iterations: Fixed enclosing Repeat (name, zero-based iteration) pairs.
    start_us: Start relative to the execution origin.
    duration_us: Generator-clock quantized duration.
    frequency_mhz: DDS-quantized carrier frequency.
    phase_deg: Phase-word phase; carrier phase at time t is phase_deg plus
        360 * frequency_mhz * (t - phase_reference_us).
    gain: Signed normalized DAC amplitude, nominal full scale [-1, 1].
    envelopes: Candidate envelopes; None means unity. Plain pulses have one
        candidate; computed pulses include every selected candidate.
    envelope_index: int64 candidate index for each point.
    active: bool mask; False omits this event at that point.
    phase_reference_us: Fixed common oscillator origin; current frontends use 0.
    """

    gen_ch: int
    source: str
    iterations: tuple[tuple[str, int], ...]
    start_us: NDArray[np.float64]
    duration_us: NDArray[np.float64]
    frequency_mhz: NDArray[np.float64]
    phase_deg: NDArray[np.float64]
    gain: NDArray[np.float64]
    envelopes: tuple[PulseEnvelope | None, ...]
    envelope_index: NDArray[np.int64]
    active: NDArray[np.bool_]
    phase_reference_us: float = 0.0


@dataclass(frozen=True)
class ReadoutSlot:
    """One ADC's trigger layout in template.points order.

    Arrays are borrowed and have shape (points, reads); reads is the fixed
    compiled trigger count of this ADC, in trigger emission order.

    channel: Fixed zero-based ADC index.
    samples: Fixed compiled samples per trigger.
    sample_rate_mhz: Fixed ADC output rate, in samples/us.
    trigger_us: Trigger start relative to execution zero.
    frequency_mhz: Demodulation frequency per trigger.
    phase_deg: Demodulation phase per trigger, relative to execution zero.
    generator_channels: int64 generator route per trigger.
    """

    channel: int
    samples: int
    sample_rate_mhz: float
    trigger_us: NDArray[np.float64]
    frequency_mhz: NDArray[np.float64]
    phase_deg: NDArray[np.float64]
    generator_channels: NDArray[np.int64]


@dataclass(frozen=True)
class TimelineTemplate:
    """One structural class of points with borrowed parameter arrays.

    points: Sorted int64 flat indices. A request's templates partition
        range(prod(point_dims)), with point axes flattened in C order.
    events: Generator slots in semantic emission order, not time-sorted.
    readouts: One slot per declared ADC, in declaration order.
    duration_us: float64 (points,) body end times, excluding relax_delay.
    """

    points: NDArray[np.int64]
    events: tuple[EventSlot, ...]
    readouts: tuple[ReadoutSlot, ...]
    duration_us: NDArray[np.float64]


@dataclass(frozen=True)
class AcquireRequest:
    """Fixed compiled controls/layout and private sampling identity for one call.

    Attributes:
        mode: "integrated" requests unnormalized ADC sample sums; "decimated"
            requests individual ADC samples. Neither mode asks for averaging.
        loop_dims: Compiled loop lengths, including reps, ordered outermost to
            innermost. Executions flatten these axes in C order, with the last
            loop varying fastest. Their product is the decimated execution count.
        avg_level: Zero-based axis of loop_dims that QICK averages for integrated
            acquisition, normally the reps axis. It does not request backend
            averaging. Decimated acquisition retains executions and averages
            only software rounds in the existing acquisition pipeline.
        readouts: One window per declared ADC, in QICK ADC declaration order,
            not sorted by channel number. Each window defines its read order.
        point_dims: Loop lengths without the reps axis, flattened in C order
            for template point indices. Empty means a single unswept point.
        point_names: Hardware loop names aligned with point_dims, excluding reps.
            Required when templates are supplied; length mismatch raises ValueError.
        reps: Length of the reps axis; repetitions do not create new points.
        templates: Structural point classes partitioning range(prod(point_dims)).
        relax_delay_us: Nonnegative inter-shot relaxation in us, added after
            every template point for pacing; not part of template body durations.
        root_seed: Nonnegative integer entropy supplied by the environment
            author or generated once by the binding when seed is omitted.
        sequence: Zero-based, monotonically increasing acquire identity from
            the binding. An admitted failed or cancelled call consumes it;
            rejected reentry does not. It is not a software round index.
        rng_version: Binding-defined RNG/seed-derivation version string,
            currently "PCG64/SeedSequence-v1/numpy-<version>". White noise uses
            SeedSequence([root_seed, sequence, round_index]) with PCG64.

    root_seed, sequence and rng_version are private replay metadata, not normal
    GUI/MCP metadata. Replay also requires the round snapshot and backend version;
    equal seeds alone do not promise bit-exact results across library versions.
    """

    mode: AcquisitionMode
    loop_dims: tuple[int, ...]
    avg_level: int
    readouts: tuple[ReadoutWindow, ...]
    root_seed: int
    sequence: int
    rng_version: str
    point_dims: tuple[int, ...] = ()
    point_names: tuple[str, ...] = ()
    reps: int = 1
    templates: tuple[TimelineTemplate, ...] = ()
    relax_delay_us: float = 0.0

    def __post_init__(self) -> None:
        if self.templates and len(self.point_names) != len(self.point_dims):
            raise ValueError("point_names must align with point_dims for templates")

    def materialize_point(self, point: int) -> PointTimeline:
        """Expand a flat C-order point into detached diagnostic controls.

        point must be in [0, prod(point_dims)); otherwise raise ValueError.
        Coordinates pair point_names with point-axis indices, excluding reps.
        Only active events are emitted, retaining semantic order. Arrays become
        immutable scalar/tuple values. Fixed ADC gen_ch comes from readouts.
        Raises ValueError if no template covers the requested point.
        """
        if not 0 <= point < prod(self.point_dims):
            raise ValueError("point is outside point_dims")
        for template in self.templates:
            local = int(np.searchsorted(template.points, point))
            if local < len(template.points) and template.points[local] == point:
                coordinates = tuple(
                    (name, int(index))
                    for name, index in zip(
                        self.point_names,
                        np.unravel_index(point, self.point_dims),
                        strict=True,
                    )
                )
                events = tuple(
                    PulseEvent(
                        gen_ch=slot.gen_ch,
                        source=slot.source,
                        iterations=slot.iterations,
                        start_us=float(slot.start_us[local]),
                        duration_us=float(slot.duration_us[local]),
                        frequency_mhz=float(slot.frequency_mhz[local]),
                        phase_deg=float(slot.phase_deg[local]),
                        gain=float(slot.gain[local]),
                        envelope=slot.envelopes[int(slot.envelope_index[local])],
                        phase_reference_us=slot.phase_reference_us,
                    )
                    for slot in template.events
                    if slot.active[local]
                )
                routes = {window.channel: window.gen_ch for window in self.readouts}
                readouts = tuple(
                    ReadoutWindow(
                        channel=slot.channel,
                        gen_ch=routes[slot.channel],
                        samples=slot.samples,
                        sample_rate_mhz=slot.sample_rate_mhz,
                        trigger_us=tuple(float(x) for x in slot.trigger_us[local]),
                        frequency_mhz=tuple(
                            float(x) for x in slot.frequency_mhz[local]
                        ),
                        phase_deg=tuple(float(x) for x in slot.phase_deg[local]),
                        generator_channels=tuple(
                            int(x) for x in slot.generator_channels[local]
                        ),
                    )
                    for slot in template.readouts
                )
                return PointTimeline(
                    coordinates, events, readouts, float(template.duration_us[local])
                )
        raise ValueError("no template covers point")


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Frozen numeric device values and resolvable model/calibration identity.

    Attributes:
        version: Environment-author-defined configuration/schema version string,
            for example "white_noise-v1". It is not the binding's backend or RNG
            version and has no shared format enforced by this record.
        model_identity: Environment-author-defined string identifying the model
            and calibration content for this round. For white noise it is
            "std_adc=<value>". It need not be a path, digest or public identifier.
        values: Tuple of (name, numeric value) pairs captured for this round.
            Names and native device units are defined by the environment author,
            not by this record. Empty means no numeric device inputs. Pair order
            is author-defined; the record does not sort or check unique names.

    The environment captures this snapshot immediately before round execution.
    Authors retain any referenced model/calibration content for replay; this
    record neither resolves references nor retains that content itself. All fields
    are private replay inputs, not normal GUI/MCP metadata or public model truth.
    """

    version: str
    model_identity: str
    values: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RoundRequest:
    """Private numeric inputs for one synchronous backend round.

    Attributes:
        acquire: The fixed AcquireRequest shared by every round of this call,
            including controls, layout, mode and root/acquire sampling identity.
        snapshot: Environment inputs captured immediately before this round,
            after any successful preceding round hook. Later device changes do
            not alter the snapshot or this acquire's controls/layout.
        index: Zero-based software round index within this acquire, not an
            execution/shot index. White noise combines it with acquire.root_seed
            and acquire.sequence to select this round's random stream.
        cancel: Optional read-only query checked between evolution chunks. None
            disables cancellation. Cancellation discards this round, not earlier
            completed rounds. It is excluded from equality and replay identity.
    """

    acquire: AcquireRequest
    snapshot: EnvironmentSnapshot
    index: int
    cancel: CancellationQuery | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ChannelSignals:
    """One ADC's unrounded float64 I/Q data from a joint round realization.

    Attributes:
        channel: Zero-based ADC/readout index matching a ReadoutWindow.channel
            in the request. It is not the position in the returned channel tuple.
        iq: Finite float64 array with last axis [I, Q]. Decimated mode uses
            (execution, read, sample, 2) and individual ADC
            sample units, with sample length equal to the window's samples.
            Executions follow loop_dims C order; reads follow trigger_us order;
            samples follow increasing time at 1 / sample_rate_mhz us intervals.
            No reps/rounds averaging or integer rounding has been applied.

    The backend owns iq and lends it until the adapter finishes copying the round.
    It must not mutate or reuse that storage during the borrow. A frozen record
    does not make the array read-only. The adapter checks shape and finiteness.
    """

    channel: int
    iq: NDArray[np.float64]


@dataclass(frozen=True)
class RoundSignals:
    """One joint decimated realization for every ADC in a software round.

    Attributes:
        channels: One ChannelSignals per requested ADC. Tuple order is arbitrary;
            the adapter restores AcquireRequest.readouts order by channel index.
            All entries belong to the same round realization. ADCs may have
            different sample counts and are not padded to a common length.

    ChannelSignals defines the decimated array layout, units and borrow lifetime.
    The backend lends arrays until the adapter copies them; repeated getters use
    the stored round buffers and do not execute the backend again. The adapter
    rejects missing, duplicate or extra ADCs, wrong shapes and nonfinite values.
    It copies decimated traces and sums those same samples for companion raw
    data, using nearest-even int64 rounding and rejecting overflow. QICK owns
    normalization and averaging afterward.
    """

    channels: tuple[ChannelSignals, ...]


@dataclass(frozen=True)
class ChannelStatistics:
    """Borrowed integrated round statistics of one ADC, in ADC sample-sum units.

    With P = point_dims, R = reads, S = readout states:

    channel: Zero-based ADC index, independent of tuple position.
    sums: float64 (*P, R, 2) totals over reps, without normalization. Includes
        receiver noise and the int64 rounding variance reps/12 per component.
    conditional_sums: float64 (*P, R, S, 2) noiseless per-shot sums conditioned
        on the initial readout state. State order matches state_counts.
    shot_covariance: float64 (2, 2) positive-semidefinite receiver-noise
        covariance of one sum, samples times the per-sample covariance.
        Zero and singular covariance are valid. It excludes rounding variance.

    Final axes use [I, Q]. Producers must not mutate arrays during the borrow.
    Constructors do not validate these layouts; the adapter checks
    correspondence to the acquire request. This is pure statistics data.
    """

    channel: int
    sums: NDArray[np.float64]
    conditional_sums: NDArray[np.float64]
    shot_covariance: NDArray[np.float64]


@dataclass(frozen=True)
class RoundStatistics:
    """Integrated-mode summary of one software round, with borrowed arrays.

    state_counts: int64 (*point_dims, states) nonnegative multinomial counts,
        summing to reps per point. Counts are shared by all ADCs and reads.
    channels: One ChannelStatistics per requested ADC, in any order.
    raw_identity: Informational local PCG64 identity for the environment's raw
        source. This record never generates shots. Producers use stream
        (acquire_sequence, round_index, 1), separate from counts/sums stream
        (acquire_sequence, round_index, 0). Drawing raw does not change either.

    Arrays must remain unchanged during the borrow. Adapter validation checks
    channels/shapes/finiteness against the request; this record has no request.
    """

    state_counts: NDArray[np.int64]
    channels: tuple[ChannelStatistics, ...]
    raw_identity: SamplingIdentity
