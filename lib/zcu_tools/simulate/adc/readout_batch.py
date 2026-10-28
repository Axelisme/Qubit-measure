"""Batched readout inputs and all-channel round sampling contracts.

Readout uses the DC model at readout start. Flux tails and overlapping controls
are not inputs. Conditional means contain receiver calibration but no noise;
direct-mean sampling adds initial-state mixture, receiver noise and rounding
variance. Physical shot sampling lives in sampling and raw.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    ChannelStatistics,
    RoundStatistics,
    SamplingIdentity,
)

from ._noise import noise_root
from .model import ReadoutModel, RealArray, Receiver, readonly_array
from .projection import ProjectionReadout
from .request import ReadoutPulse, SampleGrid


def _indices(value: NDArray[np.int64], contents: int, name: str) -> NDArray[np.int64]:
    """Own read-only int64 content indices, rejecting rather than truncating."""
    indices = np.asarray(value)
    if indices.dtype != np.int64 or np.any(indices < 0) or np.any(indices >= contents):
        raise ValueError(f"{name} must be int64 content indices in [0, {contents})")
    result = indices.copy()
    result.setflags(write=False)
    return result


def _output_order(
    value: NDArray[np.int64] | None, contents: int, reads: int
) -> NDArray[np.int64]:
    """Own complete per-content permutations from event-major to output columns."""
    identity = np.arange(reads, dtype=np.int64)
    order = np.tile(identity, (contents, 1)) if value is None else np.asarray(value)
    if (
        order.dtype != np.int64
        or order.shape != (contents, reads)
        or np.any(np.sort(order, axis=1) != identity)
    ):
        raise ValueError("output_order must be int64 (contents, reads) permutations")
    result = order.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReadoutEvent:
    """One logical read event in physical ns/GHz units.

    pulses: nonempty tuple of coherent incident pulses sharing one CTMC path.
    grids: ADC windows in channel-local trigger order, sharing that path.
        Empty means this ADC does not observe the event and emits no read column.
        Windows may include lookback, gaps and ringdown.
    Empty pulses raise ValueError; physical records validate themselves.
    """

    pulses: tuple[ReadoutPulse, ...]
    grids: tuple[SampleGrid, ...]

    def __post_init__(self) -> None:
        if not self.pulses:
            raise ValueError("readout event requires nonempty pulses")


@dataclass(frozen=True)
class ReadoutBatch:
    """Distinct ordered read-event contents for many points of one ADC.

    model: DC model whose state order defines each event's initial-state axis.
    events: nonempty tuple of contents, each a nonempty ordered event tuple.
        Contents have identical event counts and grid counts per event. Grids
        within one event share a CTMC across ADCs; different events do not.
        Events may have zero grids, but each ADC content must observe at least
        one grid across its events. Responses concatenate only observed grids
        in event order, without re-numbering their logical event indices.
    content_index: owned/read-only int64 (points,) mapping points to contents.
    include_traces: request conditional per-sample traces as well as sums.
    output_order: optional int64 (contents, reads) complete permutations. Entry
        [c, r] selects the event-major grid column to emit at output column r.
        content_index selects each point's row. None selects identity event-major
        order. Explicit permutations are owned/read-only; None remains the
        identity policy when replacing a batch with a different grid layout.
        Events and conditional responses remain chronological; raw, integrated
        statistics and decimated outputs apply this order after physical work.
    Empty/misaligned contents or invalid indices raise ValueError.
    """

    model: ReadoutModel
    events: tuple[tuple[ReadoutEvent, ...], ...]
    content_index: NDArray[np.int64]
    include_traces: bool = False
    output_order: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        if not self.events or not self.events[0]:
            raise ValueError("readout contents and events must be nonempty")
        layout = tuple(len(event.grids) for event in self.events[0])
        if not any(layout):
            raise ValueError("each ADC content must have at least one observed grid")
        if any(
            tuple(len(event.grids) for event in events) != layout
            for events in self.events
        ):
            raise ValueError("contents must have the same event and grid counts")
        indices = _indices(self.content_index, len(self.events), "content_index")
        if indices.ndim != 1:
            raise ValueError("content_index must have shape (points,)")
        object.__setattr__(self, "content_index", indices)
        object.__setattr__(
            self,
            "output_order",
            None
            if self.output_order is None
            else _output_order(self.output_order, len(self.events), sum(layout)),
        )

    @property
    def column_order(self) -> NDArray[np.int64]:
        """Return read-only int64 (contents, reads) output-to-event-major indices.

        Resolve the identity policy to a complete permutation per content.
        Point rows are selected by content_index; physical events stay unchanged.
        """
        return _output_order(
            self.output_order,
            len(self.events),
            sum(len(e.grids) for e in self.events[0]),
        )


@dataclass(frozen=True)
class ConditionalSums:
    """Noiseless per-shot ADC means, conditional on the readout-start state.

    sums: owned/read-only finite float64 (contents, reads, states, 2), with
        positive content/read/state counts and final axis [I,Q]. Values are
        unnormalized discrete sample sums after gain, phase and offset, with
        no sample-period factor or receiver noise.
    sample_counts: owned/read-only positive int64 (reads,) sample counts, shared
        by every content. Counts may differ between reads in this record;
        round_statistics requires equal counts per channel
        because ChannelStatistics has one shot covariance per ADC.
    traces: optional owned/read-only finite float64
        (contents, reads, states, samples, 2) in ADC sample units, using the same
        calibration as sums. Dense traces require every sample_count to equal
        the samples axis. None means traces were not requested.
    event_indices: one nonnegative logical event index per read. Empty means
        all reads belong to event zero. Integrated sampling validates indices
        against ProjectionReadout and expands reads onto its joint count axis.
    output_order: optional int64 (contents, reads) complete permutations, with
        the same output-column-to-event-major-column direction as ReadoutBatch.
        None retains the identity policy; explicit permutations are owned and
        read-only. sums, traces, sample_counts and event_indices stay in
        event-major order. Integrated
        sampling applies this permutation after expanding the joint count axis.

    Invalid shapes, nonpositive/non-int64 sample counts or nonfinite values
    raise ValueError. Constructors do not recompute sums from optional traces.
    """

    sums: RealArray
    sample_counts: NDArray[np.int64]
    traces: RealArray | None = None
    event_indices: tuple[int, ...] = ()
    output_order: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        sums = readonly_array(self.sums, "conditional sums")
        if (
            sums.ndim != 4
            or sums.shape[-1] != 2
            or any(n == 0 for n in sums.shape[:-1])
        ):
            raise ValueError(
                "sums must have shape (contents, reads, states, 2) with positive axes"
            )
        counts = np.asarray(self.sample_counts)
        if (
            counts.dtype != np.int64
            or counts.shape != (sums.shape[1],)
            or np.any(counts <= 0)
        ):
            raise ValueError("sample_counts must be positive int64 with shape (reads,)")
        counts = counts.copy()
        counts.setflags(write=False)
        events = self.event_indices or (0,) * sums.shape[1]
        if len(events) != sums.shape[1] or any(
            type(index) is not int or index < 0 for index in events
        ):
            raise ValueError(
                "event_indices must contain one nonnegative integer per read"
            )
        object.__setattr__(self, "event_indices", events)
        object.__setattr__(self, "sums", sums)
        object.__setattr__(self, "sample_counts", counts)
        object.__setattr__(
            self,
            "output_order",
            None
            if self.output_order is None
            else _output_order(self.output_order, *sums.shape[:2]),
        )
        if self.traces is not None:
            traces = readonly_array(self.traces, "conditional traces")
            if (
                traces.ndim != 5
                or traces.shape[:3] != sums.shape[:3]
                or traces.shape[-1] != 2
                or np.any(counts != traces.shape[-2])
            ):
                raise ValueError("traces must align with sums and sample_counts")
            object.__setattr__(self, "traces", traces)

    @property
    def column_order(self) -> NDArray[np.int64]:
        """Return read-only int64 (contents, reads) output-to-event-major indices.

        Resolve identity without reordering sums, traces or event_indices.
        ChannelReadout.content_index selects each point's permutation.
        """
        return _output_order(self.output_order, *self.sums.shape[:2])


@dataclass(frozen=True)
class ChannelReadout:
    """Conditional inputs of one ADC for a joint round draw.

    channel: nonnegative integer ADC index, not a tuple position.
    conditional: noiseless sums and optional traces in the shared readout-state
        order. It owns distinct content responses, not per-point duplicates.
    content_index: owned/read-only int64 (*point_dims,) mapping each point to a
        conditional content. Scalar indices represent a single unswept point.
    receiver: per-sample noise covariance. Conditional means already include
        gain/phase/offset; sampling must not apply those calibrations again.

    Invalid channel or out-of-range/non-int64 content indices raise ValueError.
    Population shape/state order correspondence is checked by the draw function.
    """

    channel: int
    conditional: ConditionalSums
    content_index: NDArray[np.int64]
    receiver: Receiver

    def __post_init__(self) -> None:
        if type(self.channel) is not int or self.channel < 0:
            raise ValueError("channel must be a nonnegative integer")
        object.__setattr__(
            self,
            "content_index",
            _indices(
                self.content_index, self.conditional.sums.shape[0], "content_index"
            ),
        )


def round_statistics(
    readout: ProjectionReadout,
    channels: tuple[ChannelReadout, ...],
    *,
    reps: int,
    seed: int,
    acquire_sequence: int,
    round_index: int,
) -> RoundStatistics:
    """Draw integrated statistics for all ADCs from one joint state-count draw.

    readout: immutable outcome weights and event-specific populations. State
        order must match every channel's conditional state axis. Probabilities
        are validated by ProjectionReadout, never normalized by sampling.
    channels: unique ADCs; content_index shape must equal point_dims and each
        conditional state axis must match readout populations. Reads of one ADC must
        have equal sample_counts; ADCs may have different sample counts.
    reps: nonnegative integer shots per point; zero produces zero counts/sums.
    seed/acquire_sequence/round_index: nonnegative integers selecting the local
        SamplingIdentity(seed, (acquire_sequence, round_index, 0)) PCG64 stream.
        One stream serves every point and ADC without changing global RNG state.

    counts[p] is Multinomial(reps, weights[k] * terminal_populations[k,s]),
    with (k,s) flattened in C order and shared across ADCs/reads. Earlier reads'
    conditional means average their event populations for k, repeated across s;
    terminal reads depend on s, repeated across k. Each ADC/read sum is counts
    @ those conditional means plus independent Gaussian
    noise with covariance reps * (sample_count * receiver.noise_covariance
    + I/12). The I/12 term models nearest-even int64 rounding of individual shots.
    Returned conditional_sums have shape (*point_dims, reads, outcomes*states, 2);
    shot_covariance excludes rounding. raw_identity uses the same seed with
    stream (acquire_sequence, round_index, 1), isolating deferred raw draws.
    Inputs remain unchanged and equal identities reproduce results within the
    same RNG/numeric version. No jump-time trajectory variance is modeled.

    Raise ValueError for invalid probabilities, reps/identity, duplicate ADCs,
    incompatible point/state shapes or unequal read sample counts.
    """
    return PreparedStatistics(readout, channels, reps=reps).draw(
        seed=seed, acquire_sequence=acquire_sequence, round_index=round_index
    )


@dataclass(frozen=True)
class _StatisticsChannel:
    channel: int
    conditional: RealArray
    covariance: RealArray
    noise_root: RealArray


@dataclass(frozen=True, init=False)
class PreparedStatistics:
    """Owned immutable integrated sampling inputs; no seed, counts or noisy IQ.

    Constructor inputs and errors match round_statistics, including probability,
    reps and channel layout validation. Copies probabilities and expands owned
    conditional means once. Each channel's covariance is factored once, without
    changing draw order or arithmetic. populations is read-only float64
    (*point_dims, outcomes*states) in C-order (outcome, terminal state) order;
    reps is the validated shots per point. draw
    validates a fresh identity on every call and returns independently owned
    counts/noisy totals, so previous rounds and delayed raw remain usable.
    """

    populations: RealArray
    reps: int
    _channels: tuple[_StatisticsChannel, ...]

    def __init__(
        self,
        readout: ProjectionReadout,
        channels: tuple[ChannelReadout, ...],
        *,
        reps: int,
    ) -> None:
        probabilities = _validate_round_inputs(readout, channels, reps)
        object.__setattr__(
            self, "populations", readonly_array(probabilities, "populations")
        )
        object.__setattr__(self, "reps", reps)
        prepared = []
        for channel in channels:
            conditional = _joint_conditional(readout, channel)
            covariance = (
                int(channel.conditional.sample_counts[0])
                * channel.receiver.noise_covariance
            )
            root = noise_root(reps * (covariance + np.eye(2) / 12))
            prepared.append(
                _StatisticsChannel(
                    channel.channel,
                    readonly_array(conditional, "conditional means"),
                    readonly_array(covariance, "shot covariance"),
                    readonly_array(root, "noise root"),
                )
            )
        object.__setattr__(self, "_channels", tuple(prepared))

    def draw(
        self, *, seed: int, acquire_sequence: int, round_index: int
    ) -> RoundStatistics:
        """Draw a joint multinomial followed by ADC-ordered Gaussian noise.

        Identity parameters are nonnegative integers with the same PCG64 stream
        policy and errors as round_statistics. No global RNG state is changed.
        """
        identity = SamplingIdentity(seed, (acquire_sequence, round_index, 0))
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence((identity.seed, *identity.stream)))
        )
        counts = rng.multinomial(self.reps, self.populations)
        output = []
        for channel in self._channels:
            means = np.einsum("...s,...rsi->...ri", counts, channel.conditional)
            noise = rng.standard_normal(means.shape) @ channel.noise_root.T
            output.append(
                ChannelStatistics(
                    channel.channel,
                    means + noise,
                    channel.conditional,
                    channel.covariance,
                )
            )
        return RoundStatistics(
            counts,
            tuple(output),
            SamplingIdentity(seed, (acquire_sequence, round_index, 1)),
        )


def _joint_conditional(
    readout: ProjectionReadout, channel: ChannelReadout
) -> RealArray:
    """Expand each read onto the (outcome, final state) count axis."""
    values = channel.conditional.sums[channel.content_index]
    outcomes, states = readout.populations[-1].shape[-2:]
    reads = []
    for index, event in enumerate(channel.conditional.event_indices):
        conditional = values[..., index, :, :]
        if event == len(readout.populations) - 1:
            expanded = np.broadcast_to(
                conditional[..., None, :, :],
                (*conditional.shape[:-2], outcomes, states, 2),
            )
        else:
            means = np.einsum(
                "...ks,...si->...ki", readout.populations[event], conditional
            )
            expanded = np.broadcast_to(
                means[..., :, None, :], (*means.shape[:-2], outcomes, states, 2)
            )
        reads.append(expanded.reshape((*expanded.shape[:-3], outcomes * states, 2)))
    expanded = np.stack(reads, axis=-3)
    order = channel.conditional.column_order
    return np.take_along_axis(
        expanded, order[channel.content_index][..., :, None, None], axis=-3
    )


def _validate_round_inputs(
    readout: ProjectionReadout, channels: tuple[ChannelReadout, ...], reps: int
) -> RealArray:
    """Validate the shared probability/layout contract without repairing inputs."""
    joint = readout.weights[..., :, None] * readout.populations[-1]
    probabilities = joint.reshape((*readout.weights.shape[:-1], -1))
    if type(reps) is not int or reps < 0:
        raise ValueError("reps must be a nonnegative integer")
    if len({channel.channel for channel in channels}) != len(channels):
        raise ValueError("duplicate channel in round readout")
    for channel in channels:
        if channel.content_index.shape != probabilities.shape[:-1]:
            raise ValueError("channel point shape must match populations")
        conditional = channel.conditional
        if max(conditional.event_indices) >= len(readout.populations):
            raise ValueError("conditional event index outside readout events")
        if conditional.sums.shape[2] != readout.populations[-1].shape[-1]:
            raise ValueError("conditional state axis must match populations")
        if np.any(conditional.sample_counts != conditional.sample_counts[0]):
            raise ValueError("reads of one channel must have equal sample counts")
    return probabilities
