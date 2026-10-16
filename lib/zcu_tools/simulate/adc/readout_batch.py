"""Batched readout inputs and all-channel round sampling contracts.

Readout uses the DC model at readout start. Flux tails and overlapping controls
are not inputs. Conditional means contain receiver calibration but no noise;
round sampling adds only initial-state mixture, receiver noise and rounding
variance, not jump-time trajectory variance.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import RoundSignals, RoundStatistics

from .model import ReadoutModel, RealArray, Receiver, readonly_array
from .request import ReadoutPulse, SampleGrid


def _indices(value: NDArray[np.int64], contents: int, name: str) -> NDArray[np.int64]:
    """Own read-only int64 content indices, rejecting rather than truncating."""
    indices = np.asarray(value)
    if indices.dtype != np.int64 or np.any(indices < 0) or np.any(indices >= contents):
        raise ValueError(f"{name} must be int64 content indices in [0, {contents})")
    result = indices.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReadoutBatch:
    """Distinct readout contents for many points of one ADC, in engine units.

    model: DC ReadoutModel; its state order defines the conditional-state axis.
        Frequencies are GHz, times ns and phases radians. No flux tail or
        overlapping pulse/flux correction is applied.
    pulses: one tuple per distinct content, each containing exactly one finite
        ReadoutPulse. Pulses are shared by all reads of that content.
    grids: one tuple of SampleGrid per content, in read order. Every content
        has the same positive number of reads. Grids retain physical timing
        and demodulation, including pre-pulse lookback.
    content_index: owned/read-only int64 (points,) indices into pulses/grids.
        Repeated indices reuse the same conditional response.
    include_traces: request conditional per-sample traces for decimated mode;
        False returns only sums, with ConditionalSums.traces=None.

    Empty/misaligned contents, reads or out-of-range/non-int64 indices raise
    ValueError. Pulse, grid and model records validate their own physical inputs.
    """

    model: ReadoutModel
    pulses: tuple[tuple[ReadoutPulse, ...], ...]
    grids: tuple[tuple[SampleGrid, ...], ...]
    content_index: NDArray[np.int64]
    include_traces: bool = False

    def __post_init__(self) -> None:
        if not self.pulses or len(self.grids) != len(self.pulses):
            raise ValueError("pulse and grid content counts must match and be nonempty")
        if any(len(pulses) != 1 for pulses in self.pulses):
            raise ValueError("each readout content must contain exactly one pulse")
        reads = len(self.grids[0])
        if not reads or any(len(grids) != reads for grids in self.grids):
            raise ValueError("grid contents must have the same positive read count")
        indices = _indices(self.content_index, len(self.pulses), "content_index")
        if indices.ndim != 1:
            raise ValueError("content_index must have shape (points,)")
        object.__setattr__(self, "content_index", indices)


@dataclass(frozen=True)
class ConditionalSums:
    """Noiseless per-shot ADC means, conditional on the readout-start state.

    sums: owned/read-only finite float64 (contents, reads, states, 2), with
        positive content/read/state counts and final axis [I,Q]. Values are
        unnormalized discrete sample sums after gain, phase and offset, with
        no sample-period factor or receiver noise.
    sample_counts: owned/read-only positive int64 (reads,) sample counts, shared
        by every content. Counts may differ between reads in this record;
        round_statistics and sample_traces require equal counts per channel
        because ChannelStatistics has one shot covariance per ADC.
    traces: optional owned/read-only finite float64
        (contents, reads, states, samples, 2) in ADC sample units, using the same
        calibration as sums. Dense traces require every sample_count to equal
        the samples axis. None means traces were not requested.

    Invalid shapes, nonpositive/non-int64 sample counts or nonfinite values
    raise ValueError. Constructors do not recompute sums from optional traces.
    """

    sums: RealArray
    sample_counts: NDArray[np.int64]
    traces: RealArray | None = None

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
        object.__setattr__(self, "sums", sums)
        object.__setattr__(self, "sample_counts", counts)
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
    populations: NDArray[np.float64],
    channels: tuple[ChannelReadout, ...],
    *,
    reps: int,
    seed: int,
    acquire_sequence: int,
    round_index: int,
) -> RoundStatistics:
    """Draw integrated statistics for all ADCs from one joint state-count draw.

    populations: finite float64 (*point_dims, states), nonnegative rows summing
        to one within 1e-10, in conditional state order. No clipping or repair.
    channels: unique ADCs; content_index shape must equal point_dims and each
        conditional state axis must match populations. Reads of one ADC must
        have equal sample_counts; ADCs may have different sample counts.
    reps: nonnegative integer shots per point; zero produces zero counts/sums.
    seed/acquire_sequence/round_index: nonnegative integers selecting the local
        SamplingIdentity(seed, (acquire_sequence, round_index, 0)) PCG64 stream.
        One stream serves every point and ADC without changing global RNG state.

    counts[p] is Multinomial(reps, populations[p]), shared across ADCs/reads.
    Each ADC/read sum is counts @ conditional means plus independent Gaussian
    noise with covariance reps * (sample_count * receiver.noise_covariance
    + I/12). The I/12 term models nearest-even int64 rounding of individual shots.
    Returned conditional_sums are expanded to (*point_dims, reads, states, 2);
    shot_covariance excludes rounding. raw_identity uses the same seed with
    stream (acquire_sequence, round_index, 1), isolating deferred raw draws.
    Inputs remain unchanged and equal identities reproduce results within the
    same RNG/numeric version. No jump-time trajectory variance is modeled.

    Raise ValueError for invalid probabilities, reps/identity, duplicate ADCs,
    incompatible point/state shapes or unequal read sample counts. The batched
    sampler is not implemented yet and currently raises NotImplementedError.
    """
    raise NotImplementedError("batched round statistics are not implemented")


def sample_traces(
    populations: NDArray[np.float64],
    channels: tuple[ChannelReadout, ...],
    *,
    reps: int,
    seed: int,
    acquire_sequence: int,
    round_index: int,
) -> RoundSignals:
    """Draw vectorized decimated shots for every ADC, with shared initial states.

    Parameters, population/state ordering, identity rules and failures match
    round_statistics. Each conditional.traces must be present and reads of one
    ADC must have equal sample_counts. Each shot shares its initial state across
    ADCs/reads, but receiver noise is independent across ADCs/reads/samples/shots.
    Use one SamplingIdentity(seed, (acquire_sequence, round_index, 0)) stream.

    Return one ChannelSignals per ADC, with float64 iq shape
    (prod(point_dims) * reps, reads, samples, 2), flattening points in C order
    then shots. Traces already contain gain/phase/offset; only per-sample noise
    is added, with no integrated rounding term or normalization. No trajectory
    jump-time variance is modeled. Inputs and global RNG state remain unchanged.
    Missing traces also raise ValueError. The batched sampler is not implemented
    yet and currently raises NotImplementedError.
    """
    raise NotImplementedError("batched trace sampling is not implemented")
