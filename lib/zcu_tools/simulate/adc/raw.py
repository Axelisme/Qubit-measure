"""ADC-owned deferred integrated shots, independent of the program protocol."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.simulate.rounding import shot_layout

from ._path_sampling import physical_shots
from .model import RealArray, readonly_array
from .populations import validate_populations
from .readout_batch import ReadoutBatch


@dataclass(frozen=True, init=False)
class PreparedRawShots:
    """Retain noiseless inputs for deferred integrated ADC paths and noise.

    populations: finite nonnegative float64 (*point_dims,states) normalized to
        one within ADC's 1e-6 tolerance, in every batch model's state order.
        Copied read-only; this constructor validates rather than normalizes.
    batches: constructor mapping from unique nonnegative ADC indices to ReadoutBatch;
        retained as an immutable tuple of (channel,batch) pairs. Each
        batch.content_index is flat C-order (prod(point_dims),), and the state
        labels and directional rates must agree across receivers. Sample
        counts may differ by read, but must agree across contents of one ADC.
        Numeric records own immutable data; opaque envelopes remain
        caller-owned and must stay deterministic.
    reps: nonnegative integer shots per point, fixed for this preparation.
    rtol/atol: positive finite tolerances for noiseless arbitrary-envelope
        baseline solves; no shot LUT or jump-time interpolation is used.

    The preparation stores no identity, counts, shot paths, noisy IQ or RNG.
    bind returns a small per-round source; materialize alone draws shots. Each
    source keeps its preparation alive independently of any cache eviction or
    next round. Invalid shapes, states, channel indices, reps or tolerances raise
    ValueError. Numeric/solver failures propagate at materialization time.
    """

    populations: RealArray
    batches: tuple[tuple[int, ReadoutBatch], ...]
    reps: int
    rtol: float
    atol: float

    def __init__(
        self,
        populations: RealArray,
        batches: Mapping[int, ReadoutBatch],
        *,
        reps: int,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        probabilities = validate_populations(populations)
        if type(reps) is not int or reps < 0:
            raise ValueError("reps must be a nonnegative integer")
        if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
            raise ValueError("readout tolerances must be positive finite")
        labels = None
        rates = None
        for channel, batch in batches.items():
            if type(channel) is not int or channel < 0:
                raise ValueError("channel must be a nonnegative integer")
            if batch.content_index.shape != (prod(probabilities.shape[:-1]),):
                raise ValueError("batch point shape must match populations")
            if len(batch.model.state_labels) != probabilities.shape[-1]:
                raise ValueError("model state axis must match populations")
            if labels is not None and labels != batch.model.state_labels:
                raise ValueError("readout model state order must match across ADCs")
            if rates is not None and not np.array_equal(
                rates, batch.model.rates_per_ns
            ):
                raise ValueError("directional rates must match across ADCs")
            labels = batch.model.state_labels
            rates = batch.model.rates_per_ns
            counts = tuple(len(grid.times_ns) for grid in batch.grids[0])
            if any(
                tuple(len(grid.times_ns) for grid in grids) != counts
                for grids in batch.grids
            ):
                raise ValueError("sample counts must match across grid contents")
        object.__setattr__(
            self, "populations", readonly_array(probabilities, "populations")
        )
        object.__setattr__(self, "batches", tuple(batches.items()))
        object.__setattr__(self, "reps", reps)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "atol", atol)

    def bind(self, identity: SamplingIdentity) -> "ADCRawShots":
        """Return a deferred source using this validated local PCG64 identity.

        Binding performs no sampling or solver work. Equal inputs and identities
        reproduce shots within the same numeric/RNG version. No global RNG is
        used, and direct-mean/count streams are never consumed or adjusted.
        """
        return ADCRawShots(self, identity)


@dataclass(frozen=True)
class ADCRawShots:
    """One ADC round's raw source, structurally satisfying program RawShots.

    prepared: immutable noiseless probabilities and readout inputs, retained for
        this source's lifetime, not borrowed from a round's mutable statistics.
    identity: local reproducible PCG64 seed/stream, normally stream suffix 1.
        Repeated materialization restarts this stream, owns new arrays and never
        caches noisy results. This source does not import or call program code.
    """

    prepared: PreparedRawShots
    identity: SamplingIdentity

    def materialize(self, reps: int, avg_level: int) -> dict[int, NDArray[np.int64]]:
        """Own complete rounded ADC shots, shaped (*loop_dims,reads,2).

        reps must equal prepared.reps. avg_level inserts reps in the C-order
        point dimensions and must be in [0,len(point_dims)]. Invalid arguments,
        nonfinite receiver output or int64 overflow raise ValueError; baseline
        solver failures raise RuntimeError. Receiver noise is independent per
        ADC/read/shot. Initial states and full continuous-time jump paths are
        shared across every carrier, ADC and read, including all multi-jump
        tails and gaps. No raw is shifted to fit round sums.
        """
        prepared = self.prepared
        point_dims = prepared.populations.shape[:-1]
        if type(reps) is not int or reps != prepared.reps:
            raise ValueError("reps must match the raw preparation")
        if type(avg_level) is not int or not 0 <= avg_level <= len(point_dims):
            raise ValueError("avg_level is outside the point-axis layout")
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence((self.identity.seed, *self.identity.stream))
            )
        )
        shots = physical_shots(prepared, rng)
        return {
            channel: shot_layout(values, point_dims, reps, avg_level)
            for channel, values in shots.items()
        }
