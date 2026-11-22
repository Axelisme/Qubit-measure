"""ADC-owned deferred integrated shots, independent of the program protocol."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from threading import Lock

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.boundary import CancellationQuery, Cancelled, SamplingIdentity
from zcu_tools.simulate.rounding import shot_layout

from ._path_sampling import EventPlanMemo, physical_shots
from .projection import ProjectionChain, ProjectionReadout
from .readout_batch import ReadoutBatch


def _validate_event_layout(batch: ReadoutBatch, event_count: int) -> None:
    """Validate event correspondence and stable per-read sample counts."""
    if len(batch.events[0]) != event_count:
        raise ValueError("batch event count must match readout populations")
    counts = tuple(
        len(grid.times_ns) for event in batch.events[0] for grid in event.grids
    )
    if any(
        tuple(len(grid.times_ns) for event in events for grid in event.grids) != counts
        for events in batch.events
    ):
        raise ValueError("sample counts must match across grid contents")


@dataclass(frozen=True, init=False)
class PreparedRawShots:
    """Retain noiseless inputs for deferred integrated ADC paths and noise.

    readout: owned immutable ProjectionReadout or ProjectionChain. Its states
        use every batch model's identical state order. Each batch must have
        exactly the same number of logical events as the readout.
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

    The preparation stores no plans, identity, counts, shot paths, noisy IQ or RNG.
    bind returns a per-round source without plan reuse. Materialize alone draws
    shots. Each source keeps its preparation alive independently of cache eviction or
    next round. Invalid shapes, states, channel indices, reps or tolerances raise
    ValueError. Numeric/solver failures propagate at materialization time.
    """

    readout: ProjectionReadout | ProjectionChain
    batches: tuple[tuple[int, ReadoutBatch], ...]
    reps: int
    rtol: float
    atol: float

    def __init__(
        self,
        readout: ProjectionReadout | ProjectionChain,
        batches: Mapping[int, ReadoutBatch],
        *,
        reps: int,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        probabilities = (
            readout.initial
            if isinstance(readout, ProjectionChain)
            else readout.populations[-1]
        )
        point_dims = (
            probabilities.shape[:-1]
            if isinstance(readout, ProjectionChain)
            else readout.weights.shape[:-1]
        )
        event_count = (
            1 + len(readout.transitions)
            if isinstance(readout, ProjectionChain)
            else len(readout.populations)
        )
        if type(reps) is not int or reps < 0:
            raise ValueError("reps must be a nonnegative integer")
        if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
            raise ValueError("readout tolerances must be positive finite")
        labels = None
        rates = None
        for channel, batch in batches.items():
            if type(channel) is not int or channel < 0:
                raise ValueError("channel must be a nonnegative integer")
            if batch.content_index.shape != (prod(point_dims),):
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
            _validate_event_layout(batch, event_count)
        object.__setattr__(self, "readout", readout)
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


# ADR-0078: ADC owns batch identity and plan reuse; the acquire session owns retirement.
class RawPlanOwner:
    """Reuse deterministic integrated raw plans for one acquire's current batch.

    An acquire session creates one owner and serves integrated rounds through
    bind. It calls retire when the current batch is discarded or the acquire
    ends, independently of any source preparation cache.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._active: tuple[PreparedRawShots, EventPlanMemo] | None = None

    def bind(
        self, prepared: PreparedRawShots, identity: SamplingIdentity
    ) -> "ADCRawShots":
        """Bind validated inputs and a local RNG identity without sampling.

        Reuse the memo only for the identical prepared object. A different
        object retires the old generation before admitting a new one.
        """
        with self._lock:
            active = self._active
            if active is None or active[0] is not prepared:
                if active is not None:
                    active[1].retire()
                active = self._active = (prepared, EventPlanMemo())
        source = ADCRawShots(prepared, identity)
        # Keep the memo tied to this binding, not copied by dataclasses.replace.
        object.__setattr__(source, "_memo", active[1])
        return source

    def retire(self) -> None:
        """Drop the current plans; already delivered sources can still draw cold.

        Idempotent. Late builds cannot refill the retired generation, and a
        subsequent bind starts cold even for the same prepared object.
        """
        with self._lock:
            if self._active is not None:
                self._active[1].retire()
                self._active = None


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
    _memo: EventPlanMemo | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def materialize(self, reps: int, avg_level: int) -> dict[int, NDArray[np.int64]]:
        """Own complete rounded ADC shots, shaped (*loop_dims,reads,2).

        reps must equal prepared.reps. avg_level inserts reps in the C-order
        point dimensions and must be in [0,len(point_dims)]. Invalid arguments,
        nonfinite receiver output or int64 overflow raise ValueError; baseline
        solver failures raise RuntimeError. Receiver noise is independent per
        ADC/read/shot. ProjectionReadout shares one outcome across the shot;
        ProjectionChain draws successive states conditional on the previous
        projection state, not on the readout CTMC's final state. Within each event,
        initial states and full CTMC paths are shared across carriers, ADCs and
        grids, including multi-jump tails and gaps. Different events draw
        independent paths. No raw is shifted to fit round sums.
        """
        return self._materialize(reps, avg_level, cancel=None)

    def materialize_cancellable(
        self, reps: int, avg_level: int, *, cancel: CancellationQuery
    ) -> dict[int, NDArray[np.int64]]:
        """Produce the same shots as materialize, or raise Cancelled without data.

        reps and avg_level have the same layout contract as materialize. The
        read-only cancel query is borrowed for this call, not retained. A later
        call can regenerate the entire round with or without a new query.
        """
        return self._materialize(reps, avg_level, cancel=cancel)

    def _materialize(
        self, reps: int, avg_level: int, *, cancel: CancellationQuery | None
    ) -> dict[int, NDArray[np.int64]]:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        prepared = self.prepared
        point_dims = (
            prepared.readout.initial.shape[:-1]
            if isinstance(prepared.readout, ProjectionChain)
            else prepared.readout.weights.shape[:-1]
        )
        if type(reps) is not int or reps != prepared.reps:
            raise ValueError("reps must match the raw preparation")
        if type(avg_level) is not int or not 0 <= avg_level <= len(point_dims):
            raise ValueError("avg_level is outside the point-axis layout")
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence((self.identity.seed, *self.identity.stream))
            )
        )
        shots = physical_shots(prepared, rng, cancel=cancel, memo=self._memo)
        result = {
            channel: shot_layout(values, point_dims, reps, avg_level)
            for channel, values in shots.items()
        }
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        return result
