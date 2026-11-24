"""Shared physical path observations for integrated raw and batched traces."""

from collections.abc import Iterator
from itertools import groupby
from math import prod
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.boundary import CancellationQuery, Cancelled

from ._integrated_paths import IntegratedPaths
from ._jump_paths import JumpPaths
from ._noise import noise_root
from ._time import adc_to_device_time_ns
from .model import RealArray
from .projection import ProjectionChain, ProjectionReadout

if TYPE_CHECKING:
    from .raw import PreparedRawShots
    from .readout_batch import ReadoutBatch


type _EventPlans = dict[tuple[int, int], IntegratedPaths]


class EventPlanMemo:
    """Publish deterministic event plans and chain CDFs for one live batch."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: dict[int, _EventPlans] | None = {}
        self._chain_cdfs: dict[int, RealArray] | None = {}

    def retire(self) -> None:
        with self._lock:
            self._events = None
            self._chain_cdfs = None

    def plans(self, prepared: "PreparedRawShots", event_index: int) -> _EventPlans:
        with self._lock:
            if self._events is not None and event_index in self._events:
                return self._events[event_index]
        plans = _build_event_plans(prepared, event_index)
        with self._lock:
            # A late solve uses its own result without refilling a retired memo.
            if self._events is not None:
                return self._events.setdefault(event_index, plans)
        return plans

    def chain_cdf(self, event_index: int, probabilities: RealArray) -> RealArray | None:
        """Get the event's row-wise CDF, or None if already retired on lookup."""
        with self._lock:
            cdfs = self._chain_cdfs
            if cdfs is None:
                return None
            if (cdf := cdfs.get(event_index)) is None:
                cdf = cdfs[event_index] = probabilities.cumsum(axis=-1)
            return cdf


def physical_shots(
    prepared: "PreparedRawShots",
    rng: np.random.Generator,
    *,
    traces: bool = False,
    cancel: CancellationQuery | None = None,
    memo: EventPlanMemo | None = None,
) -> dict[int, RealArray]:
    """Observe one outcome per shot, with an independent CTMC per read event.

    Return owned float64 (points,reps,reads[,samples],IQ) arrays. Within each
    event all ADCs and grids share initial states and paths; noise is independent
    per receiver/read/sample. Traces require equal sample counts per ADC.
    For integrated raw, memo is the source's batch generation; traces omit it.
    Raise Cancelled at event or point-group checkpoints if cancel is set.
    """
    readout = prepared.readout
    point_dims = (
        readout.initial.shape[:-1]
        if isinstance(readout, ProjectionChain)
        else readout.weights.shape[:-1]
    )
    points_count = prod(point_dims)
    batches, reps = prepared.batches, prepared.reps
    output = {}
    for channel, batch in batches:
        counts = tuple(
            len(grid.times_ns) for event in batch.events[0] for grid in event.grids
        )
        if traces and len(set(counts)) != 1:
            raise ValueError("trace reads of one ADC must have equal sample counts")
        shape = (points_count, reps, len(counts))
        output[channel] = np.zeros((*shape, *((counts[0],) if traces else ()), 2))
        _fill_empty_events(batch, output[channel], traces=traces)
    if not batches or not reps or not points_count:
        return output
    # ADCs may deduplicate contents differently. Group by the complete tuple.
    keys = [
        tuple(int(batch.content_index[p]) for _, batch in batches)
        for p in range(points_count)
    ]
    ordered = sorted(range(len(keys)), key=keys.__getitem__)
    offsets = dict.fromkeys((channel for channel, _ in batches), 0)
    for event_index, states in enumerate(_event_states(readout, reps, rng, memo=memo)):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        plans = (
            _build_event_plans(prepared, event_index)
            if memo is None
            else memo.plans(prepared, event_index)
        )
        for key, group in groupby(ordered, key=keys.__getitem__):
            points = np.array(list(group), dtype=np.int64)
            observers = [
                (channel, plans[channel, content])
                for (channel, _), content in zip(batches, key, strict=True)
                if (channel, content) in plans
            ]
            if observers:
                # Receiver-only windows still bound the shared physical event.
                # Retuning one ADC must not shorten paths or shift receiver-noise draws.
                end = max(
                    max(observer.end for _, observer in observers),
                    max(
                        adc_to_device_time_ns(
                            float(grid.times_ns[-1]),
                            time_of_flight_ns=batch.model.receiver.time_of_flight_ns,
                        )
                        for (_, batch), content in zip(batches, key, strict=True)
                        for grid in batch.events[content][event_index].grids
                    ),
                )
                paths = JumpPaths(
                    states[points].ravel(),
                    min(observer.start for _, observer in observers),
                    end,
                    batches[0][1].model.rates_per_ns,
                    rng,
                )
                for channel, observer in observers:
                    values = np.stack(observer.evaluate(paths, traces=traces), axis=1)
                    offset = offsets[channel]
                    output[channel][points, :, offset : offset + values.shape[1]] = (
                        values.reshape((len(points), reps, *values.shape[1:]))
                    )
            if cancel is not None and cancel.is_set():
                raise Cancelled()
        for channel, batch in batches:
            offsets[channel] += len(batch.events[0][event_index].grids)
    _add_receiver_noise(prepared, output, rng, traces=traces)
    for channel, batch in batches:
        order = batch.column_order[batch.content_index][:, None, :]
        output[channel] = np.take_along_axis(
            output[channel],
            order[..., None, None] if traces else order[..., None],
            axis=2,
        )
    return output


def _build_event_plans(prepared: "PreparedRawShots", event_index: int) -> _EventPlans:
    """Prepare noiseless observers for one logical event, keyed by ADC and content."""
    return {
        (channel, content): IntegratedPaths(
            batch.model, event.pulses, event.grids, prepared.rtol, prepared.atol
        )
        for channel, batch in prepared.batches
        for content, events in enumerate(batch.events)
        if (event := events[event_index]).grids and event.pulses
    }


def _chain_event_states(
    readout: ProjectionChain,
    reps: int,
    rng: np.random.Generator,
    memo: EventPlanMemo | None,
) -> Iterator[NDArray[np.int64]]:
    """Draw successive projection states, reusing CDF rows only within the memo."""
    state_count = readout.initial.shape[-1]
    initial = readout.initial.reshape((-1, 1, state_count))
    shot_shape = (len(initial), reps)
    cdf = memo.chain_cdf(0, initial) if memo is not None else None
    if cdf is None:
        initial = np.broadcast_to(initial, (*shot_shape, state_count))
    states = np.sum(
        rng.random(shot_shape)[..., None]
        >= (initial.cumsum(axis=-1) if cdf is None else cdf)[..., :-1],
        axis=-1,
    )
    yield states
    points = np.arange(len(initial))[:, None]
    for event_index, transition in enumerate(readout.transitions, start=1):
        rows = transition.reshape((len(initial), state_count, state_count))
        cdf = memo.chain_cdf(event_index, rows) if memo is not None else None
        # The cached route cumulates before gathering; both use the same float64
        # addition order within the state axis. Cold sources only sum selected rows.
        selected = (
            rows[points, states].cumsum(axis=-1)[..., :-1]
            if cdf is None
            else cdf[points, states, :-1]
        )
        states = np.sum(rng.random(shot_shape)[..., None] >= selected, axis=-1)
        yield states


def _event_states(
    readout: ProjectionReadout | ProjectionChain,
    reps: int,
    rng: np.random.Generator,
    *,
    memo: EventPlanMemo | None = None,
) -> Iterator[NDArray[np.int64]]:
    """Yield projection states, interleaving each event draw with its CTMC draws."""
    if isinstance(readout, ProjectionChain):
        yield from _chain_event_states(readout, reps, rng, memo)
    else:
        weights = readout.weights.reshape((-1, readout.weights.shape[-1]))
        if weights.shape[-1] == 1:
            outcomes = np.zeros((len(weights), reps), dtype=np.int64)
        else:
            outcomes = np.sum(
                rng.random((len(weights), reps))[..., None]
                >= weights.cumsum(axis=-1)[:, None, :-1],
                axis=-1,
            )
        for populations in readout.populations:
            probabilities = populations.reshape((len(weights), weights.shape[-1], -1))
            selected = probabilities[np.arange(len(weights))[:, None], outcomes]
            yield np.sum(
                rng.random((len(weights), reps))[..., None]
                >= selected.cumsum(axis=-1)[..., :-1],
                axis=-1,
            )


def _fill_empty_events(
    batch: "ReadoutBatch", shots: RealArray, *, traces: bool
) -> None:
    """Fill receiver-only windows without constructing physical path observers."""
    for content, events in enumerate(batch.events):
        offset = 0
        for event in events:
            if not event.pulses:
                for index, grid in enumerate(event.grids):
                    shots[batch.content_index == content, :, offset + index] = (
                        batch.model.receiver.offset
                        * (1 if traces else len(grid.times_ns))
                    )
            offset += len(event.grids)


def _add_receiver_noise(
    prepared: "PreparedRawShots",
    output: dict[int, RealArray],
    rng: np.random.Generator,
    *,
    traces: bool,
) -> None:
    """Add independent receiver noise in ADC/read order, in place.

    Traces receive sample covariance; integrated values receive each window's
    sample-count-scaled covariance. Keep this phase after every event's paths so
    the unprojected stream retains its original state/path/noise draw order.
    """
    for channel, batch in prepared.batches:
        root = noise_root(batch.model.receiver.noise_covariance)
        if np.any(root):
            shots = output[channel]
            noise = rng.standard_normal(shots.shape) @ root.T
            if not traces:
                counts = np.array(
                    [
                        len(grid.times_ns)
                        for event in batch.events[0]
                        for grid in event.grids
                    ]
                )
                noise *= np.sqrt(counts)[None, None, :, None]
            shots += noise
