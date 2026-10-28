"""Shared physical path observations for integrated raw and batched traces."""

from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np

from ._integrated_paths import IntegratedPaths
from ._jump_paths import JumpPaths
from ._noise import noise_root
from .model import RealArray

if TYPE_CHECKING:
    from .raw import PreparedRawShots


def physical_shots(
    prepared: "PreparedRawShots",
    rng: np.random.Generator,
    *,
    traces: bool = False,
) -> dict[int, RealArray]:
    """Observe one outcome per shot, with an independent CTMC per read event.

    Return owned float64 (points,reps,reads[,samples],IQ) arrays. Within each
    event all ADCs and grids share initial states and paths; noise is independent
    per receiver/read/sample. Traces require equal sample counts per ADC.
    """
    readout = prepared.readout
    weights = readout.weights.reshape((-1, readout.weights.shape[-1]))
    batches, reps = prepared.batches, prepared.reps
    if weights.shape[-1] == 1:
        outcomes = np.zeros((len(weights), reps), dtype=np.int64)
    else:
        outcomes = np.sum(
            rng.random((len(weights), reps))[..., None]
            >= weights.cumsum(axis=-1)[:, None, :-1],
            axis=-1,
        )
    output = {}
    for channel, batch in batches:
        counts = tuple(
            len(grid.times_ns) for event in batch.events[0] for grid in event.grids
        )
        if traces and len(set(counts)) != 1:
            raise ValueError("trace reads of one ADC must have equal sample counts")
        shape = (len(weights), reps, len(counts))
        output[channel] = np.zeros((*shape, *((counts[0],) if traces else ()), 2))
    if not batches or not reps or not len(weights):
        return output
    # ADCs may deduplicate contents differently. Group by the complete tuple.
    keys = [
        tuple(int(batch.content_index[p]) for _, batch in batches)
        for p in range(len(weights))
    ]
    ordered = sorted(range(len(keys)), key=keys.__getitem__)
    offsets = dict.fromkeys((channel for channel, _ in batches), 0)
    for event_index, populations in enumerate(readout.populations):
        probabilities = populations.reshape((len(weights), weights.shape[-1], -1))
        selected = probabilities[np.arange(len(weights))[:, None], outcomes]
        states = np.sum(
            rng.random((len(weights), reps))[..., None]
            >= selected.cumsum(axis=-1)[..., :-1],
            axis=-1,
        )
        plans = {
            (channel, content): IntegratedPaths(
                batch.model,
                events[event_index].pulses,
                events[event_index].grids,
                prepared.rtol,
                prepared.atol,
            )
            for channel, batch in batches
            for content, events in enumerate(batch.events)
            if events[event_index].grids
        }
        for key, group in groupby(ordered, key=keys.__getitem__):
            points = np.array(list(group), dtype=np.int64)
            observers = [
                (channel, plans[channel, content])
                for (channel, _), content in zip(batches, key, strict=True)
                if (channel, content) in plans
            ]
            if not observers:
                continue
            paths = JumpPaths(
                states[points].ravel(),
                min(observer.start for _, observer in observers),
                max(observer.end for _, observer in observers),
                batches[0][1].model.rates_per_ns,
                rng,
            )
            for channel, observer in observers:
                values = np.stack(observer.evaluate(paths, traces=traces), axis=1)
                offset = offsets[channel]
                output[channel][points, :, offset : offset + values.shape[1]] = (
                    values.reshape((len(points), reps, *values.shape[1:]))
                )
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
