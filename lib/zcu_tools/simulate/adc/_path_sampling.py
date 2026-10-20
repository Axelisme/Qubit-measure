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
    """Observe validated physical inputs with one CTMC path per point/shot.

    prepared owns validated probabilities, physical batches, reps and tolerances.
    Return owned float64 arrays (points,reps,reads[,samples],IQ), adding noise
    independently for every receiver/read/sample. traces requires equal sample
    counts within each ADC. No noisy result or path survives this call.
    """
    probabilities = prepared.populations.reshape((-1, prepared.populations.shape[-1]))
    batches, reps = prepared.batches, prepared.reps
    uniforms = rng.random((len(probabilities), reps))
    states = np.sum(
        uniforms[..., None] >= probabilities.cumsum(axis=-1)[:, None, :-1], axis=-1
    )
    output = {}
    for channel, batch in batches:
        counts = tuple(len(grid.times_ns) for grid in batch.grids[0])
        if traces and len(set(counts)) != 1:
            raise ValueError("trace reads of one ADC must have equal sample counts")
        shape = (len(probabilities), reps, len(counts))
        output[channel] = np.zeros((*shape, *((counts[0],) if traces else ()), 2))
    if not batches or not reps or not len(probabilities):
        return output
    plans = {
        (channel, content): IntegratedPaths(
            batch.model, pulses, grids, prepared.rtol, prepared.atol
        )
        for channel, batch in batches
        for content, (pulses, grids) in enumerate(
            zip(batch.pulses, batch.grids, strict=True)
        )
    }
    # Different ADCs may deduplicate contents differently. Group points by the
    # full content tuple, never by a receiver's local content index alone.
    keys = [
        tuple(int(batch.content_index[p]) for _, batch in batches)
        for p in range(len(probabilities))
    ]
    ordered = sorted(range(len(keys)), key=keys.__getitem__)
    for key, group in groupby(ordered, key=keys.__getitem__):
        points = np.array(list(group), dtype=np.int64)
        observers = [
            plans[channel, content]
            for (channel, _), content in zip(batches, key, strict=True)
        ]
        start = min(observer.start for observer in observers)
        end = max(observer.end for observer in observers)
        paths = JumpPaths(
            states[points].ravel(), start, end, batches[0][1].model.rates_per_ns, rng
        )
        for (channel, _), observer in zip(batches, observers, strict=True):
            values = np.stack(observer.evaluate(paths, traces=traces), axis=1)
            output[channel][points] = values.reshape(
                (len(points), reps, *values.shape[1:])
            )
    for channel, batch in batches:
        root = noise_root(batch.model.receiver.noise_covariance)
        if np.any(root):
            shots = output[channel]
            noise = rng.standard_normal(shots.shape) @ root.T
            if not traces:
                counts = np.array([len(grid.times_ns) for grid in batch.grids[0]])
                noise *= np.sqrt(counts)[None, None, :, None]
            shots += noise
    return output
