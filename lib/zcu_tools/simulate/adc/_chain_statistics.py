"""Flow-count sampling and category-conditioned sums for projection chains."""

import numpy as np
from numpy.typing import NDArray

from .histograms import HistogramBins, cumulative_count_bins
from .model import RealArray
from .projection import ProjectionChain


def chain_counts(
    chain: ProjectionChain, reps: int, rng: np.random.Generator
) -> HistogramBins:
    """Draw event occupancy histograms without enumerating shots or histories."""
    counts = rng.multinomial(reps, chain.initial)
    histograms = [counts]
    for transition in chain.transitions:
        following = np.zeros_like(counts)
        for source in range(counts.shape[-1]):
            following += rng.multinomial(
                counts[..., source], transition[..., source, :]
            )
        histograms.append(following)
        counts = following
    return cumulative_count_bins(tuple(histograms))


def chain_conditional(
    values: RealArray, events: NDArray[np.int64], bins: HistogramBins
) -> RealArray:
    """Select each output read's state-conditioned means on the common bins.

    values is (*points, reads, states, 2), events is (*points, reads) in the
    same output-column order. bins.indices is in logical event order.
    """
    states = np.take_along_axis(
        np.stack(bins.indices, axis=-2), events[..., :, None], axis=-2
    )
    return np.take_along_axis(values, states[..., None], axis=-2)
