"""Immutable classical outcome weights and event-specific readout populations."""

from dataclasses import dataclass

import numpy as np

from .model import RealArray, readonly_array
from .populations import validate_populations


@dataclass(frozen=True)
class ProjectionReadout:
    """Conditional initial populations for an ordered sequence of read events.

    weights: finite nonnegative (*point_dims, outcomes) probabilities, with a
        nonempty outcome axis and rows summing to one within ADC's 1e-6 tolerance.
    populations: nonempty tuple in logical event order. Each array has shape
        (*point_dims, outcomes, states), with the same positive state count,
        nonnegative finite entries and normalized rows within that tolerance.
        Each event starts an independent CTMC; transitions during a read affect
        its IQ only, not populations of later events.

    Arrays are copied read-only. Invalid shapes or probabilities raise ValueError.
    Validation never clips or renormalizes; callers explicitly prepare evolution
    populations with normalize_populations before constructing this record.
    """

    weights: RealArray
    populations: tuple[RealArray, ...]

    def __post_init__(self) -> None:
        weights = readonly_array(validate_populations(self.weights), "outcome weights")
        if not self.populations:
            raise ValueError("projection readout requires at least one event")
        populations = tuple(
            readonly_array(validate_populations(values), "event populations")
            for values in self.populations
        )
        shape = (*weights.shape, populations[0].shape[-1])
        if any(values.shape != shape for values in populations):
            raise ValueError(
                "event populations must have shape (*points, outcomes, states)"
            )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "populations", populations)

    @classmethod
    def unprojected(cls, populations: RealArray) -> "ProjectionReadout":
        """Construct one event and one outcome from (*point_dims, states) rows.

        Preserve the caller's probabilities without normalization or extra RNG
        draws. Shape and probability errors are the same as the constructor.
        """
        values = validate_populations(populations)
        return cls(np.ones((*values.shape[:-1], 1)), (values[..., None, :],))
