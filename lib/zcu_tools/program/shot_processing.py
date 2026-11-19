"""State and output validation for joint-ADC shot reduction."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

type ShotIQ = Mapping[int, NDArray[np.float64]]


# ADR-0074: one joint-ADC reducer owns the calculation, not acquisition state.
class ShotReducer[State](Protocol):
    """Reduce complete rounds of normalized IQ and publish per-point results.

    The framework owns the state for one acquisition. Reducers may mutate it
    in add, but do not retain it on the reducer itself. Inputs are borrowed.
    """

    def start(self, points: int) -> State:
        """Return fresh state for points, the product of all non-rep loop sizes."""
        ...

    def add(self, state: State, shots: ShotIQ, mask: NDArray[np.bool_]) -> State:
        """Consume one round and return the new state.

        shots maps visible ADCs to float64 (points, reps, reads, 2) IQ divided
        by integration length without offset subtraction. mask is bool
        (points, reps), True for kept shots, and is shared by all ADCs.
        """
        ...

    def output(self, state: State) -> NDArray[np.float64]:
        """Read state without changing it; return float64 (points, *axes).

        The first axis must match the input point count. All other axes must
        be positive and the shape must stay fixed for this acquisition.
        """
        ...


class RoundReduction(Protocol):
    """One acquisition's running reduction, independent of the state type."""

    def add(self, shots: ShotIQ, mask: NDArray[np.bool_]) -> None:
        """Accumulate one complete round of normalized visible shots and its mask."""
        ...

    def output(self) -> NDArray[np.float64]:
        """Detach the cumulative output as (*point_dims, *axes).

        Invalid dtype, point count or a changed output shape raises ValueError.
        """
        ...


def start_reduction[State](
    reducer: ShotReducer[State], point_dims: tuple[int, ...]
) -> RoundReduction:
    """Start a reducer for the acquisition's non-rep loop sizes in loop order."""
    return _RoundReduction(reducer, point_dims)


class _RoundReduction[State]:
    def __init__(
        self, reducer: ShotReducer[State], point_dims: tuple[int, ...]
    ) -> None:
        self._reducer = reducer
        self._point_dims = point_dims
        self._points = prod(point_dims)
        self._state = reducer.start(self._points)
        self._output_shape: tuple[int, ...] | None = None

    def add(self, shots: ShotIQ, mask: NDArray[np.bool_]) -> None:
        self._state = self._reducer.add(self._state, shots, mask)

    def output(self) -> NDArray[np.float64]:
        payload = _float64_output(self._reducer.output(self._state))
        expected = self._output_shape
        if (
            payload.ndim < 1
            or payload.shape[0] != self._points
            or any(size < 1 for size in payload.shape[1:])
            or (expected is not None and payload.shape != expected)
        ):
            raise ValueError(
                f"reducer expected shape {expected or f'({self._points}, *positive_axes)'}; "
                f"actual shape {payload.shape}"
            )
        self._output_shape = payload.shape
        return payload.reshape(self._point_dims + payload.shape[1:]).copy()


def _float64_output(payload: object) -> NDArray[np.float64]:
    if not isinstance(payload, np.ndarray) or payload.dtype != np.float64:
        raise ValueError(
            f"reducer expected float64 ndarray; actual {type(payload).__name__} "
            f"dtype={getattr(payload, 'dtype', None)}"
        )
    return payload


@dataclass(frozen=True)
class AdditiveShotReducer:
    """Accumulate fixed-shape float64 arrays from completed rounds.

    round_sums computes one (points, *axes) array from normalized shots and a
    keep mask. axes are the trailing dimension sizes, excluding points.
    """

    round_sums: Callable[[ShotIQ, NDArray[np.bool_]], NDArray[np.float64]]
    axes: tuple[int, ...]

    def start(self, points: int) -> NDArray[np.float64]:
        """Return fresh float64 zeros of shape (points, *axes)."""
        return np.zeros((points, *self.axes), dtype=np.float64)

    def add(
        self, state: NDArray[np.float64], shots: ShotIQ, mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Compute and accumulate this round's per-point sums."""
        sums = self.round_sums(shots, mask)
        if sums.shape != state.shape:
            raise ValueError(
                f"reducer expected shape {state.shape}; actual {sums.shape}"
            )
        state += sums
        return state

    def output(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the accumulated sums."""
        return state


def zero_rejected_iq(
    values: NDArray[np.float64], mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Zero the IQ of rejected whole shots, without mutating the input.

    Used by sums whose features other than the count vanish at zero IQ.
    Keep the original layout when all shots are kept.
    """
    if mask.all():
        return values
    return np.where(mask[..., None, None], values, 0.0)
