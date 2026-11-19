"""State and output validation for joint-ADC shot reduction."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
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

    def start(self) -> State:
        """Return fresh state before the first round."""
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


@dataclass(frozen=True)
class RoundReduction:
    """One acquisition's stateful callbacks, independent of the state type."""

    add: Callable[[ShotIQ, NDArray[np.bool_], tuple[int, ...]], None]
    output: Callable[[], NDArray[np.float64]]


def start_reduction[State](reducer: ShotReducer[State]) -> RoundReduction:
    """Start a reducer and bind its state to the acquisition's round callbacks."""
    state = reducer.start()
    output_shape: tuple[int, ...] | None = None
    point_dims: tuple[int, ...] = ()
    points = 0

    def add(shots: ShotIQ, mask: NDArray[np.bool_], dims: tuple[int, ...]) -> None:
        nonlocal state, point_dims, points
        state = reducer.add(state, shots, mask)
        point_dims = dims
        points = mask.shape[0]

    def output() -> NDArray[np.float64]:
        nonlocal output_shape
        payload = _float64_output(reducer.output(state))
        if (
            payload.ndim < 1
            or payload.shape[0] != points
            or any(size < 1 for size in payload.shape[1:])
            or (output_shape is not None and payload.shape != output_shape)
        ):
            raise ValueError(
                f"reducer expected shape {output_shape or f'({points}, *positive_axes)'}; "
                f"actual shape {payload.shape}"
            )
        output_shape = payload.shape
        return payload.reshape(point_dims + payload.shape[1:]).copy()

    return RoundReduction(add, output)


def _float64_output(payload: object) -> NDArray[np.float64]:
    if not isinstance(payload, np.ndarray) or payload.dtype != np.float64:
        raise ValueError(
            f"reducer expected float64 ndarray; actual {type(payload).__name__} "
            f"dtype={getattr(payload, 'dtype', None)}"
        )
    return payload


@dataclass(frozen=True)
class AdditiveShotReducer:
    """Sum one fixed-shape payload per round; the first payload sets the shape."""

    round_sums: Callable[[ShotIQ, NDArray[np.bool_]], NDArray[np.float64]]

    def start(self) -> NDArray[np.float64]:
        """Return an empty accumulator before round dimensions are known."""
        return np.empty(0, dtype=np.float64)

    def add(
        self, state: NDArray[np.float64], shots: ShotIQ, mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Compute and accumulate this round's per-point sums."""
        sums = self.round_sums(shots, mask)
        if state.size == 0:
            return sums
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
