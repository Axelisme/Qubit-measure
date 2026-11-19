"""Adapters for exercising a reducer with synthetic rounds."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.program.acquisition import ShotReducer

type _IQ = Mapping[int, NDArray[np.float64]]


@dataclass(frozen=True)
class SumRounds:
    """Add round data from a test calculation that expects no shot selection."""

    round_sums: Callable[[_IQ], NDArray[np.float64]]

    def start(self) -> NDArray[np.float64] | None:
        return None

    def add(
        self, state: NDArray[np.float64] | None, shots: _IQ, mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        assert mask.all()
        sums = self.round_sums(shots)
        return sums if state is None else state + sums

    def output(self, state: NDArray[np.float64] | None) -> NDArray[np.float64]:
        assert state is not None
        return state


def reduce_round[State](reducer: ShotReducer[State], shots: _IQ) -> NDArray[np.float64]:
    """Exercise a reducer on one synthetic round with all shots kept."""
    mask = np.ones(next(iter(shots.values())).shape[:2], dtype=np.bool_)
    return reducer.output(reducer.add(reducer.start(), shots, mask))
