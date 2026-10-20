"""Typed completed-acquisition observations shared by timeline tests."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.core.simulation import AcquireRequest
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.acquisition import ImproveAcquireMixin


@dataclass(frozen=True)
class FixedRawShots:
    """Synthetic environment source with one literal per-point/read IQ sum.

    values maps ADC indices to float64 (*point_dims,reads,2) noiseless sums.
    Each materialization repeats these explicit inputs along reps at avg_level;
    it does not derive them from, or fit them to, program statistics.
    """

    values: dict[int, NDArray[np.float64]]

    def materialize(self, reps: int, avg_level: int) -> dict[int, NDArray[np.int64]]:
        """Return owned int64 shots by repeating literal sums in loop order."""
        return {
            channel: np.rint(
                np.repeat(np.expand_dims(value, avg_level), reps, axis=avg_level)
            ).astype(np.int64)
            for channel, value in self.values.items()
        }


def request_of(env: SimulationEnvironment) -> AcquireRequest:
    """Return the public retained request after successful synchronous acquisition."""
    request = env.soc.binding.last_request
    if request is None:
        raise AssertionError("completed acquisition must retain its numeric request")
    return request


def raw_of(prog: ImproveAcquireMixin) -> list[NDArray[np.int64]]:
    """Return retained last-round ADC sums, failing if acquisition left no data."""
    raw = prog.get_raw()
    if raw is None:
        raise AssertionError("completed acquisition must retain its raw ADC sums")
    return raw
