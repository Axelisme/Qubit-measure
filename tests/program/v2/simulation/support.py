"""Typed completed-acquisition observations shared by timeline tests."""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.core.simulation import AcquireRequest
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.acquisition import ImproveAcquireMixin


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
