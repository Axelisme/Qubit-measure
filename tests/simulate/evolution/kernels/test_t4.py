"""Static secular propagation contract including phase-bearing states."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import ModelSource

from .._tables import idle_table
from ..contract.kernel_suite import KernelCase, KernelContract, synthetic_model


@pytest.fixture(params=[3, 4])
def kernel_case(request):
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    model = synthetic_model(request.param)
    vector = np.exp(1j * np.arange(request.param)) / np.sqrt(request.param)
    state = np.outer(vector, vector.conj())
    source = ModelSource(state, lambda level: replace(model, level=level))
    return KernelCase(
        T4(),
        source,
        idle_table([0.0, 0.7, 1.3, 0.7]),
        np.repeat(state[None], 4, axis=0),
    )


class TestT4(KernelContract):
    pass
