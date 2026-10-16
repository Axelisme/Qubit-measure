"""Static secular propagation contract including phase-bearing states."""
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import ModelSource
from zcu_tools.simulate.evolution.kernel import SegmentTable

from ..contract.kernel_suite import KernelCase, KernelContract, synthetic_model


def idle_table(durations):
    n = len(durations)
    empty = np.empty((n, 0))
    return SegmentTable(np.arange(n), np.arange(n, dtype=float), np.asarray(durations), np.zeros((n, 1), dtype=np.int64), np.zeros(n, dtype=np.int64), empty.astype(np.int64), empty, empty, empty, empty.astype(np.int64), empty, ())


@pytest.fixture(params=[3, 4])
def kernel_case(request):
    from zcu_tools.simulate.evolution.kernels.t4 import T4
    model = synthetic_model(request.param)
    vector = np.exp(1j * np.arange(request.param)) / np.sqrt(request.param)
    state = np.outer(vector, vector.conj())
    source = ModelSource(state, lambda level: replace(model, level=level))
    return KernelCase(T4(), source, idle_table([0., .7, 1.3, .7]), np.repeat(state[None], 4, axis=0))


class TestT4(KernelContract):
    pass
