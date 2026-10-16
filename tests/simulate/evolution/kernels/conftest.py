"""Small, phase-bearing batches for the weak constant-drive kernels."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import SegmentTable

from ..contract.kernel_suite import KernelCase, synthetic_model


@pytest.fixture(params=[3, 4])
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    name = request.node.module.KERNEL_NAME
    module = import_module(f"zcu_tools.simulate.evolution.kernels.{name}")
    algorithm = module.DysonKernel() if name == "dyson" else module.RWAKernel()
    model = synthetic_model(request.param)
    size = len(model.energies_ghz)
    vector = np.exp(1j * np.arange(size)) / np.sqrt(size)
    state = np.outer(vector, vector.conj())
    source = ModelSource(
        np.diag(np.arange(size, 0, -1) / sum(range(1, size + 1))),
        lambda level: replace(model, level=level),
    )
    durations = np.array([2.0, 3.0, 1.5, 0.0])
    gains = np.array([0.0004, 0.0006, -0.0003, 0.0])
    if name == "rwa":
        durations = np.array([20.0, 30.0, 15.0, 0.0])
        gains = np.array([0.003, 0.002, -0.004, 0.0])
    table = SegmentTable(
        np.arange(4),
        np.array([0.0, 2.0, 7.0, 1.0]),
        durations,
        np.zeros((4, 1), dtype=np.int64),
        np.ones(4, dtype=np.int64),
        np.zeros((4, 1), dtype=np.int64),
        np.array([[0.12], [0.15], [0.2], [0.13]]),
        gains[:, None],
        np.array([[0.0], [1.2], [-0.7], [2.1]]),
        np.zeros((4, 1), dtype=np.int64),
        np.zeros((4, 1)),
        (EnvelopeSpec("constant"),),
    )
    return KernelCase(algorithm, source, table, np.repeat(state[None], 4, axis=0))
