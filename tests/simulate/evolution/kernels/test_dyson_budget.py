"""Dyson truncation eligibility must respect the segment's density budget."""

import numpy as np
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
)
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels.dyson import DysonKernel


def test_resonant_third_order_coherence_error_exceeds_budget_and_is_rejected():
    x = np.zeros((3, 3), complex)
    x[0, 1] = x[1, 0] = 0.5
    y = np.zeros((3, 3), complex)
    y[0, 1], y[1, 0] = 0.5j, -0.5j
    model = SecularModel(
        (0,),
        "resonant-pair",
        np.array([0.0, 0.1, 0.3]),
        np.eye(3, dtype=complex),
        (SecularDrive("drive", x, y),),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    initial = np.diag([1.0, 0.0, 0.0]).astype(complex)
    source = ModelSource(initial, lambda level: model)
    table = SegmentTable(
        np.array([0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([[0]]),
        np.array([1]),
        np.array([[0]]),
        np.array([[0.1]]),
        np.array([[0.04]]),
        np.array([[0.0]]),
        np.array([[0]]),
        np.array([[0.0]]),
        (EnvelopeSpec("constant"),),
    )
    # Exact circular resonant drive: the uncoupled third state remains empty.
    # Dyson keeps coherence pi*g*t, while the exact value is sin(2*pi*g*t)/2.
    angle = np.pi * 0.04
    error = angle - np.sin(2 * angle) / 2
    assert 1.31e-3 < error < 1.33e-3
    kernel = DysonKernel()
    rows = np.array([0])
    assert not kernel.guard(table, rows, source, 1e-3)[0]
