"""Dyson truncation eligibility must respect the segment's density budget."""

import numpy as np
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
)
from zcu_tools.simulate.evolution.kernel import ApplyContext, SegmentTable
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


def test_detuned_third_order_error_rejects_ratio_above_budget_limit():
    from scipy.linalg import expm

    x = np.zeros((3, 3), complex)
    x[0, 1] = x[1, 0] = 0.5
    x[1, 2] = x[2, 1] = 0.35
    y = 1j * np.triu(x) - 1j * np.tril(x)
    energies = np.array([0.0, 0.1, 0.23])
    model = SecularModel(
        (0,),
        "detuned-ladder",
        energies,
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
        np.array([5.0]),
        np.array([[0]]),
        np.array([1]),
        np.array([[0]]),
        np.array([[0.0]]),
        np.array([[0.01]]),
        np.array([[0.0]]),
        np.array([[0]]),
        np.array([[0.0]]),
        (EnvelopeSpec("constant"),),
    )
    kernel = DysonKernel()
    rows = np.array([0])
    keys = kernel.prepare_keys(table, rows)
    prepared = kernel.prepare(keys, table, rows, source, 1e-3)
    actual = initial[None].copy()
    kernel.apply(prepared, np.array([0]), table, rows, actual, ApplyContext(1e-3))
    # At f=0 the full Hamiltonian is constant, giving an exact reference.
    unitary = expm(-2j * np.pi * (np.diag(energies) + 0.01 * x) * 5)
    expected = unitary @ initial @ unitary.conj().T
    assert 1e-3 < np.max(abs(actual[0] - expected)) < 1.2e-3
    assert not kernel.guard(table, rows, source, 1e-3)[0]
