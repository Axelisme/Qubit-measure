"""Shaped single-tone channels at the period-table accuracy allowance."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource, SecularDrive
from zcu_tools.simulate.evolution.kernel import SegmentTable

from ..contract.kernel_suite import (
    KernelCase,
    KernelContract,
    assert_budget,
    reference,
    synthetic_model,
)


@pytest.fixture(params=[3, 4])
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    from zcu_tools.simulate.evolution.kernels.period_table import PeriodTableKernel

    size = int(request.param)
    model = synthetic_model(size)
    model = replace(
        model,
        # Rescale time by 1/8 and every frequency/rate by 8 to retain
        # dimensionless dynamics and period counts with fewer envelope knots.
        energies_ghz=model.energies_ghz * 8,
        rates_per_ns=model.rates_per_ns * 0.05 * 8,
        dephasing_per_ns=model.dephasing_per_ns * 8,
        drives=tuple(
            SecularDrive(f"port-{p}", d.in_phase_ghz * scale, d.quadrature_ghz * scale)
            for p, scale in enumerate((0.096, 0.072))
            for d in model.drives
        ),
    )
    times = np.linspace(0, 6, 13)
    gaussian = np.exp(-0.5 * ((times - 3) / 1.5) ** 2)
    drag = gaussian * (1 - 0.4j * (times - 3) / 18)
    ramp = np.minimum(times / 3, 1)
    envelopes = tuple(
        EnvelopeSpec("sampled", values.astype(complex), times)
        for values in (gaussian, drag, ramp)
    )
    # Amplitude Rabi, length Rabi, DRAG, ramp, reversed carrier, DC,
    # and zero drive. Nonzero local offsets model cut flat-top ramps.
    gains = np.array([0.3, 0.75, -0.6, 0.7, 0.7, 0.8, 0.0, 1.5])
    count = len(gains)
    table = SegmentTable(
        np.arange(count),
        np.arange(count, dtype=np.float64) * 1.3,
        # Preserve full periods, fractional tails, DC, zero drive and band > 1.
        np.array([48, 48, 35.7, 48, 19.3, 8.3, 9.1, 24.1]) / 8,
        np.array([[0], [0], [1], [0], [0], [1], [0], [0]]),
        np.ones(count, dtype=np.int64),
        np.array([[0], [0], [0], [1], [0], [1], [0], [0]]),
        np.array([[4.0], [4.0], [4.0], [4.0], [-4.0], [0], [4.0], [4.0]]),
        gains[:, None],
        np.array([[0.17], [1.3], [-0.7], [2.1], [0.5], [-1.1], [0], [0.9]]),
        np.array([[0], [0], [0], [1], [2], [1], [0], [2]]),
        np.array([[0], [0], [3.2], [0], [4.7], [0], [0], [0]]) / 8,
        envelopes,
    )
    vector = np.exp(0.7j * np.arange(size)) / np.sqrt(size)
    state = np.outer(vector, vector.conj())
    source = ModelSource(state, lambda level: replace(model, level=level))
    return KernelCase(
        PeriodTableKernel(), source, table, np.repeat(state[None], count, axis=0), 1e-2
    )


def test_guard_is_row_local(kernel_case: KernelCase) -> None:
    KernelContract().test_guard_is_row_local(kernel_case)


def test_rows_can_be_split(kernel_case: KernelCase) -> None:
    KernelContract().test_rows_can_be_split(kernel_case)


def test_prepare_is_density_independent(kernel_case: KernelCase) -> None:
    KernelContract().test_prepare_is_density_independent(kernel_case)


def test_shaped_accuracy(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(len(case.states), dtype=np.int64)
    assert case.kernel.guard(case.table, rows, case.source, 1e-3).all()
    keys, representatives, inverse = np.unique(
        case.kernel.prepare_keys(case.table, rows),
        return_index=True,
        return_inverse=True,
    )
    prepared = case.kernel.prepare(
        keys, case.table, rows[representatives], case.source, 1e-3
    )
    assert prepared.nbytes > 0
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, case.table, rows, actual, 1e-3)
    expected = np.stack([reference(case, r, s) for r, s in enumerate(case.states)])
    assert_budget(actual, expected, 1e-2)


def test_keys_share_shapes_phases_and_ports_but_separate_bands(
    kernel_case: KernelCase,
) -> None:
    case = kernel_case
    rows = np.arange(len(case.states), dtype=np.int64)
    table = replace(
        case.table,
        level=np.zeros_like(case.table.level),
        tone_frequency_ghz=np.full_like(case.table.tone_frequency_ghz, 0.5),
        tone_amplitude=np.array([0.6, -0.8, 0.6, 0.7, 0.7, 0.8, 0.0, 1.5])[:, None],
    )
    keys = case.kernel.prepare_keys(table, rows)
    assert len(np.unique(keys[:6])) == 1
    assert keys[0] != keys[7]
    assert keys[0] != keys[6]
    assert np.isfinite(case.kernel.cost(table, rows)).all()
