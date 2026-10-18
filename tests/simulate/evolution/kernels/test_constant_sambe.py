"""Cost-selected constant-tone preparation through the planner/kernel seam."""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution._dispatch import create_work_plan
from zcu_tools.simulate.evolution._gain_plan import period_builder
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel

from ..contract.kernel_suite import KernelCase, reference, synthetic_model


def sambe_case():
    model = synthetic_model()
    model = replace(
        model,
        energies_ghz=np.array([0.0, 4.0, 8.0]),
        rates_per_ns=np.zeros((3, 3)),
        dephasing_per_ns=np.zeros((3, 3)),
    )
    source = ModelSource(np.ones((3, 3), dtype=np.complex128) / 3, lambda level: model)
    table = SegmentTable(
        timeline=np.arange(2),
        start_ns=np.array([0.0, 13.0]),
        duration_ns=np.array([0.73, 21.7]),
        level=np.zeros((2, 1), dtype=np.int64),
        tones=np.ones(2, dtype=np.int64),
        tone_port=np.zeros((2, 1), dtype=np.int64),
        tone_frequency_ghz=np.array([[0.05], [-0.05]]),
        tone_amplitude=np.array([[0.01], [-0.02]]),
        tone_theta_rad=np.array([[1.73], [-5.9]]),
        tone_envelope=np.zeros((2, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((2, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    states = np.repeat(source.initial_density[None], 2, axis=0)
    return KernelCase(FloquetPeriodKernel(), source, table, states)


def test_low_carrier_uses_sambe_and_preserves_translated_coherent_tail():
    case = sambe_case()
    kernel = FloquetPeriodKernel()
    context = KernelContext(
        case.source, case.budget, create_work_plan(case.table, case.source, case.budget)
    )
    rows = np.arange(2)
    for row in rows:
        assert (
            period_builder(
                case.source.model((0,)),
                0,
                float(case.table.tone_frequency_ghz[row, 0]),
                float(case.table.tone_amplitude[row, 0]),
            ).kind
            == "sambe"
        )
    prepared = kernel.prepare(
        kernel.prepare_keys(case.table, rows, context),
        case.table,
        rows,
        context,
    )
    actual = case.states.copy()
    kernel.apply(prepared, rows, case.table, rows, actual, ApplyContext(case.budget))
    for row in rows:
        np.testing.assert_allclose(
            actual[row], reference(case, int(row), case.states[row]), atol=1e-4, rtol=0
        )
    assert prepared.nbytes > 0


def test_unresolved_cutoff_falls_back_to_independent_cf4():
    case = sambe_case()
    model = case.source.model((0,))
    diagonal = np.array([0.0, 1.0, 2.0])
    drive = replace(
        model.drives[0],
        in_phase_ghz=np.diag(diagonal).astype(complex),
        quadrature_ghz=np.zeros((3, 3), dtype=complex),
    )
    model = replace(model, drives=(drive,))
    source = ModelSource(case.source.initial_density, lambda level: model)
    table = replace(case.table, tone_amplitude=np.ones((2, 1)))
    rows = np.array([0])
    kernel = FloquetPeriodKernel()
    context = KernelContext(source, 0.001, create_work_plan(table, source, 0.001))
    assert period_builder(model, 0, 0.05, 1.0).kind == "sambe"
    prepared = kernel.prepare(
        kernel.prepare_keys(table, rows, context),
        table,
        rows,
        context,
    )
    actual = case.states[:1].copy()
    kernel.apply(prepared, np.array([0]), table, rows, actual, ApplyContext(1e-3))
    # Independent exact solution for commuting diagonal H(t), tighter than a
    # numerical strict reference and with no integration-step assumptions.
    duration = table.duration_ns[0]
    theta = table.tone_theta_rad[0, 0]
    omega = 2 * np.pi * table.tone_frequency_ghz[0, 0]
    integral = (
        model.energies_ghz * duration
        + diagonal * (np.sin(theta + omega * duration) - np.sin(theta)) / omega
    )
    unitary = np.diag(np.exp(-2j * np.pi * integral))
    expected = unitary @ case.states[0] @ unitary.conj().T
    np.testing.assert_allclose(actual[0], expected, atol=1e-4, rtol=0)


def test_high_carrier_keeps_existing_builder():
    case = sambe_case()
    for frequency, kind in ((0.582, "cf4"), (8.0, "dop853")):
        assert period_builder(case.source.model((0,)), 0, frequency, 0.01).kind == kind
        table = replace(case.table, tone_frequency_ghz=np.full((2, 1), frequency))
        kernel = FloquetPeriodKernel()
        context = KernelContext(
            case.source, case.budget, create_work_plan(table, case.source, case.budget)
        )
        rows = np.arange(2)
        prepared = kernel.prepare(
            kernel.prepare_keys(table, rows, context),
            table,
            rows,
            context,
        )
        actual = case.states.copy()
        kernel.apply(prepared, rows, table, rows, actual, ApplyContext(case.budget))
        high = replace(case, table=table)
        np.testing.assert_allclose(
            actual[0], reference(high, 0, case.states[0]), atol=1e-4, rtol=0
        )


def test_sambe_selection_and_playback_ignore_worker_and_key_order():
    from concurrent.futures import ThreadPoolExecutor

    case = sambe_case()
    rows = np.arange(2)
    kernel = FloquetPeriodKernel()
    context = KernelContext(
        case.source, case.budget, create_work_plan(case.table, case.source, case.budget)
    )
    prepared = kernel.prepare(
        kernel.prepare_keys(case.table, rows, context),
        case.table,
        rows,
        context,
    )
    expected = case.states.copy()
    kernel.apply(prepared, rows, case.table, rows, expected, ApplyContext(case.budget))
    other = FloquetPeriodKernel()
    other_context = KernelContext(
        case.source, case.budget, create_work_plan(case.table, case.source, case.budget)
    )
    actual = case.states.copy()

    def play(row):
        subset = np.array([row])
        payload = other.prepare(
            other.prepare_keys(case.table, subset, other_context),
            case.table,
            subset,
            other_context,
        )
        state = case.states[subset].copy()
        other.apply(
            payload, np.array([0]), case.table, subset, state, ApplyContext(case.budget)
        )
        actual[subset] = state

    # The main thread and a distinct worker prepare in reverse order.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(play, 1).result(timeout=25)
    play(0)
    np.testing.assert_array_equal(actual, expected)


def test_sambe_subperiods_preserve_secular_dissipation():
    case = sambe_case()
    bath = synthetic_model()
    model = replace(
        case.source.model((0,)),
        rates_per_ns=bath.rates_per_ns * 0.05,
        dephasing_per_ns=bath.dephasing_per_ns * 0.05,
    )
    source = ModelSource(case.source.initial_density, lambda level: model)
    table = replace(case.table, tone_amplitude=np.full((2, 1), 0.1))
    case = replace(case, source=source, table=table)
    kernel = FloquetPeriodKernel()
    context = KernelContext(source, 0.001, create_work_plan(table, source, 0.001))
    rows = np.arange(2)
    prepared = kernel.prepare(
        kernel.prepare_keys(table, rows, context),
        table,
        rows,
        context,
    )
    actual = case.states.copy()
    kernel.apply(prepared, rows, table, rows, actual, ApplyContext(case.budget))
    np.testing.assert_allclose(
        actual[1], reference(case, 1, case.states[1]), atol=case.budget, rtol=0
    )
