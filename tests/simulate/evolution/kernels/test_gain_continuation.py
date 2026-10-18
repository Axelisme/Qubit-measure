"""Calibrated gain reuse through planner records and kernel density playback."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution._dispatch import create_work_plan
from zcu_tools.simulate.evolution._gain_plan import (
    admit_shared_builds,
    gain_anchor_resource,
)
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel
from zcu_tools.simulate.evolution.workplan import WorkPlan

from ..contract.kernel_suite import synthetic_model


def scan(gains, *, frequency=0.05, port=0):
    count = len(gains)
    return SegmentTable(
        timeline=np.arange(count),
        start_ns=np.zeros(count),
        duration_ns=np.full(count, 23.7),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.ones(count, dtype=np.int64),
        tone_port=np.full((count, 1), port, dtype=np.int64),
        tone_frequency_ghz=np.full((count, 1), frequency),
        tone_amplitude=np.array(gains)[:, None],
        tone_theta_rad=np.full((count, 1), 1.73),
        tone_envelope=np.zeros((count, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((count, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )


@pytest.fixture
def source():
    model = synthetic_model()
    model = replace(
        model,
        energies_ghz=np.array([0.0, 0.8, 1.6]),
        rates_per_ns=model.rates_per_ns * 1e-3,
        dephasing_per_ns=model.dephasing_per_ns * 1e-3,
        drives=(model.drives[0], replace(model.drives[0], name="other")),
    )
    return ModelSource(np.ones((3, 3), dtype=np.complex128) / 3, lambda level: model)


def scoped(table, source, budget=1e-3):
    context = KernelContext(source, budget, create_work_plan(table, source))
    return FloquetPeriodKernel(), context


def anchors(context):
    return [
        gain_anchor_resource(context, group, gain)
        for group in context.work_plan.gain_groups
        for gain in sorted({a for b in group.builders for a in b.anchors})
    ]


def prepare(kernel, table, source, budget=1e-3, *, rows=None, context=None):
    if rows is None:
        rows = np.arange(len(table.timeline), dtype=np.int64)
    if context is None:
        context = KernelContext(source, budget, WorkPlan())
    return kernel.prepare(
        kernel.prepare_keys(table, rows, context), table, rows, context
    )


def playback(kernel, prepared, table, source, budget=1e-3, *, rows=None):
    if rows is None:
        rows = np.arange(len(table.timeline), dtype=np.int64)
    states = np.repeat(source.initial_density[None], len(rows), axis=0)
    kernel.apply(
        prepared, np.arange(len(rows)), table, rows, states, ApplyContext(budget)
    )
    return states


def test_gain_predictor_is_checked_and_matches_independent_playback(source):
    table = scan(np.arange(6) * 1e-5 + 0.003)
    kernel, context = scoped(table, source)
    rows = np.arange(6)
    (group,) = context.work_plan.gain_groups
    assert group.gains == tuple(table.tone_amplitude[:, 0])
    assert group.builders[1].anchors == (group.gains[0], group.gains[-1])
    # Prefix resolution replaces the full table with owned copies of its arrays.
    admit_shared_builds(context, replace(table), rows, np.empty(0, dtype=np.int64))
    handles = anchors(context)
    assert all(h.sealed and not h.released for h in handles)
    first_rows, last_rows = np.arange(3), np.arange(3, 6)
    first = prepare(kernel, table, source, rows=first_rows, context=context)
    assert any(h.remaining_consumers for h in handles)
    last = prepare(kernel, table, source, rows=last_rows, context=context)
    assert all(h.released and not h.remaining_consumers for h in handles)
    independent = prepare(FloquetPeriodKernel(), table, source)
    # Include translated phase, a cross-period tail, and coherent input.
    actual = np.concatenate(
        [
            playback(kernel, first, table, source, rows=first_rows),
            playback(kernel, last, table, source, rows=last_rows),
        ]
    )
    np.testing.assert_allclose(
        actual,
        playback(FloquetPeriodKernel(), independent, table, source),
        atol=1e-4,
        rtol=0,
    )
    # A new evolve does not inherit the previous call's retained tables.
    _, fresh = scoped(table, source)
    assert all(not h.actual_consumers for h in anchors(fresh))
    context.work_plan.close()
    fresh.work_plan.close()


@pytest.mark.parametrize("order", ["thread-handoff", "reverse"])
def test_scoped_preparation_is_bitwise_invariant_to_worker_and_key_order(source, order):
    table = scan(np.arange(8) * 1e-5 + 0.003)
    reference_kernel, reference_context = scoped(table, source)
    reference = playback(
        reference_kernel,
        prepare(reference_kernel, table, source, context=reference_context),
        table,
        source,
    )
    kernel, context = scoped(table, source)
    # Deferred timeline builds admit incrementally, even after an early handoff.
    admit_shared_builds(context, table, np.empty(0, dtype=np.int64), np.arange(8))
    handles = anchors(context)
    actual = np.empty_like(reference)

    def prepare_rows(rows):
        prepared = kernel.prepare(
            kernel.prepare_keys(table, rows, context),
            table,
            rows,
            context,
        )
        states = np.repeat(source.initial_density[None], len(rows), axis=0)
        kernel.apply(
            prepared, np.arange(len(rows)), table, rows, states, ApplyContext(1e-3)
        )
        actual[rows] = states

    if order == "thread-handoff":
        prepare_rows(np.arange(4))
        # A separate worker necessarily has a different identity from the
        # calling thread. Await it explicitly, without relying on timing.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(prepare_rows, np.arange(4, 8)).result(timeout=25)
    else:
        for row in range(7, -1, -1):
            prepare_rows(np.array([row]))
    np.testing.assert_array_equal(actual, reference)
    assert all(not h.sealed and not h.released for h in handles)
    assert all(h.actual_consumers == h.planned_consumers for h in handles)
    assert all(not h.remaining_consumers for h in handles)
    context.work_plan.close()
    assert all(h.released for h in handles)
    reference_context.work_plan.close()


@pytest.mark.parametrize(
    "gains",
    [
        [0.003, 0.0031, 0.0032, 0.0033],
        [0.003, 0.006, 0.009, 0.012],
    ],
    ids=["coarse-defect-rejection", "unequal-anchor-grids"],
)
def test_rejected_predictor_preserves_independent_result_bitwise(source, gains):
    table = scan(gains)
    kernel, context = scoped(table, source, 1e-10)
    actual = prepare(kernel, table, source, 1e-10, context=context)
    independent = prepare(FloquetPeriodKernel(), table, source, 1e-10)
    np.testing.assert_array_equal(
        playback(kernel, actual, table, source, 1e-10),
        playback(FloquetPeriodKernel(), independent, table, source, 1e-10),
    )


def test_reverse_handoff_releases_sealed_resources_and_repeated_keys_stay_bitwise(
    source,
):
    table = scan(np.arange(8) * 1e-5 + 0.003)
    kernel, context = scoped(table, source)
    # A key omitted by caller admission must use the independent fallback,
    # including while other admitted keys still retain sealed anchors.
    pending = np.array([0, 1, 2, 4, 5, 6, 7])
    admit_shared_builds(context, table, pending, np.empty(0, dtype=np.int64))
    handles = anchors(context)
    omitted_row = np.array([3])
    omitted = prepare(kernel, table, source, rows=omitted_row, context=context)
    independent = prepare(FloquetPeriodKernel(), table, source, rows=omitted_row)
    np.testing.assert_array_equal(
        playback(kernel, omitted, table, source, rows=omitted_row),
        playback(kernel, independent, table, source, rows=omitted_row),
    )
    last_row = np.array([7])
    first = prepare(kernel, table, source, rows=last_row, context=context)
    expected = playback(kernel, first, table, source, rows=last_row)
    for row in range(6, -1, -1):
        prepare(kernel, table, source, rows=np.array([row]), context=context)
    assert all(h.released for h in handles)
    repeated = prepare(kernel, table, source, rows=last_row, context=context)
    np.testing.assert_array_equal(
        playback(kernel, repeated, table, source, rows=last_row), expected
    )
    assert all(h.released for h in handles)
    context.work_plan.close()


@pytest.mark.parametrize("changed", ["carrier", "budget", "model", "port"])
def test_history_never_crosses_physical_identity_or_budget(source, changed):
    table = scan([0.003, 0.00301])
    kernel, context = scoped(table, source)
    prepare(kernel, table, source, context=context)
    target = scan([0.00302])
    budget = 1e-3
    if changed == "carrier":
        target = scan([0.00302], frequency=-0.05)
    elif changed == "port":
        target = scan([0.00302], port=1)
    elif changed == "budget":
        budget = 1e-4
    else:
        model = replace(source.model((0,)), identity="another-model")
        source = ModelSource(source.initial_density, lambda level: model)
    target_context = KernelContext(source, budget, context.work_plan)
    if changed == "budget":
        target_context = KernelContext(source, budget, WorkPlan())
    actual = prepare(kernel, target, source, budget, context=target_context)
    independent = prepare(FloquetPeriodKernel(), target, source, budget)
    np.testing.assert_array_equal(
        playback(kernel, actual, target, source, budget),
        playback(FloquetPeriodKernel(), independent, target, source, budget),
    )
