"""Constant-tone strategy selection observed at the kernel endpoint seam."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution._t6_dispatch import publish_inventory
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.kernels import dop853
from zcu_tools.simulate.evolution.kernels._dop853_data import pack_segments
from zcu_tools.simulate.evolution.kernels.t6 import T6Kernel
from zcu_tools.simulate.evolution.workplan import WorkPlan

from ..contract.kernel_suite import KernelCase, reference, synthetic_model


def constant_case(count: int, frequency: float = 0.7) -> KernelCase:
    """Independent coherent inputs sharing a short constant-tone prepare key."""
    model = synthetic_model(3)
    vector = np.exp(1j * np.array([0.1, 0.8, -0.4])) / np.sqrt(3)
    density = np.outer(vector, vector.conj())
    source = ModelSource(density, lambda level: model)
    table = SegmentTable(
        timeline=np.arange(count),
        start_ns=np.arange(count, dtype=np.float64) * 0.37,
        # A short single use stays direct; many reused rows amortize a channel.
        duration_ns=np.full(count, 16.0),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.ones(count, dtype=np.int64),
        tone_port=np.zeros((count, 1), dtype=np.int64),
        tone_frequency_ghz=np.full((count, 1), frequency),
        tone_amplitude=np.full((count, 1), 0.15),
        tone_theta_rad=np.linspace(-0.3, 2.1, count)[:, None],
        tone_envelope=np.zeros((count, 1), dtype=np.int64),
        tone_envelope_offset_ns=np.zeros((count, 1)),
        envelopes=(EnvelopeSpec("constant"),),
    )
    return KernelCase(
        T6Kernel(), source, table, np.repeat(density[None], count, axis=0)
    )


@pytest.fixture
def endpoint_used(monkeypatch: pytest.MonkeyPatch) -> Callable[[], bool]:
    # Family diagnostics do not expose the strategy. Observe the named shared
    # endpoint seam, delegating unchanged; no payload or private RHS is inspected.
    called = False
    original = dop853.advance_endpoints

    def observe(
        data: dop853.EndpointData,
        row_models: tuple[NDArray[np.int64], NDArray[np.int64]],
        states: NDArray[np.complex128],
        budget: float,
        plan: dop853.DOP853Plan,
    ) -> None:
        nonlocal called
        called = True
        original(data, row_models, states, budget, plan)

    monkeypatch.setattr(dop853, "advance_endpoints", observe)
    return lambda: called


def apply_case(case: KernelCase, work_plan: WorkPlan) -> NDArray[np.complex128]:
    """Prepare all distinct keys and apply them through the Kernel protocol."""
    rows = np.arange(len(case.states), dtype=np.int64)
    context = KernelContext(case.source, case.budget, work_plan)
    assert case.kernel.guard(case.table, rows, context).all()
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(case.table, rows, context),
        return_index=True,
        return_inverse=True,
    )
    prepared = case.kernel.prepare(keys, case.table, rows[first], context)
    actual = case.states.copy()
    case.kernel.apply(
        prepared, inverse, case.table, rows, actual, ApplyContext(case.budget)
    )
    return actual


@pytest.mark.parametrize("frequency", [0.7, -0.7, 0.0])
@pytest.mark.parametrize("count", [1, 64], ids=["single-use-direct", "reused-channel"])
def test_key_reuse_selects_strategy_with_full_density_accuracy(
    endpoint_used: Callable[[], bool], count: int, frequency: float
) -> None:
    case = constant_case(count, frequency)
    actual = apply_case(case, WorkPlan())
    expected = np.stack(
        [reference(case, row, state) for row, state in enumerate(case.states)]
    )
    tolerance = case.budget if count == 1 else 2e-9
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    assert endpoint_used() is (count == 1)


@pytest.mark.parametrize("budget", [1e-3, 1e-5])
def test_direct_density_uses_shared_budget_policy(budget: float) -> None:
    case = replace(constant_case(1, -0.37), budget=budget)
    case = replace(
        case,
        table=replace(
            case.table,
            duration_ns=np.array([4.0]),
            tone_amplitude=np.array([[0.51]]),
        ),
    )
    shared = case.states.copy()
    dop853.advance_endpoints(
        dop853.EndpointData(
            dop853.prepare_models([case.source.model((0,))]).models,
            pack_segments(case.table),
        ),
        (np.array([0]), np.array([0])),
        shared,
        budget,
        dop853.DOP853Plan(np.array(["interaction"]), np.array([-1])),
    )
    # Single-use selection is covered separately. Here both interfaces must
    # honor the same budget policy, not merely fall below an accuracy ceiling.
    actual = apply_case(case, WorkPlan())
    expected = reference(case, 0, case.states[0])
    np.testing.assert_allclose(actual[0], expected, rtol=0, atol=budget)
    np.testing.assert_allclose(shared[0], expected, rtol=0, atol=budget)
    np.testing.assert_allclose(actual, shared, rtol=0, atol=5e-12)


def test_published_inventory_survives_singleton_worker_table(
    endpoint_used: Callable[[], bool],
) -> None:
    complete = constant_case(64)
    plan = WorkPlan()
    publish_inventory(complete.table, complete.source, plan)
    singleton = replace(constant_case(1), source=complete.source)
    actual = apply_case(singleton, plan)
    expected = reference(singleton, 0, singleton.states[0])
    np.testing.assert_allclose(actual[0], expected, rtol=0, atol=2e-9)
    assert not endpoint_used(), "a worker must borrow complete-acquire reuse"


@pytest.mark.parametrize("published", [False, True], ids=["standalone", "published"])
@pytest.mark.parametrize("zero_field", ["frequency", "amplitude", "duration"])
def test_signed_zero_representative_uses_complete_inventory(
    endpoint_used: Callable[[], bool], published: bool, zero_field: str
) -> None:
    case = constant_case(64, 0.0)
    frequency = case.table.tone_frequency_ghz.copy()
    amplitude = case.table.tone_amplitude.copy()
    duration = case.table.duration_ns.copy()
    if zero_field == "frequency":
        frequency[:, 0] = 0.0
        frequency[1, 0] = -0.0
    elif zero_field == "amplitude":
        amplitude[:, 0] = 0.0
        amplitude[1, 0] = -0.0
    else:
        duration[:] = 0.0
        duration[1] = -0.0
    case = replace(
        case,
        table=replace(
            case.table,
            tone_frequency_ghz=frequency,
            tone_amplitude=amplitude,
            duration_ns=duration,
        ),
    )
    plan = WorkPlan()
    if published:
        publish_inventory(case.table, case.source, plan)
    context = KernelContext(case.source, case.budget, plan)
    representative = np.array([1], dtype=np.int64)
    assert case.kernel.guard(case.table, representative, context).all()
    keys = case.kernel.prepare_keys(case.table, representative, context)
    prepared = case.kernel.prepare(keys, case.table, representative, context)
    rows = np.array([1, 0], dtype=np.int64)
    actual = case.states[rows].copy()
    case.kernel.apply(
        prepared,
        np.zeros(len(rows), dtype=np.int64),
        case.table,
        rows,
        actual,
        ApplyContext(case.budget),
    )
    expected = np.stack([reference(case, row, case.states[row]) for row in rows])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-9)
    # Reuse comes from all 64 rows, not the single prepare representative.
    # Zero-duration keys always select direct because they have no apply work.
    assert endpoint_used() is (zero_field == "duration")


def test_published_inventory_rejects_unknown_key() -> None:
    case = constant_case(1)
    plan = WorkPlan()
    publish_inventory(case.table, case.source, plan)
    changed = replace(case, table=replace(case.table, duration_ns=np.array([0.5])))
    with pytest.raises(ValueError, match="key absent from complete acquire inventory"):
        apply_case(changed, plan)


def test_mixed_strategies_keep_row_order_and_partition_independence() -> None:
    case = constant_case(65)
    durations = case.table.duration_ns.copy()
    durations[0] = 0.5
    case = replace(case, table=replace(case.table, duration_ns=durations))
    whole = apply_case(case, WorkPlan())
    rows = np.arange(len(case.states), dtype=np.int64)
    context = KernelContext(case.source, case.budget, WorkPlan())
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(case.table, rows, context),
        return_index=True,
        return_inverse=True,
    )
    prepared = case.kernel.prepare(keys, case.table, rows[first], context)
    split = case.states.copy()
    for selected in (rows[::2][::-1], rows[1::2][::-1]):
        chunk = split[selected].copy()
        case.kernel.apply(
            prepared,
            inverse[selected],
            case.table,
            selected,
            chunk,
            ApplyContext(case.budget),
        )
        split[selected] = chunk
    np.testing.assert_array_equal(split, whole)
    expected = np.stack(
        [reference(case, row, state) for row, state in enumerate(case.states)]
    )
    # Only row zero has a single-use key; all remaining rows reuse a channel.
    np.testing.assert_allclose(whole[:1], expected[:1], rtol=0, atol=case.budget)
    np.testing.assert_allclose(whole[1:], expected[1:], rtol=0, atol=2e-9)


def test_direct_numerical_failure_raises_without_repair() -> None:
    case = constant_case(1)
    extreme = replace(case.table, tone_amplitude=np.array([[1e308]]))
    with pytest.raises(RuntimeError, match="nonfinite"):
        apply_case(replace(case, table=extreme), WorkPlan())
