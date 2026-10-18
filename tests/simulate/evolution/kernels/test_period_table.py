"""Shaped single-tone channels at the period-table accuracy allowance."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource, SecularDrive
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    KernelContext,
    SegmentTable,
)
from zcu_tools.simulate.evolution.workplan import WorkPlan

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
    assert case.kernel.guard(
        case.table, rows, KernelContext(case.source, 0.001, WorkPlan())
    ).all()
    keys, representatives, inverse = np.unique(
        case.kernel.prepare_keys(
            case.table, rows, KernelContext(case.source, 0.001, WorkPlan())
        ),
        return_index=True,
        return_inverse=True,
    )
    prepared = case.kernel.prepare(
        keys,
        case.table,
        rows[representatives],
        KernelContext(case.source, 0.001, WorkPlan()),
    )
    assert prepared.nbytes > 0
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, case.table, rows, actual, ApplyContext(1e-3))
    expected = np.stack([reference(case, r, s) for r, s in enumerate(case.states)])
    assert_budget(actual, expected, 1e-2)


@pytest.mark.parametrize("prefix_input", [False, True], ids=["source", "prefix"])
def test_bind_input_contract(kernel_case: KernelCase, prefix_input: bool) -> None:
    KernelContract().test_bind_input_is_read_only_and_apply_equivalent(
        kernel_case, prefix_input
    )


@pytest.mark.parametrize("frequency", [0.0, 1e-20], ids=["dc", "near-dc"])
def test_ramp_inventory_has_positive_subdivisions_for_rejected_candidates(
    kernel_case: KernelCase, frequency: float
) -> None:
    from zcu_tools.simulate.evolution._carrier_plan import plan_ramp_families

    frequencies = kernel_case.table.tone_frequency_ghz.copy()
    frequencies[[5, 6], 0] = frequency
    table = replace(kernel_case.table, tone_frequency_ghz=frequencies)
    families = plan_ramp_families(table, kernel_case.source, WorkPlan())
    counts = {row: family.subdivisions for family in families for row in family.rows}

    assert set(counts) == set(range(len(table.tones)))
    assert all(np.isfinite(count) and count > 0 for count in counts.values())
    assert counts[5] > 32, "unresolved driven DC ramps cannot be admitted"
    assert counts[6] == 1, "zero gain needs no subperiod refinement, even at DC"


def test_shaped_sambe_cached_convergence_admission(kernel_case: KernelCase) -> None:
    from zcu_tools.simulate.evolution.kernels.shaped_sambe import ShapedSambeKernel

    solver = ShapedSambeKernel()
    case = kernel_case
    rows = np.array([0, 1, 4])
    context = KernelContext(case.source, case.budget, WorkPlan())
    built = solver.guard_prepared(case.table, rows, context)
    prepared = built.preparations[0].prepared
    admitted = solver.admit_prepared(prepared, case.table, rows[::-1], context)
    assert admitted is not None, "covered spectral periods must reuse convergence"
    np.testing.assert_array_equal(admitted.accepted, built.accepted[::-1])
    changed = replace(case.table, duration_ns=case.table.duration_ns + 1)
    assert solver.admit_prepared(prepared, changed, rows, context) is None
