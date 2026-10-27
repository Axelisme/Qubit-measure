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


@pytest.mark.parametrize("family", ["period_table", "shaped_sambe"])
def test_fractional_tails_pack_only_playing_envelopes(
    kernel_case: KernelCase, family: str
) -> None:
    from unittest.mock import patch

    from zcu_tools.simulate.evolution.kernels.period_table import PeriodTableKernel
    from zcu_tools.simulate.evolution.kernels.shaped_sambe import ShapedSambeKernel

    case = kernel_case
    solver = PeriodTableKernel() if family == "period_table" else ShapedSambeKernel()
    # Multiple independent tails include DRAG, signed gain/carrier, a second
    # physical port and a nonzero envelope offset. Padding is not a playing ID.
    rows = np.array([2, 3, 4, 7], dtype=np.int64)
    table = replace(
        case.table,
        duration_ns=case.table.duration_ns + 0.031,
        tone_port=np.pad(case.table.tone_port, ((0, 0), (0, 1)), constant_values=-1),
        tone_frequency_ghz=np.pad(case.table.tone_frequency_ghz, ((0, 0), (0, 1))),
        tone_amplitude=np.pad(case.table.tone_amplitude, ((0, 0), (0, 1))),
        tone_theta_rad=np.pad(case.table.tone_theta_rad, ((0, 0), (0, 1))),
        tone_envelope=np.pad(
            case.table.tone_envelope, ((0, 0), (0, 1)), constant_values=-1
        ),
        tone_envelope_offset_ns=np.pad(
            case.table.tone_envelope_offset_ns, ((0, 0), (0, 1))
        ),
    )
    context = KernelContext(case.source, case.budget, WorkPlan())
    assert solver.guard(table, rows, context).all()
    keys, representatives, inverse = np.unique(
        solver.prepare_keys(table, rows, context),
        return_index=True,
        return_inverse=True,
    )
    prepared = solver.prepare(keys, table, rows[representatives], context)
    expected = case.states[rows].copy()
    solver.apply(prepared, inverse, table, rows, expected, ApplyContext(case.budget))

    # Put the actual envelopes at sparse, reversed inventory IDs. Extra entries
    # must not increase the amount of sample data copied by a fractional tail.
    unused = EnvelopeSpec(
        "sampled", np.ones(257, dtype=np.complex128), np.linspace(0, 10, 257)
    )
    inventory = (unused,) * 19 + table.envelopes[::-1] + (unused,) * 23
    ids = table.tone_envelope.copy()
    playing = ids >= 0
    ids[playing] = 19 + len(table.envelopes) - 1 - ids[playing]
    inflated = replace(table, envelopes=inventory, tone_envelope=ids)
    actual = case.states[rows].copy()
    with patch.object(np, "concatenate", wraps=np.concatenate) as concatenate:
        solver.apply(
            prepared, inverse, inflated, rows, actual, ApplyContext(case.budget)
        )

    assert actual.tobytes() == expected.tobytes()
    independent = case.states[rows].copy()
    for local in range(len(rows)):
        solver.apply(
            prepared,
            inverse[local : local + 1],
            inflated,
            rows[local : local + 1],
            independent[local : local + 1],
            ApplyContext(case.budget),
        )
    assert independent.tobytes() == expected.tobytes()
    sample_copies = [
        call.args[0]
        for call in concatenate.call_args_list
        if any(
            array is envelope.samples
            for array in call.args[0]
            for envelope in inflated.envelopes
        )
    ]
    assert sample_copies, "the high-carrier fractional tails must exercise packing"
    assert all(
        array is not unused.samples for arrays in sample_copies for array in arrays
    ), "tail packing must not copy unused program envelopes"
    # Each apply group here has just one physical envelope, even for many rows.
    assert all(len(arrays) == 1 for arrays in sample_copies)
