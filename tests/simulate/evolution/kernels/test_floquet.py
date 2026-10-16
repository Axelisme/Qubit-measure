"""Floquet's C5 seam: arbitrary densities, carrier phases and weak probes."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource, kernel

from ..contract.kernel_suite import (
    KernelCase,
    KernelContract,
    assert_budget,
    reference,
    synthetic_model,
)


@pytest.fixture(params=[3, 4])
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel

    size = int(request.param)
    model = synthetic_model(size)
    model = replace(
        model,
        rates_per_ns=model.rates_per_ns * 0.001,
        dephasing_per_ns=model.dephasing_per_ns * 0.001,
        drives=tuple(
            replace(d, in_phase_ghz=d.in_phase_ghz * 0.1,
                    quadrature_ghz=d.quadrature_ghz * 0.1)
            for d in model.drives
        ),
    )
    count = 4
    table = kernel.SegmentTable(
        timeline=np.arange(count),
        start_ns=np.array([0., 13., 2., 51.]),
        duration_ns=np.array([1.1, 1.7, 0., 2.1]),
        level=np.zeros((count, 1), dtype=np.int64),
        tones=np.full(count, 2),
        tone_port=np.zeros((count, 2), dtype=np.int64),
        tone_frequency_ghz=np.column_stack((np.full(count, 3.7), [.11, .23, .13, -.17])),
        tone_amplitude=np.column_stack((np.full(count, .1), [.02, -.03, .04, .01])),
        tone_theta_rad=np.array([[.7, -.3], [2., 1.4], [-1., 2.], [5., -.9]]),
        tone_envelope=np.zeros((count, 2), dtype=np.int64),
        tone_envelope_offset_ns=np.full((count, 2), 2.),
        envelopes=(EnvelopeSpec("constant"),),
    )
    vector = np.exp(1j * np.arange(size)) / np.sqrt(size)
    coherent = np.outer(vector, vector.conj())
    states = np.stack([coherent, np.eye(size) / size, coherent.conj(), coherent])
    source = ModelSource(states[0], lambda level: replace(model, level=level))
    return KernelCase(FloquetKernel(), source, table, states)


class TestFloquetContract(KernelContract):
    pass


def test_probe_parameters_share_prepare_but_stark_and_level_do_not(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    keys = case.kernel.prepare_keys(case.table, rows)
    assert len(np.unique(keys)) == 1
    for table in (
        replace(case.table, tone_frequency_ghz=case.table.tone_frequency_ghz + [1., 0.]),
        replace(case.table, tone_amplitude=case.table.tone_amplitude * [2., 1.]),
        replace(case.table, level=np.ones((4, 1), dtype=np.int64)),
    ):
        assert not np.any(case.kernel.prepare_keys(table, rows) == keys)
    costs = case.kernel.cost(case.table, rows)
    assert costs.shape == (4,)
    assert np.all(np.isfinite(costs) & (costs >= 0))


def test_tone_order_does_not_change_physical_result(kernel_case: KernelCase) -> None:
    case = kernel_case
    swapped = replace(
        case.table,
        **{field: getattr(case.table, field)[:, ::-1] for field in (
            "tone_port", "tone_frequency_ghz", "tone_amplitude", "tone_theta_rad",
            "tone_envelope", "tone_envelope_offset_ns",
        )},
    )
    rows = np.arange(4, dtype=np.int64)
    assert np.array_equal(case.kernel.prepare_keys(swapped, rows), case.kernel.prepare_keys(case.table, rows))
    KernelContract().test_error_within_budget(replace(case, table=swapped))


def test_nonzero_level_and_distinct_prepares(kernel_case: KernelCase) -> None:
    case = kernel_case
    levels = np.array([[0], [1], [0], [1]])
    original = case.source.model((0,))
    source = ModelSource(case.states[0], lambda level: replace(
        original, level=level, identity=f"level-{level}",
        energies_ghz=original.energies_ghz * (1 + .3 * level[0]),
    ))
    KernelContract().test_error_within_budget(replace(case, source=source, table=replace(case.table, level=levels)))


def test_full_micromotion_with_arbitrary_start_phase(kernel_case: KernelCase) -> None:
    case = kernel_case
    original = case.source.model((0,))
    source = ModelSource(case.states[0], lambda level: replace(
        original, level=level, rates_per_ns=np.zeros_like(original.rates_per_ns),
        dephasing_per_ns=np.zeros_like(original.dephasing_per_ns),
    ))
    table = replace(
        case.table,
        duration_ns=np.array([13.2, 7.3, 0., 4.1]),
        tone_frequency_ghz=np.tile([.61, .13], (4, 1)),
        tone_amplitude=np.tile([.2, 0.], (4, 1)),
    )
    case = replace(case, source=source, table=table)
    rows = np.arange(4, dtype=np.int64)
    assert case.kernel.guard(table, rows, source, case.budget).all()
    keys, representatives, inverse = np.unique(case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True)
    prepared = case.kernel.prepare(keys, table, rows[representatives], source, case.budget)
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
    expected = np.stack([reference(case, r, state) for r, state in enumerate(case.states)])
    assert_budget(actual, expected, 2e-8)


def test_guard_rejects_strong_probe_shapes_and_untrackable_modes(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    sampled = EnvelopeSpec("sampled", np.array([0., 100.]), np.ones(2, complex))
    for table in (
        replace(case.table, tones=np.ones(4, dtype=np.int64), tone_port=np.tile([0, -1], (4, 1)), tone_envelope=np.tile([0, -1], (4, 1))),
        replace(case.table, envelopes=(sampled,)),
        replace(case.table, duration_ns=np.full(4, 512.), tone_amplitude=np.full((4, 2), .5), tone_frequency_ghz=np.full((4, 2), .13)),
        replace(case.table, tone_frequency_ghz=np.zeros((4, 2))),
    ):
        assert not case.kernel.guard(table, rows, case.source, case.budget).any()


def test_guard_uses_error_budget_for_omitted_harmonics(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    table = replace(case.table, duration_ns=np.full(4, 5.))
    assert not case.kernel.guard(table, rows, case.source, 1e-12).any()
