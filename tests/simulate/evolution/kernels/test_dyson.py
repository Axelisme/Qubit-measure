"""Full-carrier, arbitrary-state second-order evolution at the C5 seam."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource

from ..contract.kernel_suite import KernelCase, KernelContract, assert_budget, reference

KERNEL_NAME = "dyson"


class TestDysonKernel(KernelContract):
    pass


def test_preparation_shared_over_all_sweep_axes(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    keys = case.kernel.prepare_keys(case.table, rows)
    assert len(np.unique(keys)) == 1
    changed = replace(case.table, level=np.array([[0], [1], [0], [1]]))
    assert len(np.unique(case.kernel.prepare_keys(changed, rows))) == 2
    assert np.isfinite(case.kernel.cost(case.table, rows)).all()
    assert (case.kernel.cost(case.table, rows) >= 0).all()


def test_guard_rejects_only_local_invalidity(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    strong = replace(
        case.table, tone_amplitude=np.full((4, 1), 0.3), duration_ns=np.full(4, 20.0)
    )
    assert not case.kernel.guard(strong, rows, case.source, case.budget).any()
    shaped = replace(
        case.table,
        envelopes=(
            EnvelopeSpec(
                "sampled", samples=np.ones(2, complex), sample_ns=np.array([0.0, 1.0])
            ),
        ),
    )
    assert not case.kernel.guard(shaped, rows, case.source, case.budget).any()


def test_full_carrier_and_tight_budget(kernel_case: KernelCase) -> None:
    case = kernel_case
    # Non-circular drive requires both carrier sidebands. Tiny gains isolate
    # divided-difference accuracy and pruning from the third-order remainder.
    model = case.source.model((0,))
    model = replace(
        model,
        drives=(
            replace(
                model.drives[0],
                quadrature_ghz=np.zeros_like(model.drives[0].quadrature_ghz),
            ),
        ),
    )
    source = ModelSource(
        case.source.initial_density, lambda level: replace(model, level=level)
    )
    table = replace(case.table, tone_amplitude=case.table.tone_amplitude * 1e-3)
    case = replace(case, table=table, source=source, budget=1e-11)
    rows = np.arange(4, dtype=np.int64)
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = case.kernel.prepare(keys, table, rows[first], source, case.budget)
    before = prepared.nbytes
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
    expected = np.stack(
        [reference(case, i, state) for i, state in enumerate(case.states)]
    )
    assert_budget(actual, expected, case.budget)
    assert prepared.nbytes == before


def test_batch_of_distinct_coherent_inputs(kernel_case: KernelCase) -> None:
    case = kernel_case
    size = case.states.shape[-1]
    states = case.states.copy()
    states[1] = np.eye(size) / size
    states[2] = np.diag(np.arange(1, size + 1) / sum(range(1, size + 1)))
    states[3] = states[0].conj()
    table = replace(
        case.table,
        tone_frequency_ghz=np.array([[0.0], [-0.13], [0.13], [0.2]]),
        tone_amplitude=np.array([[1e-5], [-2e-5], [0.0], [1e-5]]),
        duration_ns=np.array([1.0, 2.0, 3.0, 0.0]),
    )
    case = replace(case, table=table, states=states, budget=1e-9)
    rows = np.arange(4, dtype=np.int64)
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = case.kernel.prepare(keys, table, rows[first], case.source, case.budget)
    actual = states.copy()
    case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
    expected = np.stack(
        [reference(case, row, state) for row, state in enumerate(states)]
    )
    assert_budget(actual, expected, case.budget)
    np.testing.assert_allclose(actual[3], states[3], rtol=0, atol=1e-14)


@pytest.mark.parametrize("changed_basis", [False, True])
def test_guard_retains_first_order_population_from_coherence(
    kernel_case: KernelCase, changed_basis: bool
) -> None:
    case = kernel_case
    model = case.source.model((0,))
    size = len(model.energies_ghz)
    x = np.zeros((size, size), dtype=np.complex128)
    x[0, 1] = x[1, 0] = 1
    model = replace(
        model,
        energies_ghz=np.zeros(size),
        rates_per_ns=np.zeros((size, size)),
        dephasing_per_ns=np.zeros((size, size)),
        drives=(replace(model.drives[0], in_phase_ghz=x, quadrature_ghz=x * 0),),
    )
    vector = np.zeros(size, dtype=np.complex128)
    vector[:2] = np.array([1, 1j]) / np.sqrt(2)
    coherent = np.outer(vector, vector.conj())
    # A nontrivial ambient basis must not change same-basis propagation. For a
    # distinct target basis, rotate the DC input so its target state is coherent.
    basis = np.eye(size, dtype=np.complex128)
    basis[:2, :2] = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    dc = replace(model, eigenvectors=basis)
    target = replace(model, level=(1,)) if changed_basis else dc
    overlap = target.eigenvectors.conj().T @ dc.eigenvectors
    initial = overlap.conj().T @ coherent @ overlap
    source = ModelSource(initial, lambda level: dc if level == (0,) else target)
    table = replace(
        case.table,
        level=np.full((4, 1), int(changed_basis), dtype=np.int64),
        tone_frequency_ghz=np.zeros((4, 1)),
        tone_amplitude=np.array([[0.002], [0.004], [0.2], [0.2]]),
        tone_theta_rad=np.zeros((4, 1)),
        duration_ns=np.array([1.0, 1.0, 1.0, 0.0]),
    )
    # For H=g*X and this equatorial state, second-order population transfer
    # vanishes. The linear term is +/-2*pi*g*t and crosses .02 only in row 1.
    rows = np.arange(4, dtype=np.int64)
    np.testing.assert_array_equal(
        case.kernel.guard(table, rows, source, case.budget), [True, False, False, True]
    )
    case = replace(case, source=source, table=table)
    exact = np.diag(reference(case, 1, coherent) - coherent)
    assert np.max(abs(exact)) > 0.02
