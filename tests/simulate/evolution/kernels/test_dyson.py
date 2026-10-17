"""Full-carrier, arbitrary-state second-order evolution at the C5 seam."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels.dyson import DysonKernel

from ..contract.kernel_suite import (
    KernelCase,
    KernelContract,
    assert_budget,
    reference,
    synthetic_model,
)

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


@pytest.mark.parametrize("gap", [0.0, 9e-6, 1.1e-5, 0.249, 0.251, 0.499, 0.501])
def test_resonant_repeated_and_nearby_modes(
    kernel_case: KernelCase, gap: float
) -> None:
    case = kernel_case
    model = case.source.model((0,))
    size = len(model.energies_ghz)
    # Unit duration makes adjacent free-mode gaps equal to gap, in radians.
    # Equal spacings and a resonant carrier also create repeated shifted roots.
    # The 2*gap endpoints cross 0.5 as well as the adjacent-mode endpoints.
    model = replace(
        model,
        energies_ghz=np.arange(size) * gap / (2 * np.pi),
        rates_per_ns=np.zeros((size, size)),
        dephasing_per_ns=np.zeros((size, size)),
        drives=(
            replace(
                model.drives[0],
                quadrature_ghz=np.zeros((size, size), dtype=np.complex128),
            ),
        ),
    )
    source = ModelSource(
        case.source.initial_density, lambda level: replace(model, level=level)
    )
    table = replace(
        case.table,
        duration_ns=np.ones(4),
        tone_frequency_ghz=np.full((4, 1), gap / (2 * np.pi)),
        tone_amplitude=np.array([[1e-5], [-2e-5], [0.0], [1e-5]]),
    )
    case = replace(case, source=source, table=table, budget=1e-10)
    rows = np.arange(4, dtype=np.int64)
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = case.kernel.prepare(keys, table, rows[first], source, case.budget)
    for distinct in (False, True):
        states = case.states.copy()
        if distinct:
            states[1] = np.eye(size) / size
            states[2] = np.diag(np.arange(1, size + 1) / sum(range(1, size + 1)))
            states[3] = states[0].conj()
        actual = states.copy()
        case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
        expected = np.stack(
            [reference(case, row, state) for row, state in enumerate(states)]
        )
        assert_budget(actual, expected, case.budget)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=case.budget)
        np.testing.assert_allclose(
            actual, actual.conj().swapaxes(-1, -2), rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            np.trace(actual, axis1=-2, axis2=-1), 1, rtol=0, atol=1e-13
        )


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
        tone_amplitude=np.array([[0.001], [0.004], [0.2], [0.2]]),
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


@pytest.mark.parametrize("changed_basis", [False, True])
def test_guard_distinguishes_diagonal_mixture_from_same_population_coherence(
    kernel_case: KernelCase, changed_basis: bool
) -> None:
    case = kernel_case
    model = case.source.model((0,))
    size = len(model.energies_ghz)
    model = replace(
        model,
        rates_per_ns=np.zeros((size, size)),
        dephasing_per_ns=np.zeros((size, size)),
        drives=(
            replace(
                model.drives[0],
                in_phase_ghz=model.drives[0].in_phase_ghz * 0.02,
                quadrature_ghz=model.drives[0].quadrature_ghz * 0.02,
            ),
        ),
    )
    basis = np.eye(size, dtype=complex)
    basis[:2, :2] = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    target = replace(model, level=(1,), eigenvectors=basis) if changed_basis else model
    weights = np.arange(size, 0, -1, dtype=float)
    weights /= weights.sum()
    diagonal = np.diag(weights).astype(complex)
    vector = np.sqrt(weights) * np.exp(1j * np.arange(size))
    coherent = np.outer(vector, vector.conj())
    table = replace(
        case.table,
        level=np.full((4, 1), int(changed_basis), dtype=np.int64),
        start_ns=np.zeros(4),
        duration_ns=np.full(4, 512.0),
        tone_frequency_ghz=np.full((4, 1), 0.4),
        tone_amplitude=np.full((4, 1), 0.02),
    )
    rows = np.arange(4, dtype=np.int64)
    overlap = target.eigenvectors.conj().T @ model.eigenvectors
    for initial, expected in ((diagonal, True), (coherent, False)):
        # Source is stored in the DC basis; the promised input is in the row basis.
        dc_initial = overlap.conj().T @ initial @ overlap
        source = ModelSource(
            dc_initial, lambda level: model if level == (0,) else target
        )
        np.testing.assert_array_equal(
            case.kernel.guard(table, rows, source, case.budget), np.full(4, expected)
        )


@pytest.mark.parametrize("size", [3, 4])
@pytest.mark.parametrize("dissipative", [False, True])
def test_population_bound_is_conservative_for_ground_mixed_and_coherent_inputs(
    size: int, dissipative: bool, record_property: Callable[[str, object], None]
) -> None:
    model = synthetic_model(size)
    if not dissipative:
        model = replace(
            model,
            rates_per_ns=np.zeros((size, size)),
            dephasing_per_ns=np.zeros((size, size)),
        )
    frequency, gain, duration = np.meshgrid(
        [-0.3, 0.0, 0.129, 0.13, 0.131, 0.4],
        [0.0, 0.0004, 0.002, 0.01],
        [0.0, 1.0, 5.0],
        indexing="ij",
    )
    n = frequency.size
    table = SegmentTable(
        np.arange(n),
        np.zeros(n),
        duration.ravel(),
        np.zeros((n, 1), dtype=np.int64),
        np.ones(n, dtype=np.int64),
        np.zeros((n, 1), dtype=np.int64),
        frequency.reshape(n, 1),
        gain.reshape(n, 1),
        np.full((n, 1), 0.7),
        np.zeros((n, 1), dtype=np.int64),
        np.zeros((n, 1)),
        (EnvelopeSpec("constant"),),
    )
    kernel = DysonKernel()
    rows = np.arange(n, dtype=np.int64)
    budget = 1e-3
    threshold = min(0.02, (1.5 * budget) ** (2 / 3))
    ground = np.zeros(size, dtype=complex)
    ground[0] = 1
    coherent = np.exp(1j * np.arange(size)) / np.sqrt(size)
    outgoing = model.rates_per_ns.sum(axis=0)
    q = model.rates_per_ns - np.diag(outgoing)
    leaks, extra = 0, 0
    weights = np.arange(size, 0, -1, dtype=float)
    weights /= weights.sum()
    for initial in (
        np.outer(ground, ground.conj()),
        np.outer(coherent, coherent.conj()),
        np.diag(weights).astype(complex),
    ):
        source = ModelSource(initial, lambda level: replace(model, level=level))
        accepted = kernel.guard(table, rows, source, budget)
        keys, first, inverse = np.unique(
            kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
        )
        prepared = kernel.prepare(keys, table, rows[first], source, budget)
        actual = np.repeat(initial[None], n, axis=0)
        kernel.apply(prepared, inverse, table, rows, actual, budget)
        free = np.stack(
            [expm(q * t) @ np.real(np.diag(initial)) for t in duration.ravel()]
        )
        transfer = np.max(abs(np.diagonal(actual, axis1=1, axis2=2) - free), axis=1)
        leaks += int(np.count_nonzero(accepted & (transfer > threshold)))
        extra += int(np.count_nonzero(~accepted & (transfer <= threshold)))
    record_property("rows", 3 * n)
    record_property("leaks", leaks)
    record_property("extra_rejections", extra)
    assert leaks == 0
    assert extra > 0


def test_population_guard_reserves_pruning_margin(kernel_case: KernelCase) -> None:
    case = kernel_case
    model = case.source.model((0,))
    size = len(model.energies_ghz)
    x = np.zeros((size, size), dtype=complex)
    x[0, 1] = x[1, 0] = 1
    model = replace(
        model,
        energies_ghz=np.zeros(size),
        rates_per_ns=np.zeros((size, size)),
        dephasing_per_ns=np.zeros((size, size)),
        drives=(replace(model.drives[0], in_phase_ghz=x, quadrature_ghz=x * 0),),
    )
    source = ModelSource(
        np.diag([1.0] + [0.0] * (size - 1)).astype(complex),
        lambda level: replace(model, level=level),
    )
    threshold = min(0.02, (1.5 * case.budget) ** (2 / 3))
    # On resonance, second-order transfer and B2 are both (2*pi*g*T)**2.
    # Both rows are below the old population threshold, but only row 0 has
    # enough headroom for the omitted-path allowance.
    gains = np.sqrt(
        threshold - case.budget * np.array([1 / 4, 1 / 16, 1 / 4, 1 / 16])
    ) / (2 * np.pi)
    table = replace(
        case.table,
        start_ns=np.zeros(4),
        duration_ns=np.ones(4),
        tone_frequency_ghz=np.zeros((4, 1)),
        tone_theta_rad=np.zeros((4, 1)),
        tone_amplitude=gains[:, None],
    )
    rows = np.arange(2, dtype=np.int64)
    np.testing.assert_array_equal(
        case.kernel.guard(table, rows, source, case.budget), [True, False]
    )


@pytest.mark.parametrize("size", [3, 4])
def test_guard_bounds_unknown_coherence_phase_and_prior_history(size: int) -> None:
    model = synthetic_model(size)
    x = np.zeros((size, size), dtype=complex)
    x[1, 2] = x[2, 1] = 1
    model = replace(
        model,
        energies_ghz=np.zeros(size),
        rates_per_ns=np.zeros((size, size)),
        dephasing_per_ns=np.zeros((size, size)),
        drives=(replace(model.drives[0], in_phase_ghz=x, quadrature_ghz=x * 0),),
    )
    table = SegmentTable(
        np.arange(2),
        np.array([0.0, 1.0]),
        np.ones(2),
        np.zeros((2, 1), dtype=np.int64),
        np.ones(2, dtype=np.int64),
        np.zeros((2, 1), dtype=np.int64),
        np.zeros((2, 1)),
        np.full((2, 1), 0.003),
        np.zeros((2, 1)),
        np.zeros((2, 1), dtype=np.int64),
        np.zeros((2, 1)),
        (EnvelopeSpec("constant"),),
    )
    ground = np.diag([0.999, 0.001] + [0.0] * (size - 2)).astype(complex)
    source = ModelSource(ground, lambda level: replace(model, level=level))
    kernel = DysonKernel()
    rows = np.arange(2, dtype=np.int64)
    # Near-ground DC populations give negligible transfer on this edge.
    # Only the first row is allowed to rely on them.
    np.testing.assert_array_equal(
        kernel.guard(table, rows, source, 1e-3), [True, False]
    )
    vector = np.zeros(size, dtype=complex)
    vector[1:3] = 1 / np.sqrt(2)
    stationary = np.outer(vector, vector.conj())
    coherent_source = ModelSource(stationary, lambda level: replace(model, level=level))
    # The actual real coherence commutes with X. B2 must nevertheless account
    # for every coherence phase consistent with these populations.
    assert not kernel.guard(table, rows[:1], coherent_source, 1e-3)[0]


@pytest.mark.parametrize("budget", [1e-3, 1.0])
def test_apply_reports_population_transfer_without_correcting_output(
    kernel_case: KernelCase, budget: float
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
    src = ModelSource(coherent, lambda level: replace(model, level=level))
    table = replace(
        case.table,
        tone_frequency_ghz=np.zeros((4, 1)),
        tone_amplitude=np.array([[0.004], [0.0005], [0.003], [0.0]]),
        tone_theta_rad=np.zeros((4, 1)),
        duration_ns=np.ones(4),
    )
    rows = np.array([2, 0, 3, 1], dtype=np.int64)
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = case.kernel.prepare(keys, table, rows[first], src, budget)
    actual = np.repeat(coherent[None], 4, axis=0)
    mask = case.kernel.apply(prepared, inverse, table, rows, actual, budget)
    assert isinstance(mask, np.ndarray), "apply must return an exceeded mask"
    # 2*pi*g is the population transfer. .003 crosses the budget-scaled
    # threshold .0131 but not the .02 cap; .004 crosses both.
    expected_mask = (
        [True, True, False, False] if budget < 1 else [False, True, False, False]
    )
    np.testing.assert_array_equal(mask, expected_mask)
    # Independent second-order polynomial for static H=g*X. A warning must
    # neither correct nor recompute the output, including exceeded rows.
    for slot, row in enumerate(rows):
        h = 2 * np.pi * table.tone_amplitude[row, 0] * x
        commutator = h @ coherent - coherent @ h
        expected = coherent - 1j * commutator - (h @ commutator - commutator @ h) / 2
        np.testing.assert_allclose(actual[slot], expected, rtol=0, atol=1e-14)
    split_masks = []
    for slot, row in enumerate(rows):
        state = coherent[None].copy()
        split_masks.append(
            case.kernel.apply(
                prepared,
                inverse[slot : slot + 1],
                table,
                np.array([row]),
                state,
                budget,
            )
        )
        np.testing.assert_allclose(state[0], actual[slot], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(np.concatenate(split_masks), mask)


def test_apply_population_check_subtracts_free_decay(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    table = replace(
        case.table, tone_amplitude=np.zeros((4, 1)), duration_ns=np.full(4, 20.0)
    )
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = case.kernel.prepare(keys, table, rows[first], case.source, case.budget)
    actual = case.states.copy()
    mask = case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
    assert isinstance(mask, np.ndarray), "apply must return an exceeded mask"
    assert not mask.any()
    change = np.diagonal(actual - case.states, axis1=1, axis2=2)
    assert np.max(abs(change)) > 0.02


@pytest.mark.parametrize("gap", [0.0, 9e-6, 1.1e-5, 0.249, 0.251, 0.499, 0.501])
def test_input_bound_paths_match_reference_at_modal_boundaries(
    kernel_case: KernelCase, gap: float
) -> None:
    case = kernel_case
    bind = getattr(case.kernel, "bind_input", None)
    assert callable(bind), "Dyson must offer input-bound path evaluation"
    model = case.source.model((0,))
    size = len(model.energies_ghz)
    model = replace(
        model,
        energies_ghz=np.arange(size) * gap / (2 * np.pi),
        rates_per_ns=np.zeros((size, size)),
        dephasing_per_ns=np.zeros((size, size)),
    )
    source = ModelSource(
        case.source.initial_density, lambda level: replace(model, level=level)
    )
    table = replace(
        case.table,
        start_ns=np.zeros(4),
        duration_ns=np.array([1.0, 2.0, 0.0, 1.0]),
        tone_frequency_ghz=np.array(
            [[0.0], [gap / (2 * np.pi)], [0.13], [-gap / (2 * np.pi)]]
        ),
        tone_amplitude=np.array([[1e-5], [-2e-5], [1e-5], [0.0]]),
    )
    case = replace(case, source=source, table=table, budget=1e-9)
    rows = np.arange(4, dtype=np.int64)
    keys = np.unique(case.kernel.prepare_keys(table, rows))
    prepared = case.kernel.prepare(keys, table, rows[:1], source, case.budget)
    for initial in (case.states[0], source.initial_density):
        bound = case.kernel.bind_input(
            prepared, 0, table, rows[::-1], initial, case.budget
        )
        assert bound is not None
        actual = np.repeat(initial[None], 4, axis=0)
        for subset in (rows[::2][::-1], rows[1::2][::-1]):
            states = actual[subset].copy()
            case.kernel.apply(
                bound,
                np.zeros(len(subset), dtype=np.int64),
                table,
                subset,
                states,
                case.budget,
            )
            actual[subset] = states
        expected = np.stack([reference(case, int(row), initial) for row in rows])
        assert_budget(actual, expected, case.budget)
        np.testing.assert_allclose(
            actual, actual.conj().swapaxes(-1, -2), rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            np.trace(actual, axis1=-2, axis2=-1), 1, rtol=0, atol=1e-13
        )
