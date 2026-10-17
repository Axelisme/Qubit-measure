"""Shared path truncation at the array and input-binding seams."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels import dyson

from ..contract.kernel_suite import synthetic_model


def prune_paths(*args):
    prune = getattr(dyson, "prune_dyson_paths", None)
    assert callable(prune), "shared Dyson paths need budgeted conjugate-pair pruning"
    return dyson.prune_dyson_paths(*args)


@pytest.mark.parametrize("growth", [0.0, 0.2])
def test_pruning_sorts_pair_bounds_but_retains_original_path_order(
    growth: float,
) -> None:
    # Interleaved pairs: (0,4) costs 4, (1,3) costs .3, (2,5) costs .4.
    # The budget/8 of 1 drops the two cheap pairs, not either half of (0,4).
    indices = np.array(
        [
            [1, 0, -1, -1, 0],
            [0, 1, -1, -1, 0],
            [3, 1, 0, -1, 1],
            [0, 2, -1, 1, 0],
            [2, 0, -1, 1, 0],
            [3, 2, 0, 1, -1],
        ],
        dtype=np.int64,
    )
    coefficients = np.array([2 / 3, 0.05, 0.4, 0.1, 0.5, 0.4], dtype=complex)
    coefficients *= np.exp(-2 * growth)
    result = prune_paths(
        indices,
        coefficients,
        np.full(4, growth, dtype=complex),
        np.array([0, 2, 1, 3]),
        np.array([2.0, 3.0, 4.0, 1.0, 8.0]),
        -0.5,
        2.0,
    )
    np.testing.assert_array_equal(result.indices, indices[[0, 4]])
    np.testing.assert_array_equal(result.coefficients, coefficients[[0, 4]])
    assert result.discarded_bound == pytest.approx(0.7)
    assert result.discarded_bound <= 1


@pytest.mark.parametrize("gain,duration", [(0.0, 2.0), (-0.5, 0.0)])
def test_zero_strength_can_prune_every_path(gain: float, duration: float) -> None:
    indices = np.array([[0, 0, -1, -1, 0], [0, 0, -1, 1, 0]])
    result = prune_paths(
        indices,
        np.ones(2, complex),
        np.array([0.2 + 0j]),
        np.array([0]),
        np.array([2.0, 1e-3]),
        gain,
        duration,
    )
    assert result.indices.shape == (0, 5)
    assert result.coefficients.shape == (0,)
    assert result.discarded_bound == 0


def static_case(initial: np.ndarray, excited_only: bool = False):
    size = len(initial)
    model = synthetic_model(size)
    x = np.array(
        [[0.0, 0.23, 0.071], [0.23, 0.11, 0.47], [0.071, 0.47, -0.17]], complex
    )
    y = np.array(
        [[0.13, 0.09j, 0.031], [-0.09j, -0.07, 0.21j], [0.031, -0.21j, 0.02]], complex
    )
    if excited_only:
        x[:] = 0
        x[1, 2] = x[2, 1] = 1
        y[:] = 0
    model = replace(
        model,
        energies_ghz=np.zeros(size),
        rates_per_ns=np.zeros((size, size)),
        dephasing_per_ns=np.zeros((size, size)),
        drives=(replace(model.drives[0], in_phase_ghz=x, quadrature_ghz=y),),
    )
    source = ModelSource(initial, lambda level: replace(model, level=level))
    table = SegmentTable(
        np.arange(4),
        np.zeros(4),
        np.array([2.0, 1.0, 3.0, 0.0]),
        np.zeros((4, 1), dtype=np.int64),
        np.ones(4, dtype=np.int64),
        np.zeros((4, 1), dtype=np.int64),
        np.zeros((4, 1)),
        np.array([[0.02], [-0.03], [0.0], [0.04]]),
        np.array([[0.1], [0.7], [-0.8], [1.0]]),
        np.zeros((4, 1), dtype=np.int64),
        np.zeros((4, 1)),
        (EnvelopeSpec("constant"),),
    )
    return source, table


def direct_paths(initial: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Enumerate the untruncated identity-basis expansion for a static model."""
    identity = np.eye(len(initial))
    couplings = np.stack(
        [
            -2j
            * np.pi
            * (
                np.kron((x - sign * 1j * y) / 2, identity)
                - np.kron(identity, ((x - sign * 1j * y) / 2).T)
            )
            for sign in (-1, 1)
        ]
    )
    modes = initial.ravel()
    indices, coefficients = [], []
    for a in range(len(modes)):
        for outer, r in enumerate((-1, 1)):
            for b in range(len(modes)):
                first = couplings[outer, a, b] * modes[b]
                if first != 0:
                    indices.append((a, b, -1, r, 0))
                    coefficients.append(first)
                for inner, s in enumerate((-1, 1)):
                    for c in range(len(modes)):
                        second = (
                            couplings[outer, a, b] * couplings[inner, b, c] * modes[c]
                        )
                        if second != 0:
                            indices.append((a, b, c, r, s))
                            coefficients.append(second)
    return np.array(indices, dtype=np.int64), np.array(coefficients, dtype=complex)


@pytest.mark.parametrize("mixed", [False, True])
def test_bound_apply_error_is_bounded_and_preserves_hermiticity(
    mixed: bool, record_property: Callable[[str, object], None]
) -> None:
    vector = np.array([1.0, 0.7j, 0.3 + 0.2j])
    vector /= np.linalg.norm(vector)
    initial = np.outer(vector, vector.conj()).astype(np.complex128)
    if mixed:
        initial = 0.65 * initial + 0.35 * np.diag([0.2, 0.3, 0.5])
    source, table = static_case(initial)
    model = source.model((0,))
    x, y = model.drives[0].in_phase_ghz, model.drives[0].quadrature_ghz
    indices, coefficients = direct_paths(initial, x, y)
    budget = 0.02
    pruned = prune_paths(
        indices,
        coefficients,
        np.zeros(9, complex),
        np.arange(9).reshape(3, 3).T.ravel(),
        np.r_[np.ones(9), budget],
        0.04,
        3.0,
    )
    assert 0 < pruned.discarded_bound <= budget / 8
    kernel = dyson.DysonKernel()
    rows = np.arange(4, dtype=np.int64)
    prepared = kernel.prepare(
        kernel.prepare_keys(table, rows[:1]), table, rows[:1], source, budget
    )
    bound = kernel.bind_input(prepared, 0, table, rows[::-1], initial, budget)
    assert bound is not None
    actual = np.repeat(initial[None], 4, axis=0)
    for subset in (rows[::2][::-1], rows[1::2][::-1]):
        states = actual[subset].copy()
        kernel.apply(
            bound, np.zeros(len(subset), dtype=np.int64), table, subset, states, budget
        )
        actual[subset] = states
    expected = []
    for row in rows:
        phase = table.tone_theta_rad[row, 0]
        h = (
            2
            * np.pi
            * table.tone_amplitude[row, 0]
            * table.duration_ns[row]
            * (np.cos(phase) * x + np.sin(phase) * y)
        )
        comm = h @ initial - initial @ h
        expected.append(initial - 1j * comm - (h @ comm - comm @ h) / 2)
    expected = np.stack(expected)
    error = float(np.max(abs(actual - expected)))
    residual = float(np.max(abs(actual - actual.conj().swapaxes(-1, -2))))
    unpruned_residual = float(np.max(abs(expected - expected.conj().swapaxes(-1, -2))))
    record_property("error_over_discarded_bound", error / pruned.discarded_bound)
    record_property("discarded_bound", pruned.discarded_bound)
    record_property("hermiticity_residual", residual)
    assert 1e-10 < error <= pruned.discarded_bound + 1e-15
    assert residual <= 10 * unpruned_residual + 3e-16
    np.testing.assert_allclose(actual[2:], expected[2:], rtol=0, atol=1e-15)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("bound_input", [False, True])
def test_ground_with_excited_only_drive_has_empty_table_and_free_output(
    bound_input: bool,
) -> None:
    initial = np.diag([1.0, 0.0, 0.0]).astype(complex)
    source, table = static_case(initial, excited_only=True)
    kernel = dyson.DysonKernel()
    rows = np.arange(4, dtype=np.int64)
    budget = 1e-3
    prepared = kernel.prepare(
        kernel.prepare_keys(table, rows[:1]), table, rows[:1], source, budget
    )
    actual = np.repeat(initial[None], 4, axis=0)
    try:
        if bound_input:
            prepared = kernel.bind_input(prepared, 0, table, rows, initial, budget)
            assert prepared is not None
        mask = kernel.apply(
            prepared, np.zeros(4, dtype=np.int64), table, rows, actual, budget
        )
    except IndexError as error:
        pytest.fail(f"empty path tables must return free evolution, not {error!r}")
    np.testing.assert_array_equal(actual, np.repeat(initial[None], 4, axis=0))
    assert not mask.any()


def test_array_backend_empty_paths_return_free_evolution() -> None:
    initial = np.array([[1.0 + 0j, 0j], [0.5 + 0j, 0j]])
    values = np.array([0.2 + 0j, -0.1 + 0j])
    params = np.array([[0.1, 0.02, 2.0, 0.7], [0.2, -0.03, 1.0, -0.8]])
    try:
        actual = dyson.evaluate_dyson(
            params, values, initial, np.zeros((2, 2, 2), complex), np.ones(3)
        )
    except IndexError as error:
        pytest.fail(f"empty path tables must return free evolution, not {error!r}")
    np.testing.assert_allclose(
        actual, initial * np.exp(params[:, 2:3] * values), rtol=0, atol=1e-15
    )


def test_bound_apply_returns_free_evolution_when_all_paths_are_pruned() -> None:
    initial = np.full((3, 3), 1 / 3, dtype=complex)
    source, table = static_case(initial)
    table = replace(table, tone_amplitude=table.tone_amplitude * 1e-5)
    kernel = dyson.DysonKernel()
    rows = np.arange(4, dtype=np.int64)
    prepared = kernel.prepare(
        kernel.prepare_keys(table, rows[:1]), table, rows[:1], source, 1e-3
    )
    bound = kernel.bind_input(prepared, 0, table, rows, initial, 1e-3)
    assert bound is not None
    actual = np.repeat(initial[None], 4, axis=0)
    kernel.apply(bound, np.zeros(4, dtype=np.int64), table, rows, actual, 1e-3)
    np.testing.assert_array_equal(actual, np.repeat(initial[None], 4, axis=0))
