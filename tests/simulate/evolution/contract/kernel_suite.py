"""Reusable black-box checks for any C5 kernel.

Kernel tickets supply a KernelCase fixture to subclasses of KernelContract.
The default Hamiltonian is a constant envelope carrier. Shaped/continuous-flux
cases provide hamiltonian_ghz(time_ns, row) in that row's model coordinates;
model_at(time_ns, row) optionally supplies piecewise baths. The independent
reference constructs individual Lindblad jumps, never production RHS helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

import zcu_tools.simulate.evolution as evolution

if TYPE_CHECKING:
    from zcu_tools.simulate.evolution.kernel import Kernel, SegmentTable


def synthetic_model(size: int = 3) -> evolution.SecularModel:
    """Return an M=3/4 model with bidirectional decay and pure dephasing."""
    rates = np.zeros((size, size))
    rates[0, 1], rates[1, 0], rates[1, 2] = .015, .002, .007
    slopes = np.arange(size) * .03
    x = np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1)
    y = 1j * (np.diag(np.ones(size - 1), 1) - np.diag(np.ones(size - 1), -1))
    return evolution.SecularModel((0,), f"synthetic-{size}", np.arange(size, dtype=float) * .13, np.eye(size, dtype=complex), (evolution.SecularDrive("drive", x.astype(complex), y),), rates, (slopes[:, None] - slopes[None, :]) ** 2 / 2)


@dataclass(frozen=True)
class KernelCase:
    """One accepted set of independent rows and the states to evolve.

    kernel: algorithm under test; all rows must pass its guard.
    source, table: model source and segment parameters passed unchanged to C5.
    states: complex (rows,M,M) input densities, including coherence.
    budget: population and low-four absolute error allowance.
    hamiltonian_ghz: optional independent full H/h evaluation for each row.
    model_at: optional time-dependent secular bath model for each row.
    breakpoints_ns: optional absolute reference integration boundaries by row.
    """

    kernel: Kernel
    source: evolution.ModelSource
    table: SegmentTable
    states: NDArray[np.complex128]
    budget: float = 1e-3
    hamiltonian_ghz: Callable[[float, int], NDArray[np.complex128]] | None = None
    model_at: Callable[[float, int], evolution.SecularModel] | None = None
    breakpoints_ns: tuple[tuple[float, ...], ...] = ()


def reference(case: KernelCase, row: int, state: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Integrate the full Lindblad equation at rtol=2e-12, atol=2e-14."""
    table = case.table
    model = case.source.model(tuple(int(v) for v in table.level[row]))
    size = len(model.energies_ghz)
    def rhs(time: float, flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        bath = model if case.model_at is None else case.model_at(time, row)
        rho = flat.reshape(size, size)
        h = np.diag(model.energies_ghz).astype(complex)
        if case.hamiltonian_ghz is not None:
            h = case.hamiltonian_ghz(time, row)
        else:
            for tone in range(int(table.tones[row])):
                drive = model.drives[table.tone_port[row, tone]]
                phase = table.tone_theta_rad[row, tone] + 2 * np.pi * table.tone_frequency_ghz[row, tone] * (time - table.start_ns[row])
                z = table.tone_amplitude[row, tone] * np.exp(1j * phase)
                h += z.real * drive.in_phase_ghz + z.imag * drive.quadrature_ghz
        derivative = -2j * np.pi * (h @ rho - rho @ h)
        for target in range(size):
            for source in range(size):
                if target != source:
                    jump = np.zeros((size, size), complex)
                    jump[target, source] = np.sqrt(bath.rates_per_ns[target, source])
                    loss = jump.conj().T @ jump
                    derivative += jump @ rho @ jump.conj().T - (loss @ rho + rho @ loss) / 2
        derivative -= bath.dephasing_per_ns * rho
        return derivative.ravel()
    start = float(table.start_ns[row])
    end = start + float(table.duration_ns[row])
    cuts = case.breakpoints_ns[row] if case.breakpoints_ns else ()
    result = state.copy().ravel()
    for left, right in zip((start, *cuts), (*cuts, end), strict=True):
        if left == right:
            continue
        solution = solve_ivp(rhs, (left, right), result, method="DOP853", rtol=2e-12, atol=2e-14, max_step=.05)
        assert solution.success, solution.message
        result = solution.y[:, -1]
    return result.reshape(size, size)


def assert_budget(actual: NDArray[np.complex128], expected: NDArray[np.complex128], budget: float) -> None:
    """Check populations and the low-four density block independently."""
    assert np.isfinite(actual).all()
    assert np.max(np.abs(np.diagonal(actual, axis1=-2, axis2=-1) - np.diagonal(expected, axis1=-2, axis2=-1))) <= budget
    assert np.max(np.abs(actual[..., :4, :4] - expected[..., :4, :4])) <= budget


class KernelContract:
    """Inherit and supply the kernel_case fixture; no kernel internals used."""

    def test_error_within_budget(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.states), dtype=np.int64)
        assert case.kernel.guard(case.table, rows, case.source, case.budget).all()
        keys, inverse = np.unique(case.kernel.prepare_keys(case.table, rows), return_inverse=True)
        prepared = case.kernel.prepare(keys, case.source, case.budget)
        assert prepared.nbytes >= 0
        actual = case.states.copy()
        case.kernel.apply(prepared, inverse, case.table, rows, actual, case.budget)
        expected = np.stack([reference(case, row, state) for row, state in enumerate(case.states)])
        assert_budget(actual, expected, case.budget)

    def test_rows_can_be_split(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.states), dtype=np.int64)
        keys, inverse = np.unique(case.kernel.prepare_keys(case.table, rows), return_inverse=True)
        prepared = case.kernel.prepare(keys, case.source, case.budget)
        whole = case.states.copy()
        case.kernel.apply(prepared, inverse, case.table, rows, whole, case.budget)
        split = case.states.copy()
        # Non-contiguous, reversed subsets detect accidental row-position indexing.
        for subset in (rows[::2][::-1], rows[1::2][::-1]):
            chunk = split[subset].copy()
            case.kernel.apply(prepared, inverse[subset], case.table, subset, chunk, case.budget)
            split[subset] = chunk
        np.testing.assert_allclose(split, whole, rtol=0, atol=1e-12)

    def test_prepare_is_density_independent(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.states), dtype=np.int64)
        keys, inverse = np.unique(case.kernel.prepare_keys(case.table, rows), return_inverse=True)
        def no_density_source() -> evolution.ModelSource:
            class NoDensity(evolution.ModelSource):
                @property
                def initial_density(self) -> NDArray[np.complex128]:
                    raise AssertionError("prepare must not read density")
            return NoDensity(case.states[0], case.source.model)
        prepared = case.kernel.prepare(keys, no_density_source(), case.budget)
        for states in (case.states, np.repeat((np.eye(case.states.shape[-1]) / case.states.shape[-1])[None].astype(complex), len(rows), axis=0)):
            actual = states.copy()
            case.kernel.apply(prepared, inverse, case.table, rows, actual, case.budget)
            expected = np.stack([reference(case, row, state) for row, state in enumerate(states)])
            assert_budget(actual, expected, case.budget)
