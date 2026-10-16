"""Reusable black-box checks for any C5 kernel.

Kernel tickets supply a KernelCase fixture to subclasses of KernelContract.
Each row uses its segment eigenbasis, with carriers and constant or sampled
envelopes. When table.flux_waveform is supplied the reference evaluates the
exact continuous Hamiltonian and can switch quantized baths inside a segment.
It constructs individual Lindblad jumps, never production RHS helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import zcu_tools.simulate.evolution as evolution
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from zcu_tools.simulate.evolution import kernel


def synthetic_model(size: int = 3) -> evolution.SecularModel:
    """Return an M=3/4 model with bidirectional decay and pure dephasing."""
    rates = np.zeros((size, size))
    rates[0, 1], rates[1, 0], rates[1, 2] = 0.015, 0.002, 0.007
    slopes = np.arange(size) * 0.03
    x = np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1)
    y = 1j * (np.diag(np.ones(size - 1), 1) - np.diag(np.ones(size - 1), -1))
    return evolution.SecularModel(
        (0,),
        f"synthetic-{size}",
        np.arange(size, dtype=float) * 0.13,
        np.eye(size, dtype=complex),
        (
            evolution.SecularDrive(
                "drive", x.astype(np.complex128), y.astype(np.complex128)
            ),
        ),
        rates,
        (slopes[:, None] - slopes[None, :]) ** 2 / 2,
    )


@dataclass(frozen=True)
class KernelCase:
    """One accepted set of independent rows and the states to evolve.

    kernel: algorithm under test; all rows must pass its guard.
    source, table: model source and segment parameters passed unchanged to C5.
    states: complex (rows,M,M) input densities, including coherence.
    budget: population and low-four absolute error allowance.
    """

    kernel: kernel.Kernel
    source: evolution.ModelSource
    table: kernel.SegmentTable
    states: NDArray[np.complex128]
    budget: float = 1e-3


def _reference_hamiltonian(
    case: KernelCase,
    row: int,
    time: float,
    model: evolution.SecularModel,
) -> NDArray[np.complex128]:
    """Assemble H/h independently from expansion terms and playback samples."""
    table = case.table
    h = np.diag(model.energies_ghz).astype(complex)
    waveform = table.flux_waveform
    if waveform is not None:
        first, last = waveform.offsets[table.timeline[row] : table.timeline[row] + 2]
        sample_time = waveform.time_ns[first:last]
        sample_flux = waveform.flux[first:last]
        expansion = case.source.flux_expansion()
        flux = np.array(
            [
                np.interp(time, sample_time, sample_flux[:, loop])
                for loop in range(sample_flux.shape[1])
            ]
        )
        fixed_h = expansion.static.copy()
        for term in expansion.terms:
            angle = np.pi * flux[term.loop] * (2 if "2pi" in term.kind else 1)
            value = np.cos(angle) if term.kind.startswith("cos") else np.sin(angle)
            fixed_h += value * term.matrix
        h = model.eigenvectors.conj().T @ fixed_h @ model.eigenvectors
    for tone in range(int(table.tones[row])):
        drive = model.drives[table.tone_port[row, tone]]
        elapsed = time - table.start_ns[row]
        phase = (
            table.tone_theta_rad[row, tone]
            + 2 * np.pi * table.tone_frequency_ghz[row, tone] * elapsed
        )
        envelope = table.envelopes[table.tone_envelope[row, tone]]
        amplitude = 1 + 0j
        if envelope.kind == "sampled":
            local = elapsed + table.tone_envelope_offset_ns[row, tone]
            amplitude = complex(
                np.interp(local, envelope.sample_ns, envelope.samples.real),
                np.interp(local, envelope.sample_ns, envelope.samples.imag),
            )
        z = table.tone_amplitude[row, tone] * amplitude * np.exp(1j * phase)
        h += z.real * drive.in_phase_ghz + z.imag * drive.quadrature_ghz
    return h


def reference(
    case: KernelCase,
    row: int,
    state: NDArray[np.complex128],
    *,
    bath: evolution.FluxSchedule | None = None,
) -> NDArray[np.complex128]:
    """Integrate full Lindblad at rtol=2e-12, atol=2e-14 in row eigenbasis.

    table.flux_waveform selects continuous H(flux) from source.flux_expansion;
    otherwise H0 is static. Optional bath supplies a ragged per-timeline grid
    schedule including switches inside the row. Each bath dissipator is rotated
    by V_segment† V_bath, independently of Hamiltonian/drive evaluation. None
    uses the row's constant bath. Stops at bath switches and waveform knots.
    """
    table = case.table
    model = case.source.model(tuple(int(v) for v in table.level[row]))
    size = len(model.energies_ghz)

    waveform = table.flux_waveform
    point = int(table.timeline[row])
    sample_time = np.empty(0)
    if waveform is not None:
        first, last = waveform.offsets[point : point + 2]
        sample_time = waveform.time_ns[first:last]
    bath_time = np.empty(0)
    bath_levels = np.empty((0, table.level.shape[1]), dtype=np.int64)
    if bath is not None:
        first, last = bath.offsets[point : point + 2]
        bath_time = bath.start_ns[first:last]
        bath_levels = bath.level[first:last]

    def rhs(time: float, flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        rho = flat.reshape(size, size)
        h = _reference_hamiltonian(case, row, time, model)
        derivative = -2j * np.pi * (h @ rho - rho @ h)
        bath_rho = overlap.conj().T @ rho @ overlap
        dissipation = -bath_model.dephasing_per_ns * bath_rho
        for jump, loss in jumps:
            dissipation += (
                jump @ bath_rho @ jump.conj().T
                - (loss @ bath_rho + bath_rho @ loss) / 2
            )
        derivative += overlap @ dissipation @ overlap.conj().T
        return derivative.ravel()

    start = float(table.start_ns[row])
    end = start + float(table.duration_ns[row])
    if end == start:
        return state.copy()
    boundaries = np.unique(np.r_[start, end, bath_time, sample_time])
    boundaries = boundaries[(boundaries >= start) & (boundaries <= end)]
    result = state.copy().ravel()
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        bath_model = model
        if bath is not None:
            index = np.searchsorted(bath_time, (left + right) / 2, side="right") - 1
            bath_model = case.source.model(tuple(int(v) for v in bath_levels[index]))
        overlap = model.eigenvectors.conj().T @ bath_model.eigenvectors
        # Jumps and basis changes are constant between bath boundaries. Build
        # only nonzero jumps once, retaining an oracle independent of product RHS.
        jumps = []
        for target, origin in np.argwhere(bath_model.rates_per_ns > 0):
            if target != origin:
                jump = np.zeros((size, size), complex)
                jump[target, origin] = np.sqrt(bath_model.rates_per_ns[target, origin])
                jumps.append((jump, jump.conj().T @ jump))
        solution = solve_ivp(
            rhs,
            (left, right),
            result,
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
            max_step=0.05,
        )
        assert solution.success, solution.message
        result = solution.y[:, -1]
    return result.reshape(size, size)


def assert_budget(
    actual: NDArray[np.complex128], expected: NDArray[np.complex128], budget: float
) -> None:
    """Check populations and the low-four density block independently."""
    assert np.isfinite(actual).all()
    assert (
        np.max(
            np.abs(
                np.diagonal(actual, axis1=-2, axis2=-1)
                - np.diagonal(expected, axis1=-2, axis2=-1)
            )
        )
        <= budget
    )
    assert np.max(np.abs(actual[..., :4, :4] - expected[..., :4, :4])) <= budget


class KernelContract:
    """Inherit and supply the kernel_case fixture; no kernel internals used."""

    def test_error_within_budget(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.states), dtype=np.int64)
        assert case.kernel.guard(case.table, rows, case.source, case.budget).all()
        keys, first, inverse = np.unique(
            case.kernel.prepare_keys(case.table, rows),
            return_index=True,
            return_inverse=True,
        )
        prepared = case.kernel.prepare(
            keys, case.table, rows[first], case.source, case.budget
        )
        assert prepared.nbytes >= 0
        actual = case.states.copy()
        case.kernel.apply(prepared, inverse, case.table, rows, actual, case.budget)
        expected = np.stack(
            [reference(case, row, state) for row, state in enumerate(case.states)]
        )
        assert_budget(actual, expected, case.budget)

    def test_rows_can_be_split(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.states), dtype=np.int64)
        keys, first, inverse = np.unique(
            case.kernel.prepare_keys(case.table, rows),
            return_index=True,
            return_inverse=True,
        )
        prepared = case.kernel.prepare(
            keys, case.table, rows[first], case.source, case.budget
        )
        whole = case.states.copy()
        case.kernel.apply(prepared, inverse, case.table, rows, whole, case.budget)
        split = case.states.copy()
        # Non-contiguous, reversed subsets detect accidental row-position indexing.
        for subset in (rows[::2][::-1], rows[1::2][::-1]):
            chunk = split[subset].copy()
            case.kernel.apply(
                prepared, inverse[subset], case.table, subset, chunk, case.budget
            )
            split[subset] = chunk
        np.testing.assert_allclose(split, whole, rtol=0, atol=1e-12)

    def test_prepare_is_density_independent(self, kernel_case: KernelCase) -> None:
        case = kernel_case
        rows = np.arange(len(case.states), dtype=np.int64)
        keys, first, inverse = np.unique(
            case.kernel.prepare_keys(case.table, rows),
            return_index=True,
            return_inverse=True,
        )

        def no_density_source() -> evolution.ModelSource:
            class NoDensity(evolution.ModelSource):
                @property
                def initial_density(self) -> NDArray[np.complex128]:
                    raise AssertionError("prepare must not read density")

            expansion = (
                case.source.flux_expansion()
                if case.table.flux_waveform is not None
                else None
            )
            return NoDensity(case.states[0], case.source.model, expansion)

        prepared = case.kernel.prepare(
            keys, case.table, rows[first], no_density_source(), case.budget
        )
        for states in (
            case.states,
            np.repeat(
                (np.eye(case.states.shape[-1]) / case.states.shape[-1])[None].astype(
                    complex
                ),
                len(rows),
                axis=0,
            ),
        ):
            actual = states.copy()
            case.kernel.apply(prepared, inverse, case.table, rows, actual, case.budget)
            expected = np.stack(
                [reference(case, row, state) for row, state in enumerate(states)]
            )
            assert_budget(actual, expected, case.budget)
