"""Independent full-Hamiltonian Lindblad oracle for evolution acceptance tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import zcu_tools.simulate.evolution as evolution
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from zcu_tools.simulate.evolution import ReferenceResult, SolverTolerances

if TYPE_CHECKING:
    from .kernel_suite import KernelCase


def density_error(
    actual: NDArray[np.complex128], expected: NDArray[np.complex128]
) -> float:
    """Return max(abs(actual-expected)) over all entries and leading axes.

    Arrays must have identical, nonempty (...,M,M) shapes and finite values.
    Invalid shapes or nonfinite values raise ValueError; no projection onto
    populations or a truncated density block is performed.
    """
    if (
        actual.shape != expected.shape
        or actual.ndim < 2
        or not actual.size
        or actual.shape[-1] != actual.shape[-2]
    ):
        raise ValueError("densities must have identical nonempty (...,M,M) shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("densities must be finite")
    return float(np.max(np.abs(actual - expected)))


_COARSE = SolverTolerances(2e-10, 2e-12)
_FINE = SolverTolerances(2e-12, 2e-14)


def converged_reference(
    source: evolution.ModelSource,
    batch: evolution.TimelineBatch,
    *,
    coarse: SolverTolerances = _COARSE,
    fine: SolverTolerances = _FINE,
) -> ReferenceResult:
    """Solve a timeline at two tolerance levels and return the fine reference.

    The density is point-aligned in the DC eigenbasis. convergence_error is the
    maximum absolute density difference across all points and matrix entries.
    Reject an unconverged reference with AssertionError when that error exceeds
    2e-6. coarse and fine override local solver tolerances, not this threshold.
    UnsupportedCollapseOperators and other input errors from reference_timeline
    propagate.
    """
    approximate = reference_timeline(source, batch, rtol=coarse.rtol, atol=coarse.atol)
    density = reference_timeline(source, batch, rtol=fine.rtol, atol=fine.atol)
    error = density_error(approximate, density)
    assert error <= 2e-6, (
        f"reference did not converge: density error {error:.9g} > 2e-6"
    )
    return ReferenceResult(density, error)


class UnsupportedCollapseOperators(ValueError):
    """The independent secular reference cannot evolve an explicit jump."""


def require_secular_bath(model: evolution.SecularModel) -> None:
    """Raise UnsupportedCollapseOperators if the bath has explicit jumps."""
    if model.collapse_operators:
        raise UnsupportedCollapseOperators(
            "the reference solvers do not support explicit collapse operators "
            f"at flux level {model.level}"
        )


@dataclass(frozen=True)
class _Dynamics:
    hamiltonian: Callable[[float], NDArray[np.complex128]]
    bath: Callable[[float], evolution.SecularModel]
    basis: NDArray[np.complex128]
    boundaries: NDArray[np.float64]


def _lindblad_rhs(
    dynamics: _Dynamics,
    interval: tuple[float, float],
    bath: evolution.SecularModel,
) -> Callable[[float, NDArray[np.complex128]], NDArray[np.complex128]]:
    left, right = interval
    size = len(bath.energies_ghz)
    overlap = dynamics.basis.conj().T @ bath.eigenvectors
    rates = np.where(bath.rates_per_ns > 0, bath.rates_per_ns, 0.0)
    np.fill_diagonal(rates, 0.0)
    outgoing = rates.sum(axis=0)
    damping = bath.dephasing_per_ns + (outgoing[:, None] + outgoing[None, :]) / 2

    def rhs(time: float, flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        rho = flat.reshape(size, size)
        # Evaluate the one-sided generator at discontinuous interval ends.
        interior = min(max(time, np.nextafter(left, right)), np.nextafter(right, left))
        h = dynamics.hamiltonian(interior)
        derivative = -2j * np.pi * (h @ rho - rho @ h)
        bath_rho = overlap.conj().T @ rho @ overlap
        # Each secular jump |target><origin| feeds only target population;
        # its anticommutator damps each entry by half the two outgoing rates.
        dissipation = -damping * bath_rho
        dissipation[np.diag_indices(size)] += rates @ np.diag(bath_rho)
        return (derivative + overlap @ dissipation @ overlap.conj().T).ravel()

    return rhs


def _integrate(
    dynamics: _Dynamics, state: NDArray[np.complex128], *, rtol: float, atol: float
) -> NDArray[np.complex128]:
    intervals = list(
        zip(dynamics.boundaries[:-1], dynamics.boundaries[1:], strict=True)
    )
    baths = [dynamics.bath((left + right) / 2) for left, right in intervals]
    for model in baths or [dynamics.bath(dynamics.boundaries[0])]:
        require_secular_bath(model)

    result = state.copy().ravel()
    for (left, right), bath in zip(intervals, baths, strict=True):
        solution = solve_ivp(
            _lindblad_rhs(dynamics, (left, right), bath),
            (left, right),
            result,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=0.05,
        )
        assert solution.success, solution.message
        result = solution.y[:, -1]
    return result.reshape(state.shape)


def _boundaries(
    start: float, end: float, knots: list[NDArray[np.float64]]
) -> NDArray[np.float64]:
    times = np.unique(np.concatenate([np.array([start, end]), *knots]))
    return times[(times >= start) & (times <= end)]


def reference(
    case: KernelCase,
    row: int,
    state: NDArray[np.complex128],
    *,
    bath: evolution.FluxSchedule | None = None,
    rtol: float = 2e-12,
    atol: float = 2e-14,
) -> NDArray[np.complex128]:
    """Return one segment's final (M,M) density in that row's eigenbasis.

    case supplies source and SegmentTable; row selects a valid table row and
    state is its input density in the row eigenbasis. Optional bath overrides
    the row's constant bath with a ragged per-timeline FluxSchedule. Continuous
    H(flux) uses source.flux_expansion when table.flux_waveform is present.
    Stop at bath switches, flux samples and envelope knots, including nonzero
    envelope-local offsets. rtol/atol are DOP853 tolerances; defaults are the
    fine reference level. Explicit collapse operators in the selected bath raise
    UnsupportedCollapseOperators. Solver failure raises AssertionError.
    """
    table = case.table
    model = case.source.model(tuple(int(v) for v in table.level[row]))
    point = int(table.timeline[row])
    start = float(table.start_ns[row])
    end = start + float(table.duration_ns[row])
    knots = []
    waveform = table.flux_waveform
    if waveform is not None:
        first, last = waveform.offsets[point : point + 2]
        knots.append(waveform.time_ns[first:last])
    for tone in range(int(table.tones[row])):
        envelope = table.envelopes[table.tone_envelope[row, tone]]
        if envelope.kind == "sampled":
            knots.append(
                envelope.sample_ns + start - table.tone_envelope_offset_ns[row, tone]
            )
    if bath is None:

        def bath_at(time: float) -> evolution.SecularModel:
            return model
    else:
        first, last = bath.offsets[point : point + 2]
        bath_time = bath.start_ns[first:last]
        bath_levels = bath.level[first:last]
        knots.append(bath_time)

        def bath_at(time: float) -> evolution.SecularModel:
            index = np.searchsorted(bath_time, time, side="right") - 1
            return case.source.model(tuple(int(v) for v in bath_levels[index]))

    dynamics = _Dynamics(
        lambda time: _reference_hamiltonian(case, row, time, model),
        bath_at,
        model.eigenvectors,
        _boundaries(start, end, knots),
    )
    return _integrate(dynamics, state, rtol=rtol, atol=atol)


def _reference_hamiltonian(
    case: KernelCase, row: int, time: float, model: evolution.SecularModel
) -> NDArray[np.complex128]:
    table = case.table
    h = np.diag(model.energies_ghz).astype(complex)
    waveform = table.flux_waveform
    if waveform is not None:
        first, last = waveform.offsets[table.timeline[row] : table.timeline[row] + 2]
        fixed_h = _flux_hamiltonian(case.source, waveform, int(first), int(last), time)
        h = model.eigenvectors.conj().T @ fixed_h @ model.eigenvectors
    for tone in range(int(table.tones[row])):
        drive = model.drives[table.tone_port[row, tone]]
        elapsed = time - table.start_ns[row]
        phase = (
            table.tone_theta_rad[row, tone]
            + 2 * np.pi * table.tone_frequency_ghz[row, tone] * elapsed
        )
        envelope = table.envelopes[table.tone_envelope[row, tone]]
        amplitude = _envelope_value(
            envelope, elapsed + table.tone_envelope_offset_ns[row, tone]
        )
        z = table.tone_amplitude[row, tone] * amplitude * np.exp(1j * phase)
        h += z.real * drive.in_phase_ghz + z.imag * drive.quadrature_ghz
    return h


def _envelope_value(envelope: evolution.EnvelopeSpec, local_ns: float) -> complex:
    if envelope.kind == "constant":
        return 1 + 0j
    return complex(
        np.interp(local_ns, envelope.sample_ns, envelope.samples.real),
        np.interp(local_ns, envelope.sample_ns, envelope.samples.imag),
    )


def _flux_hamiltonian(
    source: evolution.ModelSource,
    waveform: evolution.FluxWaveform,
    first: int,
    last: int,
    time: float,
) -> NDArray[np.complex128]:
    expansion = source.flux_expansion()
    flux = np.array(
        [
            np.interp(
                time, waveform.time_ns[first:last], waveform.flux[first:last, loop]
            )
            for loop in range(waveform.flux.shape[1])
        ]
    )
    h = expansion.static.copy()
    for term in expansion.terms:
        angle = np.pi * flux[term.loop] * (2 if "2pi" in term.kind else 1)
        value = np.cos(angle) if term.kind.startswith("cos") else np.sin(angle)
        h += value * term.matrix
    return h


def reference_timeline(
    source: evolution.ModelSource,
    batch: evolution.TimelineBatch,
    *,
    rtol: float = 2e-12,
    atol: float = 2e-14,
) -> NDArray[np.complex128]:
    """Return point-aligned (n,M,M) final densities in the DC energy basis.

    Evolve source.initial_density or batch.initial_densities over each point's
    [start_ns,end_ns]. Tolerances go to full-Lindblad DOP853 without RWA.
    Stop at drive support, envelope knots, flux samples and bath switches.
    FluxSchedule selects instantaneous bath eigenbases and, without continuous
    samples, the Hamiltonian. With flux_waveform, source.flux_expansion defines
    H(flux) independently of the bath schedule. Continuations keep absolute
    carrier phase and pulse-local envelope time. SegmentShift inputs are not
    supported and raise NotImplementedError. Explicit collapse operators in any
    selected bath raise UnsupportedCollapseOperators. Solver failure raises AssertionError.
    """
    if batch.segment_shifts:
        raise NotImplementedError("reference does not support segment shifts")
    results = []
    for point in range(len(batch.points)):
        state = (
            source.initial_density
            if batch.initial_densities is None
            else batch.initial_densities[point]
        )
        dynamics = _timeline_dynamics(source, batch, point)
        results.append(_integrate(dynamics, state, rtol=rtol, atol=atol))
    return np.asarray(results, dtype=np.complex128)


def _timeline_dynamics(
    source: evolution.ModelSource, batch: evolution.TimelineBatch, point: int
) -> _Dynamics:
    dc = source.model((0,) * batch.flux.level.shape[1])
    first, last = batch.flux.offsets[point : point + 2]
    times = batch.flux.start_ns[first:last]
    levels = batch.flux.level[first:last]
    start = 0.0 if batch.start_ns is None else float(batch.start_ns[point])
    end = float(batch.end_ns[point])

    def bath_at(time: float) -> evolution.SecularModel:
        index = np.searchsorted(times, time, side="right") - 1
        return source.model(tuple(int(v) for v in levels[index]))

    def hamiltonian(time: float) -> NDArray[np.complex128]:
        model = bath_at(time)
        overlap = dc.eigenvectors.conj().T @ model.eigenvectors
        h = (overlap * model.energies_ghz) @ overlap.conj().T
        waveform = batch.flux_waveform
        if waveform is not None:
            wf_first, wf_last = waveform.offsets[point : point + 2]
            fixed_h = _flux_hamiltonian(
                source, waveform, int(wf_first), int(wf_last), time
            )
            h = dc.eigenvectors.conj().T @ fixed_h @ dc.eigenvectors
        for track in batch.tracks:
            elapsed = time - track.start_ns[point]
            if not track.active[point] or not 0 <= elapsed < track.duration_ns[point]:
                continue
            drive = next(d for d in model.drives if d.name == track.port)
            envelope = track.envelopes[track.envelope_index[point]]
            phase = (
                2 * np.pi * track.frequency_ghz[point] * time + track.phase_rad[point]
            )
            z = (
                track.amplitude[point]
                * _envelope_value(envelope, elapsed)
                * np.exp(1j * phase)
            )
            h += (
                overlap
                @ (z.real * drive.in_phase_ghz + z.imag * drive.quadrature_ghz)
                @ overlap.conj().T
            )
        return h

    knots = [times]
    if batch.segment_boundaries:
        knots.append(np.asarray(batch.segment_boundaries[point], dtype=float))
    if batch.flux_waveform is not None:
        wf_first, wf_last = batch.flux_waveform.offsets[point : point + 2]
        knots.append(batch.flux_waveform.time_ns[wf_first:wf_last])
    for track in batch.tracks:
        if track.active[point]:
            origin = track.start_ns[point]
            knots.append(np.array([origin, origin + track.duration_ns[point]]))
            envelope = track.envelopes[track.envelope_index[point]]
            if envelope.kind == "sampled":
                knots.append(origin + envelope.sample_ns)
    return _Dynamics(
        hamiltonian, bath_at, dc.eigenvectors, _boundaries(start, float(end), knots)
    )
