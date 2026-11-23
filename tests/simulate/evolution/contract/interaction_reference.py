"""Independent exact interaction-picture reference for fixed-DC timelines."""

from collections.abc import Callable

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from zcu_tools.simulate.evolution import ModelSource, SecularModel, TimelineBatch

from .reference import ReferenceResult, SolverTolerances, require_secular_bath

_COARSE = SolverTolerances(2e-10, 2e-12)
_FINE = SolverTolerances(2e-12, 2e-14)


def interaction_reference(
    source: ModelSource,
    batch: TimelineBatch,
    *,
    coarse: SolverTolerances = _COARSE,
    fine: SolverTolerances = _FINE,
    max_step_ns: float = 0.05,
) -> ReferenceResult:
    """Solve full Lindblad evolution without RWA and verify two-tolerance agreement.

    source supplies static secular H0 energies (GHz), named drive quadratures,
    rates and pure dephasing (/ns), and the initial density in the zero-level
    DC energy basis. batch supplies point-aligned absolute ns, GHz carriers,
    radian phases and dimensionless complex envelopes. Each shot must have one
    schedule level, which may be nonzero and may differ between points.
    Initial densities/start_ns, when supplied, continue in the same DC basis;
    carrier phase remains absolute and envelope time remains pulse-local.

    Return ReferenceResult with fine density (points,M,M) in the caller's DC
    energy basis and the maximum absolute coarse/fine density difference.
    coarse/fine specify DOP853 rtol/atol; defaults are 2e-10/2e-12 and
    2e-12/2e-14. max_step_ns is the positive step ceiling for both solves;
    long sampled inputs can use a larger ceiling while retaining adaptive
    tolerances. Integrations stop at envelope knots, drive support and explicit
    segment boundaries. The exact H0 interaction transform drops no drive terms;
    secular jumps and elementwise dephasing are invariant under this transform.
    The final state is rotated back to the lab frame and DC energy basis.

    Raise ValueError for any continuous flux waveform (even constant), multiple
    schedule levels within a shot, or segment shifts. Waveforms need a separate
    Hamiltonian/bath-basis treatment, outside this static-schedule oracle.
    Explicit collapse operators in a selected bath raise UnsupportedCollapseOperators.
    Raise AssertionError when the two densities differ by more than 2e-6;
    tolerances never override this threshold. Raise RuntimeError on solver
    failure. ModelSource model-building and missing-port errors propagate.
    No acquisition, production propagation, clipping or normalization runs.
    """
    approximate = interaction_density(source, batch, coarse, max_step_ns=max_step_ns)
    density = interaction_density(source, batch, fine, max_step_ns=max_step_ns)
    error = float(np.max(np.abs(approximate - density), initial=0.0))
    if error > 2e-6:
        raise AssertionError(
            f"reference did not converge: density error {error:.9g} > 2e-6"
        )
    return ReferenceResult(density, error)


def interaction_density(
    source: ModelSource,
    batch: TimelineBatch,
    tolerances: SolverTolerances,
    *,
    max_step_ns: float = 0.05,
) -> NDArray[np.complex128]:
    """Integrate one tolerance chain and return all final DC-basis densities.

    Use the same full-drive static-schedule inputs, units, initial-state rules
    and failure modes as interaction_reference, except there is no convergence
    check. tolerances supplies DOP853 rtol/atol; max_step_ns is its positive step
    ceiling in ns. Return complex128 densities (points,M,M) without clipping or
    normalization; the caller owns source.
    To continue, supply these matrices as TimelineBatch.initial_densities with
    each previous end_ns as the new start_ns. Keep track origins and phase at
    the common absolute origin unchanged. Check convergence only after both
    independent tolerance chains have finished the original timeline.
    """
    if batch.flux_waveform is not None:
        raise ValueError(
            "continuous flux waveform is unsupported: a single schedule level is required"
        )
    if batch.segment_shifts:
        raise ValueError(
            "segment shifts are unsupported by the static-schedule reference"
        )
    for point in range(len(batch.points)):
        first, last = batch.flux.offsets[point : point + 2]
        levels = batch.flux.level[first:last]
        if np.any(levels != levels[0]):
            raise ValueError("DC flux changes within shot")
    dc = source.model((0,) * batch.flux.level.shape[1])
    models = [
        source.model(tuple(int(value) for value in batch.flux.level[first]))
        for first in batch.flux.offsets[:-1]
    ]
    for model in models:
        require_secular_bath(model)
    results = []
    for point, model in enumerate(models):
        overlap = dc.eigenvectors.conj().T @ model.eigenvectors
        initial = (
            source.initial_density
            if batch.initial_densities is None
            else batch.initial_densities[point]
        )
        initial = overlap.conj().T @ initial @ overlap
        density = _solve_point(model, initial, batch, point, tolerances, max_step_ns)
        results.append(overlap @ density @ overlap.conj().T)
    return np.asarray(results, dtype=np.complex128)


def _solve_point(
    model: SecularModel,
    initial: NDArray[np.complex128],
    batch: TimelineBatch,
    point: int,
    tolerances: SolverTolerances,
    max_step_ns: float,
) -> NDArray[np.complex128]:
    size = len(model.energies_ghz)
    gaps = model.energies_ghz[:, None] - model.energies_ghz[None, :]
    end = float(batch.end_ns[point])
    start = 0.0 if batch.start_ns is None else float(batch.start_ns[point])
    state = (initial * np.exp(2j * np.pi * gaps * start)).ravel()
    boundaries = _point_boundaries(batch, point)
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        solution = solve_ivp(
            _interval_rhs(model, batch, point, float(left), float(right)),
            (left, right),
            state,
            method="DOP853",
            rtol=tolerances.rtol,
            atol=tolerances.atol,
            max_step=max_step_ns,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.y[:, -1]
    return state.reshape(size, size) * np.exp(-2j * np.pi * gaps * end)


def _point_boundaries(batch: TimelineBatch, point: int) -> NDArray[np.float64]:
    end = float(batch.end_ns[point])
    start = 0.0 if batch.start_ns is None else float(batch.start_ns[point])
    knots = [start, end]
    if batch.segment_boundaries:
        knots.extend(batch.segment_boundaries[point])
    for track in batch.tracks:
        if not track.active[point]:
            continue
        origin = track.start_ns[point]
        knots.extend((origin, origin + track.duration_ns[point]))
        envelope = track.envelopes[track.envelope_index[point]]
        if envelope.kind == "sampled":
            knots.extend(origin + envelope.sample_ns)
    values = np.unique(knots)
    return values[(values >= start) & (values <= end)]


def _interval_rhs(
    model: SecularModel,
    batch: TimelineBatch,
    point: int,
    left: float,
    right: float,
) -> Callable[[float, NDArray[np.complex128]], NDArray[np.complex128]]:
    size = len(model.energies_ghz)
    gaps = model.energies_ghz[:, None] - model.energies_ghz[None, :]
    outgoing = model.rates_per_ns.sum(axis=0)
    damping = model.dephasing_per_ns + (outgoing[:, None] + outgoing[None, :]) / 2
    bath = np.stack((model.rates_per_ns, damping))
    tones, drives = [], []
    for track in batch.tracks:
        origin = track.start_ns[point]
        if (
            not track.active[point]
            or not 0 <= (left + right) / 2 - origin < track.duration_ns[point]
        ):
            continue
        drive = next(d for d in model.drives if d.name == track.port)
        envelope = track.envelopes[track.envelope_index[point]]
        values = np.ones(2, complex)
        if envelope.kind == "sampled":
            times = np.array([left, right]) - origin
            values = np.interp(
                times, envelope.sample_ns, envelope.samples.real
            ) + 1j * np.interp(times, envelope.sample_ns, envelope.samples.imag)
        values *= track.amplitude[point] * np.exp(1j * track.phase_rad[point])
        # Every envelope is linear on this interval because all knots are stops.
        tones.append(
            (
                2 * np.pi * track.frequency_ghz[point],
                values[0],
                (values[1] - values[0]) / (right - left),
                left,
            )
        )
        drives.append((drive.in_phase_ghz, drive.quadrature_ghz))
    tone_data = np.asarray(tones, dtype=np.complex128).reshape(-1, 4)
    drive_data = np.asarray(drives, dtype=np.complex128).reshape(-1, 2, size, size)

    def rhs(time: float, flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        interior = min(max(time, np.nextafter(left, right)), np.nextafter(right, left))
        return _interaction_derivative(
            interior, flat, gaps, bath, tone_data, drive_data
        )

    return rhs


@njit(cache=True)
def _interaction_derivative(
    time: float,
    flat: NDArray[np.complex128],
    gaps: NDArray[np.float64],
    bath: NDArray[np.float64],
    tones: NDArray[np.complex128],
    drives: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    # Packed arrays keep Python objects outside the independent compiled RHS.
    # bath[0/1] = rates/damping; tones = angular carrier, envelope intercept,
    # envelope slope, interval origin; drives[:,0/1] = in-phase/quadrature.
    size = gaps.shape[0]
    rho = flat.reshape((size, size))
    h = np.zeros((size, size), np.complex128)
    for tone in range(tones.shape[0]):
        omega, value, slope, left = tones[tone]
        z = (value + slope * (time - left.real)) * np.exp(1j * omega.real * time)
        for i in range(size):
            for j in range(size):
                h[i, j] += (
                    z.real * drives[tone, 0, i, j] + z.imag * drives[tone, 1, i, j]
                )
    for i in range(size):
        for j in range(size):
            h[i, j] *= np.exp(2j * np.pi * gaps[i, j] * time)
    derivative = np.empty((size, size), np.complex128)
    for i in range(size):
        for j in range(size):
            commutator = 0j
            for k in range(size):
                commutator += h[i, k] * rho[k, j] - rho[i, k] * h[k, j]
            value = -2j * np.pi * commutator - bath[1, i, j] * rho[i, j]
            if i == j:
                for k in range(size):
                    value += bath[0, i, k] * rho[k, k]
            derivative[i, j] = value
    return derivative.ravel()
