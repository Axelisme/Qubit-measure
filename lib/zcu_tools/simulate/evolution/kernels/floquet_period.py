"""Constant-tone Hilbert propagators with compact secular Strang stepping."""

from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import SecularModel
from ..workplan import ResourceHandle, WorkPlan
from . import _period_sambe
from ._cf4_grid import coarse_count, finer_count, spectral_width
from ._cf4_spectral import CF4Nodes, build_nodes, spectral_prefixes
from ._dop853 import Tolerances, integrate_dense
from ._period_channel import advance, advance_segments
from ._secular import secular_operators
from ._workload_facts import split_rate

_MIN_FREQUENCY_GHZ = 0.03
_MAX_SPLIT_ESTIMATE = 1.0
_MAX_SUBDIVISIONS = 64


@dataclass(frozen=True)
class PeriodBuilder:
    """Dispatch-owned instructions for one constant-tone period key.

    kind selects Sambe, CF4 or dense DOP853. work_limit is the original CF4
    coarse/first-fine N-cubed estimate, passed unchanged to the Sambe guard.
    anchors lists fixed CF4 dependencies, including self for an independent
    anchor; empty means standalone, with no gain sharing. Sambe rejection
    always falls back to standalone CF4 regardless of these dependencies.
    """

    kind: Literal["sambe", "cf4", "dop853"]
    work_limit: float = 0.0
    anchors: tuple[float, ...] = ()


class _Tone(NamedTuple):
    # Model-local drive coordinates shared by independent and anchored builds.
    port: int
    frequency: float
    gain: float


class _DensePeriod(NamedTuple):
    # Numeric-only view for compiled high-carrier endpoint composition and steps.
    times: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    unitary: NDArray[np.complex128]
    population_half: NDArray[np.complex128]
    coherence_half: NDArray[np.float64]


@dataclass(frozen=True)
class PeriodData:
    """Immutable constant-tone trajectory and secular playback arrays.

    times/coefficients encode DOP853 dense output or CF4 prefixes; unitary is
    U(T). generator/decay and their half-period maps implement secular steps.
    split_rate bounds the required subdivisions. magnus stores the CF4
    Hamiltonian (h0, x, y, signed frequency), or None for other builders.
    spectrum holds a Sambe trajectory instead of dense coefficients when used.
    nbytes charges every retained numeric array, excluding Python overhead.
    """

    times: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    unitary: NDArray[np.complex128]
    generator: NDArray[np.float64]
    decay: NDArray[np.float64]
    population_half: NDArray[np.complex128]
    coherence_half: NDArray[np.float64]
    split_rate: float
    magnus: (
        tuple[
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            float,
        ]
        | None
    )

    spectrum: _period_sambe.Spectrum | None = None

    @property
    def nbytes(self) -> int:
        return (
            (self.spectrum.nbytes if self.spectrum is not None else 0)
            + sum(
                array.nbytes
                for array in (
                    self.times,
                    self.coefficients,
                    self.unitary,
                    self.generator,
                    self.decay,
                    self.population_half,
                    self.coherence_half,
                )
            )
            + (sum(a.nbytes for a in self.magnus[:3]) if self.magnus is not None else 0)
        )


@dataclass(frozen=True)
class PeriodAnchor:
    """Evolve-local anchor product, never stored as a resident payload.

    period is the independently built CF4 trajectory and its playback data.
    interaction holds exp(+2*pi*i*H0*t) U(t) only for carrier interpolation;
    gain anchors omit it because their identical H0 phases cancel algebraically.
    WorkPlan resources own and release this product after consumer handoff.
    """

    period: PeriodData
    interaction: NDArray[np.complex128] | None = None


@dataclass(frozen=True)
class _Periods:
    entries: tuple[PeriodData, ...]

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries)


def _check_budget(budget: float) -> None:
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("Floquet period budget must be finite and positive")


def _subdivisions(estimate: float) -> int:
    return max(1, int(np.ceil(np.sqrt(estimate / _MAX_SPLIT_ESTIMATE))))


@njit(nogil=True, cache=True)
def _rhs(time, flat, context):
    h0, x, y, frequency = context
    size = len(h0)
    phase = 2 * np.pi * frequency * time
    h = h0 + np.cos(phase) * x + np.sin(phase) * y
    return (-2j * np.pi * (h @ flat.reshape(size, size))).ravel()


@njit(nogil=True, cache=True)
def _integrate_dense(context, state, duration, controls):
    # A callback-free entry point caches the linked shared integrator across
    # processes; the RHS dispatcher itself has a process-specific cache key.
    return integrate_dense(_rhs, context, state, duration, controls)


@njit(nogil=True, cache=True)
def _unitary(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Retract a near-unitary Hilbert endpoint with one Newton polar step."""
    # DOP853's <= 1e-8 controls and CF4's Hermitian exponentials keep the
    # singular values near one. Newton squares that norm defect without an SVD.
    return 0.5 * (3 * matrix - matrix @ (matrix.conj().T @ matrix))


class _Hamiltonian(NamedTuple):
    """One key's lab-frame Hamiltonian in GHz, before multiplication by 2*pi.

    h0 is the diagonal static matrix with common energy zero removed; x and y
    are the gain-scaled Hermitian drive quadratures; frequency is signed GHz.
    """

    h0: NDArray[np.complex128]
    x: NDArray[np.complex128]
    y: NDArray[np.complex128]
    frequency: float


@dataclass(frozen=True)
class _Neighbour:
    coordinate: float
    period: PeriodData
    interaction: NDArray[np.complex128] | None = None


@dataclass(frozen=True)
class _Continuation:
    # Target-key interpolation supports and its optional shared coarse guard.
    neighbours: list[_Neighbour]
    coarse_nodes: ResourceHandle[CF4Nodes] | None = None


def _continued_prefixes(
    continuation: _Continuation,
    context: _Hamiltonian,
    gain: float,
    tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]] | None:
    # Interpolate gain or carrier between fixed anchors in the interaction picture, then
    # independently check the new key with the original coarse-grid probes.
    neighbours = continuation.neighbours
    h0, x, y, frequency = context
    count = len(neighbours[-1].period.times) - 1
    if any(
        n.period.magnus is None or len(n.period.times) != count + 1 for n in neighbours
    ):
        return None
    coordinates = [n.coordinate for n in neighbours]
    carrier_axis = neighbours[0].interaction is not None
    target = frequency if carrier_axis else gain
    if len(set(coordinates)) != len(coordinates):
        return None
    period = 1 / abs(frequency)
    width = spectral_width(h0, x, y)
    coarse_size = coarse_count(1.5 * period * width)
    if count < finer_count(coarse_size):
        return None
    times = np.linspace(0.0, period, count + 1)
    # Gain anchors have identical model, carrier and uniform grid. Their H0
    # phases cancel algebraically, avoiding repeated exponentials and roundoff.
    # Carrier anchors instead require the explicit IP transform below.
    weights = []
    for index in range(len(neighbours)):
        weight = 1.0
        for other, coordinate in enumerate(coordinates):
            if other != index:
                weight *= (target - coordinate) / (coordinates[index] - coordinate)
        weights.append(weight)
    if carrier_axis:
        interactions = tuple(
            n.interaction for n in neighbours if n.interaction is not None
        )
        if len(interactions) != len(neighbours):
            return None
        candidate = _interpolate_prefixes(interactions, tuple(weights))
        # Equal phase q means different physical times at different carriers.
        # Restore this key's lab frame only after interpolating IP trajectories.
        candidate *= np.exp(
            -2j * np.pi * times[:, None, None] * np.diag(h0)[None, :, None]
        )
    else:
        candidate = _interpolate_prefixes(
            tuple(n.period.coefficients for n in neighbours), tuple(weights)
        )
    return _guarded_prefixes(
        context, times, candidate, tolerance, coarse_size, continuation.coarse_nodes
    )


def _guarded_prefixes(
    context: _Hamiltonian,
    times: NDArray[np.float64],
    candidate: NDArray[np.complex128],
    tolerance: float,
    coarse_size: int,
    coarse_nodes: ResourceHandle[CF4Nodes] | None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    # Keep the coarse numerical certificate and its independent refinement
    # together; shared spectral nodes alter neither the probes nor tolerance.
    h0, x, y, frequency = context
    count = len(candidate) - 1
    if coarse_nodes is None:
        _, coarse = _magnus_prefixes(h0, x, y, frequency, coarse_size)
    else:
        nodes = coarse_nodes.get(
            partial(build_nodes, h0, x, y, float(np.sign(frequency)), coarse_size)
        )
        coarse = spectral_prefixes(
            nodes.values, nodes.outgoing, nodes.overlaps, nodes.transitions, frequency
        )
    error = float(
        np.max(
            [
                np.linalg.norm(candidate[j * count // 8] - coarse[j * coarse_size // 8])
                for j in range(9)
            ]
        )
    )
    if coarse_nodes is not None:
        # The spectral recurrence changes only guard rounding. Near the decision
        # boundary, retain the original Padé comparison. Rejected predictions
        # also refine from the original coarse array to keep fallback bitwise.
        roundoff = 128 * np.finfo(float).eps * coarse_size * len(h0)
        if not np.isfinite(error) or error >= tolerance - roundoff:
            _, coarse = _magnus_prefixes(h0, x, y, frequency, coarse_size)
            if np.isfinite(error) and abs(error - tolerance) <= roundoff:
                error = max(
                    np.linalg.norm(
                        candidate[j * count // 8] - coarse[j * coarse_size // 8]
                    )
                    for j in range(9)
                )
    if not np.isfinite(error) or error > tolerance:
        times, fine = _magnus_refine(context, tolerance, coarse_size, coarse)
        return times, fine
    return times, candidate


@njit(nogil=True, cache=True)
def _interpolate_prefixes(prefixes, weights):
    # Fuse the weighted sum without a table-sized temporary for each anchor.
    # Anchor order matches the scalar Lagrange sum; no fast-math reassociation.
    candidate = np.zeros_like(prefixes[0])
    for index in range(len(weights)):
        for step in range(candidate.shape[0]):
            for row in range(candidate.shape[1]):
                for column in range(candidate.shape[2]):
                    candidate[step, row, column] += (
                        weights[index] * prefixes[index][step, row, column]
                    )
    return candidate


def _build(
    model: SecularModel,
    tone: _Tone,
    budget: float,
    builder: PeriodBuilder,
    *,
    work_plan: WorkPlan,
    neighbours: _Continuation | None = None,
) -> PeriodData:
    port, frequency, gain = tone
    size = len(model.energies_ghz)
    period = 1 / abs(frequency)
    drive = model.drives[port]
    # Removing the common energy zero only changes an unobservable global phase.
    h0 = np.diag(model.energies_ghz - model.energies_ghz[0]).astype(complex)
    x, y = gain * drive.in_phase_ghz, gain * drive.quadrature_ghz
    magnus = None
    spectrum = None
    spectral = (
        _period_sambe.build(model, port, frequency, gain, budget, builder.work_limit)
        if builder.kind == "sambe"
        else None
    )
    if spectral is not None:
        spectrum, final = spectral
        times = np.array([0.0, period])
        coefficients = np.empty((0, 0, size, size), dtype=complex)
    elif builder.kind != "dop853":
        # Low carriers make Schrödinger DOP853 resolve many unobservable H0 turns.
        # CF4 exponentiates H0 exactly; only carrier motion needs approximation.
        tolerance = min(1e-4, budget * 0.1)
        continued = (
            _continued_prefixes(
                neighbours,
                _Hamiltonian(h0, x, y, frequency),
                gain,
                tolerance,
            )
            if neighbours is not None
            else None
        )
        if continued is None:
            times, coefficients = _magnus_build(h0, x, y, frequency, tolerance)
        else:
            times, coefficients = continued
        final = coefficients[-1]
        magnus = (h0, x, y, frequency)
        for array in (h0, x, y):
            array.setflags(write=False)
    else:
        # Keep budget scaling for tighter requests. The cap is calibrated for
        # population and coherent-projection playback through 10 us.
        tolerance = min(1e-8, budget * 1e-4)
        times, coefficients, final = _integrate_dense(
            (h0, x, y, frequency),
            np.eye(size, dtype=complex).ravel(),
            period,
            Tolerances(tolerance, tolerance * 0.01, period / 4),
        )
        coefficients = coefficients.reshape(-1, 8, size, size)
    generator, decay = secular_operators(model)
    arrays = (
        times,
        coefficients,
        _unitary(final.reshape(size, size)),
        generator,
        decay,
        expm(generator * period / 2).astype(complex),
        np.exp(-decay * period / 2),
    )
    for array in arrays:
        if not np.isfinite(array).all():
            raise RuntimeError("Floquet period preparation produced nonfinite data")
        array.setflags(write=False)
    return PeriodData(
        *arrays, split_rate(model, port, frequency, gain, work_plan), magnus, spectrum
    )


def _build_anchor(
    model: SecularModel, tone: _Tone, context: KernelContext, *, carrier_axis: bool
) -> PeriodAnchor:
    period = _build(
        model, tone, context.budget, PeriodBuilder("cf4"), work_plan=context.work_plan
    )
    if not carrier_axis:
        return PeriodAnchor(period)
    energies = model.energies_ghz - model.energies_ghz[0]
    interaction = (
        np.exp(2j * np.pi * period.times[:, None, None] * energies[None, :, None])
        * period.coefficients
    )
    return PeriodAnchor(period, interaction)


def _dependencies_reusable(
    dependencies: list[tuple[float, ResourceHandle[PeriodAnchor]]], consumer: float
) -> bool:
    """Return whether every dependency can still serve this consumer."""
    return bool(dependencies) and all(
        not handle.released
        and (
            consumer in handle.remaining_consumers
            or (not handle.sealed and consumer not in handle.actual_consumers)
        )
        for _, handle in dependencies
    )


def _prepare_entry(
    model: SecularModel, tone: _Tone, context: KernelContext
) -> PeriodData:
    # Both interpolation axes share admission, single-flight construction and
    # release. Only their physical coordinates and IP transform differ.
    from .._gain_plan import (
        admit_carrier_nodes,
        build_plan,
        carrier_anchor_resource,
        carrier_build_plan,
        gain_anchor_resource,
        period_builder,
    )

    port, frequency, gain = tone
    planned = build_plan(context, model, port, frequency, gain)
    carrier = (
        carrier_build_plan(context, model, port, frequency, gain)
        if planned is None
        else None
    )
    builder = (
        period_builder(model, port, frequency, gain) if planned is None else planned[1]
    )
    consumer = frequency if carrier is not None else gain
    if carrier is not None:
        coordinates = (
            (frequency,) if frequency in carrier.anchors_ghz else carrier.anchors_ghz
        )
        dependencies = [
            (a, carrier_anchor_resource(context, carrier, a)) for a in coordinates
        ]
    elif planned is not None:
        dependencies = [
            (a, gain_anchor_resource(context, planned[0], a)) for a in builder.anchors
        ]
    else:
        dependencies = []
    # Repeated keys never reopen a handed-off or released dependency.
    reusable = _dependencies_reusable(dependencies, consumer)
    if not reusable:
        return _build(model, tone, context.budget, builder, work_plan=context.work_plan)
    for _, handle in dependencies:
        if not handle.sealed:
            handle.admit(consumer)
    neighbours = []
    for coordinate, handle in dependencies:
        anchor_tone = (
            _Tone(port, coordinate, gain)
            if carrier is not None
            else _Tone(port, frequency, coordinate)
        )
        anchor = handle.get(
            partial(
                _build_anchor,
                model,
                anchor_tone,
                context,
                carrier_axis=carrier is not None,
            )
        )
        neighbours.append(_Neighbour(coordinate, anchor.period, anchor.interaction))
    coarse_nodes = (
        admit_carrier_nodes(context, carrier, frequency)
        if carrier is not None
        else None
    )
    entry = (
        neighbours[0].period
        if len(dependencies) == 1 and dependencies[0][0] == consumer
        else _build(
            model,
            tone,
            context.budget,
            builder,
            work_plan=context.work_plan,
            neighbours=_Continuation(neighbours, coarse_nodes),
        )
    )
    if coarse_nodes is not None:
        coarse_nodes.handoff(consumer)
    for _, handle in dependencies:
        handle.handoff(consumer)
    return entry


@njit(nogil=True, cache=True)
def _hermitian_exponential(h, step):
    # Scaling and squaring with the [7/7] Pade approximant. Its double-
    # precision backward-error radius is 0.9504178996162932. Centering the
    # spectrum reduces scaling without changing the returned global phase.
    size = len(h)
    center = np.trace(h).real / size
    identity = np.eye(size, dtype=np.complex128)
    matrix = (-2j * np.pi * step) * (h - center * identity)
    norm = np.max(np.sum(np.abs(matrix), axis=0))
    squarings = max(0, int(np.ceil(np.log2(max(norm / 0.9504178996162932, 1.0)))))
    matrix = matrix / 2.0**squarings
    square = matrix @ matrix
    fourth = square @ square
    sixth = fourth @ square
    odd = matrix @ (8648640.0 * identity + 277200.0 * square + 1512.0 * fourth + sixth)
    even = 17297280.0 * identity + 1995840.0 * square + 25200.0 * fourth + 56.0 * sixth
    result = np.ascontiguousarray(np.linalg.solve(even - odd, even + odd))
    for _ in range(squarings):
        result = result @ result
    return np.exp(-2j * np.pi * step * center) * result


@njit(nogil=True, cache=True)
def _magnus_step(h0, x, y, frequency, time, step):
    # Fourth-order commutator-free Magnus, two Gauss nodes and two exponentials.
    a = (3 - 2 * np.sqrt(3)) / 12
    b = (3 + 2 * np.sqrt(3)) / 12
    first = 2 * np.pi * frequency * (time + (0.5 - np.sqrt(3) / 6) * step)
    last = 2 * np.pi * frequency * (time + (0.5 + np.sqrt(3) / 6) * step)
    h1 = h0 + np.cos(first) * x + np.sin(first) * y
    h2 = h0 + np.cos(last) * x + np.sin(last) * y
    return _hermitian_exponential(a * h1 + b * h2, step) @ _hermitian_exponential(
        b * h1 + a * h2, step
    )


@njit(nogil=True, cache=True)
def _magnus_prefixes(h0, x, y, frequency, count):
    period = 1 / abs(frequency)
    times = np.linspace(0.0, period, count + 1)
    prefixes = np.empty((count + 1, len(h0), len(h0)), dtype=np.complex128)
    prefixes[0] = np.eye(len(h0), dtype=np.complex128)
    for j in range(count):
        prefixes[j + 1] = (
            _magnus_step(h0, x, y, frequency, times[j], period / count) @ prefixes[j]
        )
    return times, prefixes


@njit(nogil=True, cache=True)
def _magnus_build(h0, x, y, frequency, tolerance):
    period = 1 / abs(frequency)
    # Keep both grids away from a full spectral turn, where sampled carrier
    # motion can alias. The retained grid stays below half a spectral turn.
    width = spectral_width(h0, x, y)
    count = coarse_count(1.5 * period * width)
    _, coarse = _magnus_prefixes(h0, x, y, frequency, count)
    return _magnus_refine(_Hamiltonian(h0, x, y, frequency), tolerance, count, coarse)


@njit(nogil=True, cache=True)
def _magnus_refine(context, tolerance, count, coarse):
    h0, x, y, frequency = context
    # A rejected continuation has already built the independent coarse grid.
    for _ in range(12):
        finer = finer_count(count)
        times, fine = _magnus_prefixes(h0, x, y, frequency, finer)
        error = 0.0
        # Compare micromotion at eight phase intervals, not just U(T), where
        # errors can cancel. Both grids include these endpoints exactly.
        for j in range(9):
            error = max(
                error, np.linalg.norm(coarse[j * count // 8] - fine[j * finer // 8])
            )
        if error <= tolerance:
            return times, fine
        count = finer
        coarse = fine
    raise RuntimeError("Floquet Magnus preparation failed to converge")


@njit(nogil=True, cache=True)
def _dense(times, coefficients, time):
    step = min(np.searchsorted(times, time, side="right") - 1, len(times) - 2)
    step = max(0, step)
    scaled = (time - times[step]) / (times[step + 1] - times[step])
    result = np.zeros_like(coefficients[step, 0])
    for index in range(7):
        result += coefficients[step, 7 - index]
        if index % 2 == 0:
            result *= scaled
        else:
            result *= 1 - scaled
    return coefficients[step, 0] + result


@njit(nogil=True, cache=True)
def _dense_fundamental(times, coefficients, unitary, time):
    period = times[-1]
    local = time % period
    if local == 0:
        return unitary if time >= period else np.eye(len(unitary), dtype=np.complex128)
    value = _unitary(_dense(times, coefficients, local))
    if time >= period:
        value = value @ unitary
    return value


@njit(nogil=True, cache=True)
def _dense_row(entry, origin, duration, rho, tail_bath):
    # Keep endpoint playback, polar correction and all density steps in one
    # nogil call. Tiny dispatcher calls otherwise contend on the worker GIL.
    start = _dense_fundamental(entry.times, entry.coefficients, entry.unitary, origin)
    unitary = start @ entry.unitary @ start.conj().T
    period = entry.times[-1]
    count = int(duration // period)
    tail = duration - count * period
    result = advance(rho, unitary, entry.population_half, entry.coherence_half, count)
    if tail > 0:
        end = _dense_fundamental(
            entry.times, entry.coefficients, entry.unitary, origin + tail
        )
        result = advance(result, end @ start.conj().T, tail_bath[0], tail_bath[1], 1)
    return result


def _fundamental(entry: PeriodData, time: float) -> NDArray[np.complex128]:
    if entry.magnus is None and entry.spectrum is None:
        return _dense_fundamental(entry.times, entry.coefficients, entry.unitary, time)
    period = float(entry.times[-1])
    local = time % period
    if local == 0:
        # Exact boundaries need neither dense playback nor polar correction.
        return (
            entry.unitary
            if time >= period
            else np.eye(len(entry.unitary), dtype=complex)
        )
    if entry.spectrum is not None:
        value = _unitary(_period_sambe.endpoint(entry.spectrum, local))
        return value @ entry.unitary if time >= period else value
    index = min(
        int(np.searchsorted(entry.times, local, side="right")) - 1,
        len(entry.times) - 2,
    )
    assert entry.magnus is not None
    h0, x, y, frequency = entry.magnus
    value = (
        _magnus_step(
            h0, x, y, frequency, entry.times[index], local - entry.times[index]
        )
        @ entry.coefficients[index]
    )
    value = _unitary(value)
    if time >= period:
        value = value @ entry.unitary
    return value


def _subdivided_row(
    entry: PeriodData,
    origin: float,
    duration: float,
    rho: NDArray[np.complex128],
    subdivisions: int,
) -> NDArray[np.complex128]:
    period = float(entry.times[-1])
    step = period / subdivisions
    boundaries = [
        _fundamental(entry, origin + j * step) for j in range(subdivisions + 1)
    ]
    unitaries = np.array(
        [
            end @ start.conj().T
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    )
    population = expm(entry.generator * step / 2).astype(complex)
    coherence = np.exp(-entry.decay * step / 2)
    count = int(duration // period)
    result = advance_segments(rho, unitaries, population, coherence, count)
    tail = duration - count * period
    whole = min(int(tail // step), subdivisions - 1)
    result = advance_segments(result, unitaries[:whole], population, coherence, 1)
    residual = tail - whole * step
    if residual > 0:
        unitary = _fundamental(entry, origin + tail) @ boundaries[whole].conj().T
        result = advance(
            result,
            unitary,
            expm(entry.generator * residual / 2).astype(complex),
            np.exp(-entry.decay * residual / 2),
            1,
        )
    return result


def _row(
    entry: PeriodData,
    frequency: float,
    theta: float,
    duration: float,
    rho: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    if duration == 0:
        return rho
    period = float(entry.times[-1])
    origin = (theta / (2 * np.pi * frequency)) % period
    subdivisions = _subdivisions(duration * entry.split_rate)
    if subdivisions > 1:
        return _subdivided_row(entry, origin, duration, rho, subdivisions)
    if entry.magnus is None and entry.spectrum is None:
        tail = duration - int(duration // period) * period
        tail_bath = (entry.population_half, entry.coherence_half)
        if tail > 0:
            tail_bath = (
                expm(entry.generator * tail / 2).astype(complex),
                np.exp(-entry.decay * tail / 2),
            )
        dense = _DensePeriod(
            entry.times,
            entry.coefficients,
            entry.unitary,
            entry.population_half,
            entry.coherence_half,
        )
        return _dense_row(dense, origin, duration, rho, tail_bath)
    start = _fundamental(entry, origin)
    unitary = start @ entry.unitary @ start.conj().T
    count = int(duration // period)
    tail = duration - count * period
    result = advance(rho, unitary, entry.population_half, entry.coherence_half, count)
    if tail > 0:
        end = origin + tail
        tail_unitary = _fundamental(entry, end)
        tail_unitary = tail_unitary @ start.conj().T
        result = advance(
            result,
            tail_unitary,
            expm(entry.generator * tail / 2).astype(complex),
            np.exp(-entry.decay * tail / 2),
            1,
        )
    return result


class FloquetPeriodKernel:
    """One constant-tone Hilbert channel for arbitrary, possibly coherent inputs.

    Keys contain level, port, signed frequency in GHz and signed gain, never
    density, duration or start. Immutable payloads retain U(T) and endpoint data.
    Theta translates time by theta/(2*pi*f) mod T. Subperiods and fractional tails
    use the same trajectory with secular D/2,U,D/2 steps; density is not normalized.
    The density-free guard requires |f| >= 0.03 GHz and the smallest m in 1..64
    satisfying duration*T**2*Gamma*V**2/m**2 <= 1.0. T=1/|f| is in ns,
    V=2*pi*|gain|*(||X||_2+||Y||_2) in rad/ns, and Gamma is the larger population
    generator norm/coherence decay in 1/ns. Rejected rows still try T6.

    Dispatch selects the builder and fixed gain dependencies. Below 1 GHz, Sambe competes with CF4's original coarse count c and first
    1.5x finer count d, both rounded to eight. Cutoffs 2/3 cost
    0.5*(5**3+7**3)=234 units versus c+d. The common Hilbert dimension cubed
    cancels. This single-threaded 16-level calibration never consults scheduling.
    Adjacent cutoffs compare physical U(q*T), q=0,1/8,...,1, with Frobenius defect
    <= min(1e-4,budget*0.1). The finer spectrum supplies arbitrary endpoints.
    Nonfinite data or sampled unitarity defect above 1e-4 decline. Refinement
    stops at cutoff 12 or before cumulative 0.5*(2*m+1)**3 cost exceeds c+d.
    Rejection preserves independent CF4 bitwise, without gain-anchor sharing.
    This calibrated check assumes cutoff convergence and representative samples,
    not a rigorous all-time error bound. Polar correction and dissipation persist.

    Non-spectral low carriers retain CF4 and scoped gain anchors, as in the README.

    Unsubdivided 16-level rows with at least 4096 complete periods apply the
    same D/2,U,D/2 linear channel by real-matrix binary powering. Two real
    coordinate vectors preserve complex density residuals without symmetrizing.
    This changes rounding, not splitting or integration tolerances. All other
    rows retain sequential half-bath/GEMM order with reusable scratch buffers.
    Power matrices are transient (about 1.1 MiB per worker), never cached.

    Higher carriers use DOP853 with min(1e-8,budget*1e-4), max_step=T/4.
    This cap is calibrated through 10 us, not an arbitrary-duration error bound.
    One Newton polar step corrects endpoints before translation and composition.
    Invalid budgets/representatives/payloads raise ValueError; integration or
    nonfinite output raises RuntimeError without rerouting. nbytes charges all
    retained arrays, which remain immutable across reordered/split apply calls.
    """

    name = "floquet_period"
    guard_builds = False

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.bool_]:
        """Return row-local constant-tone, carrier and splitting-estimate validity."""
        source, budget = context.source, context.budget
        _check_budget(budget)
        accepted = np.zeros(len(rows), dtype=np.bool_)
        for index, row in enumerate(rows):
            if table.tones[row] != 1:
                continue
            envelope = table.envelopes[int(table.tone_envelope[row, 0])]
            frequency = float(table.tone_frequency_ghz[row, 0])
            if envelope.kind != "constant" or abs(frequency) < _MIN_FREQUENCY_GHZ:
                continue
            model = source.model(tuple(int(v) for v in table.level[row]))
            estimate = table.duration_ns[row] * split_rate(
                model,
                int(table.tone_port[row, 0]),
                frequency,
                float(table.tone_amplitude[row, 0]),
                context.work_plan,
            )
            accepted[index] = estimate <= _MAX_SPLIT_ESTIMATE * _MAX_SUBDIVISIONS**2
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Encode level/port/frequency/gain, excluding duration, start and theta."""
        del context
        keys = np.empty(
            len(rows),
            dtype=[
                ("level", "<i8", table.level.shape[1]),
                ("port", "<i8"),
                ("frequency", "<f8"),
                ("gain", "<f8"),
            ],
        )
        keys["level"] = table.level[rows]
        if len(rows):
            if np.any(table.tones[rows] != 1):
                raise ValueError("Floquet period keys require one tone")
            keys["port"] = table.tone_port[rows, 0]
            keys["frequency"] = table.tone_frequency_ghz[rows, 0]
            keys["gain"] = table.tone_amplitude[rows, 0]
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate density steps, proportional to duration times carrier frequency."""
        return 1 + table.duration_ns[rows] * np.abs(table.tone_frequency_ghz[rows, 0])

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build immutable periods with dispatch-selected builders/work ceilings.

        keys and rows are aligned representatives that must pass the scalar
        guard. Contexts carrying planner records reuse fixed gain dependencies;
        standalone WorkPlans use independent builders without gain sharing.
        """
        source, budget = context.source, context.budget
        _check_budget(budget)
        if len(keys) != len(rows) or not np.array_equal(
            keys, self.prepare_keys(table, rows, context)
        ):
            raise ValueError("Floquet period representatives must match keys")
        if not self.guard(table, rows, context).all():
            raise ValueError("Floquet period representatives must pass guard")
        entries = []
        for key in keys:
            model = source.model(tuple(int(v) for v in key["level"]))
            port, frequency, gain = (
                int(key["port"]),
                float(key["frequency"]),
                float(key["gain"]),
            )
            entries.append(_prepare_entry(model, _Tone(port, frequency, gain), context))
        return _Periods(tuple(entries))

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """Decline cached-domain admission; use the ordinary guard build path."""
        del prepared, table, rows, context

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """Decline handoff: the scalar guard builds no propagator."""
        source, budget = context.source, context.budget
        del table, rows, source, budget

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> None:
        """Decline input binding: this channel applies to arbitrary densities."""
        budget = context.budget
        del prepared, key_index, table, rows, source_initial, budget

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Advance aligned states in place; return all False without a posterior check.

        Uses only table duration, signed frequency and effective start theta.
        No refinement, normalization, retry or dispatch occurs here. Prepared
        data remain read-only across reordered/split calls and worker threads.
        """
        context.require_no_plan()
        budget = context.budget
        _check_budget(budget)
        if not isinstance(prepared, _Periods):
            raise ValueError("Floquet period requires its own prepared payload")
        for index, row in enumerate(rows):
            states[index] = _row(
                prepared.entries[int(key_index[index])],
                float(table.tone_frequency_ghz[row, 0]),
                float(table.tone_theta_rad[row, 0]),
                float(table.duration_ns[row]),
                states[index],
            )
        if not np.isfinite(states).all():
            raise RuntimeError("Floquet period apply produced nonfinite density")
        return np.zeros(len(rows), dtype=np.bool_)
