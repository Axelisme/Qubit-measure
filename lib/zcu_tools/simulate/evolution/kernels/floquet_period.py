"""Constant-tone Hilbert propagators with compact secular Strang stepping."""

from copy import copy
from dataclasses import dataclass
from threading import RLock
from typing import Literal, NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import SecularModel
from ..workplan import WorkPlan
from . import _period_sambe
from ._cf4_grid import coarse_count, finer_count, spectral_width
from ._dop853 import Tolerances, integrate_dense
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
class _Period:
    # DOP853 coefficients, CF4 prefixes, or a constant-tone Sambe spectrum.
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
class _Periods:
    entries: tuple[_Period, ...]

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
    # DOP853's <= 1e-10 controls and CF4's Hermitian exponentials keep the
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
    gain: float
    period: _Period


class _GainAnchor:
    """One lazily built independent table and its construction lock."""

    def __init__(self, references: int) -> None:
        self.remaining = references
        self.period: _Period | None = None
        self.lock = RLock()


class _GainGroup:
    """Fixed anchors for one model object, port, signed carrier and budget.

    Each anchor has its own build lock. Inventory changes use the workspace
    lock; byte readers never wait for numerical construction. A planned key
    releases its references only after handing out its result, once per key.
    Repeated keys deterministically rebuild any already released anchors.
    """

    def __init__(
        self,
        model: SecularModel,
        port: int,
        frequency: float,
        context: KernelContext,
        builders: dict[float, PeriodBuilder],
        inventory_lock: RLock,
    ) -> None:
        self.work_plan = context.work_plan
        self.model = model
        self.port = port
        self.frequency = frequency
        self.budget = context.budget
        self.builders = {g: b for g, b in builders.items() if b.anchors}
        plan = {g: b.anchors for g, b in self.builders.items()}
        self.plan = plan
        coordinates = {a for dependencies in plan.values() for a in dependencies}
        self.anchors = {
            a: _GainAnchor(sum(a in dependencies for dependencies in plan.values()))
            for a in coordinates
        }
        self.delivered: set[float] = set()
        self.inventory_lock = inventory_lock
        self.attempts = 0
        self.hits = 0

    @property
    def nbytes(self) -> int:
        # Caller holds inventory_lock. Active payloads charge their own arrays.
        return sum(
            a.period.nbytes for a in self.anchors.values() if a.period is not None
        )

    def anchor(self, gain: float) -> _Period:
        """Build an independent anchor once while planned references remain."""
        anchor = self.anchors[gain]
        with anchor.lock:
            with self.inventory_lock:
                period = anchor.period
            if period is None:
                period = _build(
                    self.model,
                    _Tone(self.port, self.frequency, gain),
                    self.budget,
                    self.builders[gain],
                    work_plan=self.work_plan,
                )
                with self.inventory_lock:
                    anchor.period = period
            return period

    def build(self, gain: float) -> _Period:
        """Return a key using only its fixed anchors, then release dependencies."""
        if gain in self.anchors:
            result = self.anchor(gain)
        else:
            result = _build(
                self.model,
                _Tone(self.port, self.frequency, gain),
                self.budget,
                self.builders[gain],
                group=self,
                work_plan=self.work_plan,
            )
        with self.inventory_lock:
            if gain not in self.delivered:
                self.delivered.add(gain)
                for coordinate in self.plan[gain]:
                    self.anchors[coordinate].remaining -= 1
            for coordinate in self.plan[gain]:
                anchor = self.anchors[coordinate]
                if anchor.remaining == 0:
                    anchor.period = None
        return result

    def prefixes(
        self, context: _Hamiltonian, gain: float
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Interpolate nearest planned anchors and independently check coarse CF4."""
        neighbours = [_Neighbour(a, self.anchor(a)) for a in self.plan[gain]]
        tolerance = min(1e-4, self.budget * 0.1)
        continued = _continued_prefixes(neighbours, context, gain, tolerance)
        if continued is not None:
            times, coefficients, hit = continued
            with self.inventory_lock:
                self.attempts += 1
                self.hits += int(hit)
            return times, coefficients
        return _magnus_build(*context, tolerance)


class _GainWorkspace:
    """One evolve's fixed gain-anchor plans and numeric-byte inventory.

    Planning uses the complete table, never workers, chunks or request order.
    Runtime groups isolate model object, port, signed carrier and budget.
    attempts/hits count checked/accepted interpolants. Spectral candidates are
    removed before constructing any anchor or reference count. sambe_builds and
    sambe_fallbacks count accepted and declined spectral prepare requests,
    including repeated keys, never unused candidates or LRU cache hits.
    nbytes charges retained
    anchors, conservatively including arrays held by active payloads too.
    No numeric data survives the owning evolve.
    """

    def __init__(self, table: SegmentTable) -> None:
        # Import lazily: dispatch knows the numerical builder types, while
        # standalone kernel calls still use that same decision authority.
        from .._dispatch import period_gain_coordinates

        self._gains = period_gain_coordinates(table)
        self._builders: dict[tuple[int, int, float], dict[float, PeriodBuilder]] = {}
        self._groups: dict[tuple[int, int, float, float], _GainGroup] = {}
        self._lock = RLock()
        self._sambe_builds = 0
        self._sambe_fallbacks = 0

    @property
    def sambe_builds(self) -> int:
        """Number of prepare keys accepted by Sambe in this evolve scope."""
        with self._lock:
            return self._sambe_builds

    @property
    def sambe_fallbacks(self) -> int:
        """Attempted Sambe keys declined to independent CF4 after cutoff checks."""
        with self._lock:
            return self._sambe_fallbacks

    @property
    def attempts(self) -> int:
        with self._lock:
            return sum(group.attempts for group in self._groups.values())

    @property
    def hits(self) -> int:
        with self._lock:
            return sum(group.hits for group in self._groups.values())

    @property
    def nbytes(self) -> int:
        """Current retained numeric bytes, excluding Python/model overhead."""
        with self._lock:
            return sum(group.nbytes for group in self._groups.values())

    def build(
        self,
        model: SecularModel,
        port: int,
        frequency: float,
        gain: float,
        context: KernelContext,
    ) -> _Period:
        """Use key-local Sambe or fixed CF4 anchors, independent of scheduling.

        Rejected Sambe candidates use independent CF4 and never gain anchors.
        Other keys retain the complete table's fixed non-spectral gain plan.
        """
        from .._dispatch import period_builders

        budget = context.budget
        builder_identity = (id(model), port, frequency)
        with self._lock:
            builders = self._builders.get(builder_identity)
            if builders is None:
                gains = self._gains.get((model.level, port, frequency), (gain,))
                builders = period_builders(model, port, frequency, gains)
                self._builders[builder_identity] = builders
            if gain not in builders:
                builders.update(period_builders(model, port, frequency, (gain,)))
            builder = builders[gain]
        if builder.kind == "sambe":
            result = _build(
                model,
                _Tone(port, frequency, gain),
                budget,
                builder,
                work_plan=context.work_plan,
            )
            with self._lock:
                self._sambe_builds += int(result.spectrum is not None)
                self._sambe_fallbacks += int(result.spectrum is None)
            return result
        if not builder.anchors:
            return _build(
                model,
                _Tone(port, frequency, gain),
                budget,
                builder,
                work_plan=context.work_plan,
            )
        identity = (id(model), port, frequency, budget)
        with self._lock:
            group = self._groups.get(identity)
            if group is None:
                group = _GainGroup(
                    model, port, frequency, context, builders, self._lock
                )
                self._groups[identity] = group
        return group.build(gain)


def _continued_prefixes(
    neighbours: list[_Neighbour],
    context: _Hamiltonian,
    gain: float,
    tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.complex128], bool] | None:
    # Interpolate only gain between fixed anchors in the interaction picture, then
    # independently check the new key with the original coarse-grid probes.
    h0, x, y, frequency = context
    count = len(neighbours[-1].period.times) - 1
    if any(
        n.period.magnus is None or len(n.period.times) != count + 1 for n in neighbours
    ):
        return None
    coordinates = [n.gain for n in neighbours]
    target = gain
    if len(set(coordinates)) != len(coordinates):
        return None
    period = 1 / abs(frequency)
    width = spectral_width(h0, x, y)
    coarse_size = coarse_count(1.5 * period * width)
    if count < finer_count(coarse_size):
        return None
    times = np.linspace(0.0, period, count + 1)
    # Identical model, carrier and uniform grid give every predecessor the
    # same analytic H0 phase. The interaction-picture transform and inverse
    # cancel algebraically in this linear combination; avoid repeated complex
    # exponentials and their roundoff without changing the predictor.
    weights = []
    for index in range(len(neighbours)):
        weight = 1.0
        for other, coordinate in enumerate(coordinates):
            if other != index:
                weight *= (target - coordinate) / (coordinates[index] - coordinate)
        weights.append(weight)
    candidate = _interpolate_prefixes(
        tuple(n.period.coefficients for n in neighbours), tuple(weights)
    )
    _, coarse = _magnus_prefixes(h0, x, y, frequency, coarse_size)
    error = float(
        np.max(
            [
                np.linalg.norm(candidate[j * count // 8] - coarse[j * coarse_size // 8])
                for j in range(9)
            ]
        )
    )
    if not np.isfinite(error) or error > tolerance:
        times, fine = _magnus_refine(context, tolerance, coarse_size, coarse)
        return times, fine, False
    return times, candidate, True


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
    group: _GainGroup | None = None,
) -> _Period:
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
        if group is None:
            times, coefficients = _magnus_build(h0, x, y, frequency, tolerance)
        else:
            times, coefficients = group.prefixes(
                _Hamiltonian(h0, x, y, frequency), gain
            )
        final = coefficients[-1]
        magnus = (h0, x, y, frequency)
        for array in (h0, x, y):
            array.setflags(write=False)
    else:
        tolerance = min(1e-10, budget * 1e-4)
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
    return _Period(
        *arrays, split_rate(model, port, frequency, gain, work_plan), magnus, spectrum
    )


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
def _dissipate(rho, population, coherence):
    diagonal = np.diag(rho).copy()
    result = rho * coherence
    updated = population @ diagonal
    for index in range(len(updated)):
        result[index, index] = updated[index]
    return result


@njit(nogil=True, cache=True)
def _advance(rho, unitary, population, coherence, count):
    adjoint = unitary.conj().T.copy()
    for _ in range(count):
        rho = _dissipate(rho, population, coherence)
        rho = unitary @ rho @ adjoint
        rho = _dissipate(rho, population, coherence)
    return rho


@njit(nogil=True, cache=True)
def _advance_segments(rho, unitaries, population, coherence, count):
    for _ in range(count):
        for unitary in unitaries:
            rho = _advance(rho, unitary, population, coherence, 1)
    return rho


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
    result = _advance(rho, unitary, entry.population_half, entry.coherence_half, count)
    if tail > 0:
        end = _dense_fundamental(
            entry.times, entry.coefficients, entry.unitary, origin + tail
        )
        result = _advance(result, end @ start.conj().T, tail_bath[0], tail_bath[1], 1)
    return result


def _fundamental(entry: _Period, time: float) -> NDArray[np.complex128]:
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
    entry: _Period,
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
    result = _advance_segments(rho, unitaries, population, coherence, count)
    tail = duration - count * period
    whole = min(int(tail // step), subdivisions - 1)
    result = _advance_segments(result, unitaries[:whole], population, coherence, 1)
    residual = tail - whole * step
    if residual > 0:
        unitary = _fundamental(entry, origin + tail) @ boundaries[whole].conj().T
        result = _advance(
            result,
            unitary,
            expm(entry.generator * residual / 2).astype(complex),
            np.exp(-entry.decay * residual / 2),
            1,
        )
    return result


def _row(
    entry: _Period,
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
    result = _advance(rho, unitary, entry.population_half, entry.coherence_half, count)
    if tail > 0:
        end = origin + tail
        tail_unitary = _fundamental(entry, end)
        tail_unitary = tail_unitary @ start.conj().T
        result = _advance(
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

    Higher carriers retain DOP853 with min(1e-10,budget*1e-4), max_step=T/4.
    One Newton polar step corrects endpoints before translation and composition.
    Invalid budgets/representatives/payloads raise ValueError; integration or
    nonfinite output raises RuntimeError without rerouting. nbytes charges all
    retained arrays, which remain immutable across reordered/split apply calls.
    """

    name = "floquet_period"
    guard_builds = False

    def __init__(self, workspace: _GainWorkspace | None = None) -> None:
        """Use an evolve-local scope, or independent builds for standalone calls."""
        self._workspace = workspace

    def evolve_scope(
        self, table: SegmentTable, context: KernelContext
    ) -> tuple["FloquetPeriodKernel", _GainWorkspace]:
        """Plan fixed gain anchors in a fresh byte-counted workspace.

        table is the complete segmented scan; context supplies this evolve's model
        source, budget and WorkPlan. Numeric tables are independent of worker count, chunking and
        request order for this table/source/budget. No density is read.
        Engine shares the scoped kernel with prefix
        and timeline workers and drops both returned objects when evolve ends.
        Standalone prepare uses the same cost-selected Sambe or independent CF4
        builder, without scope counters or gain-anchor sharing.
        """
        del context
        workspace = _GainWorkspace(table)
        # Preserve injected subclass guards/apply hooks without mutating the
        # registry instance shared by concurrent evolves.
        scoped = copy(self)
        scoped._workspace = workspace
        return scoped, workspace

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
        guard. Scoped calls reuse the complete gain plan; standalone calls ask
        dispatch for independent builders without gain-anchor participation.
        """
        source, budget = context.source, context.budget
        _check_budget(budget)
        if len(keys) != len(rows) or not np.array_equal(
            keys, self.prepare_keys(table, rows, context)
        ):
            raise ValueError("Floquet period representatives must match keys")
        if not self.guard(table, rows, context).all():
            raise ValueError("Floquet period representatives must pass guard")
        from .._dispatch import period_builder

        entries = []
        for key in keys:
            model = source.model(tuple(int(v) for v in key["level"]))
            port, frequency, gain = (
                int(key["port"]),
                float(key["frequency"]),
                float(key["gain"]),
            )
            if self._workspace is None:
                builder = period_builder(model, port, frequency, gain)
                entry = _build(
                    model,
                    _Tone(port, frequency, gain),
                    budget,
                    builder,
                    work_plan=context.work_plan,
                )
            else:
                entry = self._workspace.build(model, port, frequency, gain, context)
            entries.append(entry)
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
