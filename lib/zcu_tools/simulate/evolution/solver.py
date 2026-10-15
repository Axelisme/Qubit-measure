"""Joint direct evolution with event geometry hidden behind one public call."""

from dataclasses import dataclass
from operator import index
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .controls import ControlPulse, ControlSequence, quantize_ns
from .lindblad import LindbladAction
from .model import NumericModel, SolverSettings, hermitian_matrix
from .reuse import EvaluationReceipt, exact_key
from .state import ComplexMatrix, DensityState, owned_matrix

if TYPE_CHECKING:
    from .library import EvolutionLibrary, LibrarySnapshot


def propagator_power(propagator: ComplexMatrix, repetitions: int) -> ComplexMatrix:
    """Return U**N by binary matrix multiplication; N=0 returns identity.

    U is a finite square complex matrix, copied, never modified. N must be a
    nonnegative Python/NumPy integer (not bool), otherwise ValueError. This tool
    composes identical closed-system propagators, not dissipative evolution.
    It does not certify unitarity or normalize the result.
    """
    try:
        n = index(repetitions)
    except TypeError as error:
        raise ValueError("repetitions must be a nonnegative integer") from error
    if isinstance(repetitions, (bool, np.bool_)) or n < 0:
        raise ValueError("repetitions must be a nonnegative integer")
    base = owned_matrix(propagator, "propagator").copy()
    result = np.eye(base.shape[0], dtype=np.complex128)
    while n:
        if n & 1:
            result = result @ base
        n >>= 1
        if n:
            base = base @ base
    if not np.isfinite(result).all():
        raise RuntimeError("propagator power produced nonfinite values")
    return result


@dataclass(frozen=True)
class _Interval:
    controls: ControlSequence
    pulses: tuple[ControlPulse, ...]
    start: float
    end: float

    def time(self, t: float) -> float:
        # Evaluate the left limit at the right endpoint: later frame events
        # belong to the next interval, not the last Runge-Kutta stage here.
        return min(t, float(np.nextafter(self.end, self.start)))


class Evolution:
    """Reusable pure-numeric model/solver owner, without hidden quantum memory.

    model: fixed-coordinate NumericModel, reusable and never modified.
    settings: internal algorithm accuracy controls; caller does not choose a
        solver separately for every event interval.
    library: optional explicitly pathed deterministic storage with 4GB host LRU.
    tables: optional frozen publication snapshot; otherwise captured at construction.
        Only static closed plateau intervals use U tables. Other legal requests
        retain the full direct solver, with persistent exact density records.
    last_evaluation: author-only receipt of the last successful evolve, including
        actual content identities and full load/solve/write wall time.
    Every evolve call requires its own initial density. Independent shots must
    explicitly supply their own initial state. Explicit shot fragments may feed
    the previous result to the next call with the complete absolute controls.
    """

    def __init__(
        self,
        model: NumericModel,
        settings: SolverSettings | None = None,
        *,
        library: "EvolutionLibrary | None" = None,
        tables: "LibrarySnapshot | None" = None,
    ):
        self.model = model
        self.settings = SolverSettings() if settings is None else settings
        self.library = library
        self.tables = (
            tables
            if tables is not None
            else (library.snapshot() if library is not None else None)
        )
        self.last_evaluation = EvaluationReceipt("direct", (), 0)
        self._sources = {s.name: s for s in model.sources}

    def evolve(
        self,
        initial: DensityState,
        controls: ControlSequence | None = None,
        *,
        start_ns: float = 0.0,
        end_ns: float,
    ) -> DensityState:
        """Evolve an explicit density over an absolute nearest-2-ns interval.

        initial must already use model.basis; source/target conversion is not
        implicit. controls contains full carrier/frame history, including before
        start_ns. start_ns/end_ns are rounded from common origin zero, ties even.
        Return a new owned DensityState in model.basis; inputs are unchanged.
        Source overlap evolves jointly, not as serial single-source rotations.
        ValueError reports invalid inputs/callbacks; RuntimeError reports solver
        failure or invalid output. No eigenvalue clipping or trace repair occurs.
        """
        started = perf_counter()
        controls = ControlSequence() if controls is None else controls
        self._validate_inputs(initial, controls)
        points = controls.boundaries_ns(start_ns=start_ns, end_ns=end_ns)
        key = ""
        identities: list[str] = []
        if self.library is not None:
            key = exact_key(
                self.model, self.settings, initial, controls, (points[0], points[-1])
            )
            record = self.library.read_record(key)
            if record is not None:
                state = DensityState(self.model.basis, record[0])
                self.last_evaluation = EvaluationReceipt(
                    "direct-record", (record[1],), perf_counter() - started
                )
                self.library.last_evaluation = self.last_evaluation
                return state
        points = tuple(
            sorted(
                set(points)
                | {
                    q
                    for t in self.model.event_times_ns
                    if points[0] <= (q := quantize_ns(t)) <= points[-1]
                }
            )
        )
        density = initial.density.copy()
        for start, end in zip(points[:-1], points[1:], strict=True):
            middle = (start + end) / 2
            pulses = tuple(
                p
                for p in controls.pulses
                if quantize_ns(p.start_ns) <= middle < quantize_ns(p.end_ns)
            )
            interval = _Interval(controls, pulses, start, end)
            if self.model.collapse_operators or self.model.collapse_at is not None:
                density = self._dissipative(density, interval)
            else:
                cached = None
                if self.library is not None and self.tables is not None:
                    cached = self.library.plateau(
                        self.model, controls, (start, end), self.tables
                    )
                if cached is None:
                    u = self._closed(interval)
                else:
                    u, identity = cached
                    identities.append(identity)
                density = u @ density @ u.conj().T
        try:
            state = DensityState(self.model.basis, density)
        except ValueError as error:
            raise RuntimeError(
                f"evolution produced invalid density: {error}"
            ) from error
        if self.library is not None and not identities:
            identities.append(self.library.save_record(key, state.density))
            origin = "direct"
        else:
            origin = "table" if identities else "direct"
        self.last_evaluation = EvaluationReceipt(
            origin, tuple(dict.fromkeys(identities)), perf_counter() - started
        )
        if self.library is not None:
            self.library.last_evaluation = self.last_evaluation
        return state

    def _validate_inputs(
        self, initial: DensityState, controls: ControlSequence
    ) -> None:
        basis = self.model.basis
        if (
            initial.basis.identity != basis.identity
            or initial.basis.labels != basis.labels
            or not np.array_equal(initial.basis.vectors, basis.vectors)
        ):
            raise ValueError("initial density must use the model basis")
        names = {p.source for p in controls.pulses} | {
            e.source for e in controls.phase_events
        }
        if names - self._sources.keys():
            raise ValueError(
                f"unbound numeric sources: {sorted(names - self._sources.keys())}"
            )

    def _hamiltonian(self, t: float, interval: _Interval) -> ComplexMatrix:
        t = interval.time(t)
        h = self.model.hamiltonian_ghz.copy()
        if self.model.hamiltonian_at is not None:
            extra = hermitian_matrix(
                self.model.hamiltonian_at(t), "dynamic Hamiltonian"
            )
            if extra.shape != h.shape:
                raise ValueError("dynamic Hamiltonian shape must match basis")
            h += extra
        for pulse in interval.pulses:
            z = interval.controls.amplitude(pulse, t)
            source = self._sources[pulse.source]
            h += z.real * source.in_phase_ghz
            if source.quadrature_ghz is not None:
                h += z.imag * source.quadrature_ghz
        return h

    def _step(self, interval: _Interval) -> float:
        frequencies = [abs(p.frequency_ghz) for p in interval.pulses]
        carrier_limit = 1 / (16 * max(frequencies)) if any(frequencies) else np.inf
        return min(self.settings.max_step_ns, carrier_limit)

    def _integrate(
        self, rhs, initial: ComplexMatrix, interval: _Interval
    ) -> ComplexMatrix:
        solution = solve_ivp(
            rhs,
            (interval.start, interval.end),
            initial.ravel(),
            method="DOP853",
            rtol=self.settings.rtol,
            atol=self.settings.atol,
            max_step=self._step(interval),
        )
        if not solution.success or not np.isfinite(solution.y).all():
            raise RuntimeError(f"evolution solver failed: {solution.message}")
        return solution.y[:, -1].reshape(initial.shape)

    def _unitary(self, interval: _Interval) -> ComplexMatrix:
        n = len(self.model.basis.labels)

        def rhs(t, flat):
            return (
                -2j * np.pi * self._hamiltonian(t, interval) @ flat.reshape(n, n)
            ).ravel()

        return self._integrate(rhs, np.eye(n, dtype=complex), interval)

    def _period(self, interval: _Interval) -> float | None:
        if self.model.hamiltonian_at is not None or any(
            p.envelope.constant is None for p in interval.pulses
        ):
            return None
        frequencies = [abs(p.frequency_ghz) for p in interval.pulses if p.frequency_ghz]
        if not frequencies:
            return None
        fundamental = min(frequencies)
        # Only exact integer ratios qualify. Never perturb the supplied carrier
        # to manufacture periodicity for almost-commensurate sources.
        if any(not (f / fundamental).is_integer() for f in frequencies):
            return None
        return 1 / fundamental

    def _closed(self, interval: _Interval) -> ComplexMatrix:
        if self.model.hamiltonian_at is None and all(
            p.envelope.constant is not None and p.frequency_ghz == 0
            for p in interval.pulses
        ):
            return np.asarray(
                expm(
                    -2j
                    * np.pi
                    * self._hamiltonian(interval.start, interval)
                    * (interval.end - interval.start)
                ),
                dtype=np.complex128,
            )
        period = self._period(interval)
        if period is None or interval.end - interval.start < period:
            return self._unitary(interval)
        count = int((interval.end - interval.start) // period)
        first = _Interval(
            interval.controls, interval.pulses, interval.start, interval.start + period
        )
        u = propagator_power(self._unitary(first), count)
        tail_start = interval.start + count * period
        if tail_start < interval.end:
            tail = _Interval(
                interval.controls, interval.pulses, tail_start, interval.end
            )
            u = self._unitary(tail) @ u
        return u

    def _dissipative(
        self, density: ComplexMatrix, interval: _Interval
    ) -> ComplexMatrix:
        n = density.shape[0]
        static = LindbladAction(n)
        static.replace(self.model.collapse_operators)
        dynamic = LindbladAction(n)

        def rhs(t, flat):
            h = self._hamiltonian(t, interval)
            rho = flat.reshape(n, n)
            derivative = -2j * np.pi * (h @ rho - rho @ h) + static.apply(rho)
            if self.model.collapse_at is not None:
                dynamic.replace(self.model.collapse_at(interval.time(t)))
                derivative += dynamic.apply(rho)
            return derivative.ravel()

        return self._integrate(rhs, density, interval)
