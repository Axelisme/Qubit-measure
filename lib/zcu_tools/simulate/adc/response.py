"""Joint state-conditioned first moments for a shared two-resonator line."""

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .model import ReadoutModel, RealArray, readonly_array
from .readout_batch import ConditionalSums, ReadoutBatch
from .request import ConditionalResponse, ReadoutPulse, SampleGrid


class ConditionalReadout:
    """Own reusable numeric line/rate matrices, without shot or quantum memory.

    model: pure numeric readout/rate/calibration model, never mutated.
    rtol/atol: positive finite DOP853 local error tolerances, not a physical
        convergence guarantee. Readout origin is the earliest pulse start_ns,
        or zero for no pulses. Initial populations refer to that origin, where
        fields start in vacuum. Earlier lookback samples have zero physical
        field and do not evolve populations; receiver offset/noise still apply.
    """

    def __init__(
        self, model: ReadoutModel, *, rtol: float = 1e-10, atol: float = 1e-12
    ):
        if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
            raise ValueError("readout tolerances must be positive finite")
        self.model = model
        self.rtol, self.atol = rtol, atol
        # Keep pulse references alive: arbitrary envelope callbacks are keyed by
        # identity because the pulse contract exposes no serializable content.
        self._responses: dict[
            tuple[object, ...], tuple[tuple[ReadoutPulse, ...], tuple[RealArray, ...]]
        ] = {}
        self._v = np.sqrt(model.kappa_external_per_ns / 2)
        self._scattering = self._v * np.exp(-1j * model.mismatch_phase_rad)
        self._q = model.rates_per_ns - np.diag(model.rates_per_ns.sum(axis=0))
        detuning = model.frequencies_ghz - model.reference_frequency_ghz
        self._a = np.array(
            [
                np.diag(model.kappa_internal_per_ns / 2 + 2j * np.pi * f)
                + np.outer(self._v, self._v)
                for f in detuning
            ]
        )
        self._constant_line = _ConstantLine(self._q, self._a, self._v)

    def responses(self, batch: ReadoutBatch) -> ConditionalSums:
        """Return noiseless per-shot sums for each content/read/initial state.

        batch supplies the DC model, finite pulses and grids in ns/GHz/radians.
        Its model defines the state order; no flux tail or overlapping controls
        enter this calculation. Cache each distinct (model identity, pulses,
        grids) response for this instance's lifetime. Keys include the full
        numeric model and receiver content, not just its identity string, plus
        pulse parameters and exact grid times/demodulation. Arbitrary envelope
        callbacks are held alive and keyed by callable identity; callers must
        keep them deterministic. Equivalent separately constructed numeric inputs
        reuse the cache. Traces are retained internally, so switching include_traces
        does not solve again. Constant pulses (envelope=None) use the closed-form
        matrix exponential in the pulse carrier frame, including detuned carriers.
        Frequency-independent population and moment propagators are shared across
        contents; ill-conditioned driven blocks use the full matrix exponential.
        Other envelopes use the existing DOP853 moment ODE.
        ConditionalSums.sums has shape (contents, reads, states, 2), after
        physical delay/demodulation/conjugation/gain/phase/offset. Traces have shape
        (contents, reads, states, samples, 2) only if batch.include_traces;
        otherwise traces is None. No initial-state or noise draw occurs.
        Inputs are not mutated; populations are never clipped or normalized.
        Raise ValueError for incompatible grids/sample counts or invalid
        callbacks; RuntimeError for solver failure or invalid final populations.
        All contents must have the same sample counts per read. Dense traces
        also require equal sample counts across reads. The existing single-grid
        response continues to use DOP853.
        """
        counts = np.array([len(g.times_ns) for g in batch.grids[0]], dtype=np.int64)
        if any(
            [len(g.times_ns) for g in grids] != counts.tolist() for grids in batch.grids
        ):
            raise ValueError("sample counts must match across grid contents")
        if batch.include_traces and np.any(counts != counts[0]):
            raise ValueError("dense traces require equal sample counts across reads")
        model_key = _model_key(batch.model)
        solver = (
            self
            if batch.model is self.model
            else ConditionalReadout(batch.model, rtol=self.rtol, atol=self.atol)
        )
        contents = []
        for pulses, grids in zip(batch.pulses, batch.grids, strict=True):
            key = (model_key, _pulse_key(pulses), _grid_key(grids))
            if key not in self._responses:
                times = np.unique(np.concatenate([g.times_ns for g in grids]))
                constant = all(p.envelope is None for p in pulses)
                outgoing, _ = solver._fields(pulses, times, constant=constant)
                traces = tuple(
                    readonly_array(
                        solver._calibrate(
                            outgoing[np.searchsorted(times, g.times_ns)], g
                        ),
                        "conditional traces",
                    )
                    for g in grids
                )
                self._responses[key] = (pulses, traces)
            contents.append(self._responses[key][1])
        sums = np.stack(
            [np.stack([trace.sum(axis=-2) for trace in traces]) for traces in contents]
        )
        dense = (
            np.stack([np.stack(traces) for traces in contents])
            if batch.include_traces
            else None
        )
        return ConditionalSums(sums, counts, dense)

    def response(
        self, pulses: tuple[ReadoutPulse, ...], grid: SampleGrid
    ) -> ConditionalResponse:
        """Solve all initial-state conditional mean fields once on the ADC grid.

        pulses are finite coherent readout drives in ns/GHz. grid owns physical
        times and demodulation. Return conditional traces and final populations
        in model.state_labels order, including pre-pulse/ringup/ringdown delayed
        by receiver.time_of_flight_ns. Final populations refer to the emitted
        time of the last ADC sample, no earlier than readout origin. No
        initial-state or receiver-noise draws occur. State-conditioned moments
        obey dm_s=ΣQ_st*m_t-A_s*m_s+v*b*p_s, not a mean-frequency field equation.
        ValueError signals invalid input/callbacks; RuntimeError signals failed
        integration or invalid probability/moment output. No density Hilbert
        space, jump-time sampling, clipping or renormalization is introduced.
        """
        outgoing, state = self._fields(pulses, grid.times_ns, constant=False)
        iq = self._calibrate(outgoing, grid)
        return ConditionalResponse(self.model.state_labels, iq, state[:, 0, :].real.T)

    def _fields(
        self, pulses: tuple[ReadoutPulse, ...], times: RealArray, *, constant: bool
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Solve one physical trajectory on the union of all requested read grids."""
        # Evaluate the emitted field at retarded time. Restoring the absolute
        # carrier's frame phase below is necessary even when f_ref == f_drive.
        delay = self.model.receiver.time_of_flight_ns
        times = times - delay
        start = min(p.start_ns for p in pulses) if pulses else 0.0
        end = max(start, float(times[-1]))
        points = sorted(
            {start, end}
            | {
                t
                for p in pulses
                for t in (p.start_ns, p.end_ns, *p.breakpoints_ns)
                if start < t < end
            }
        )
        n = len(self.model.state_labels)
        # state component, [population,moment_1,moment_2], initial condition
        state: NDArray[np.complex128] = np.zeros((n, 3, n), dtype=np.complex128)
        state[:, 0, :] = np.eye(n)
        outputs = np.zeros((len(times), n, 3, n), dtype=np.complex128)
        outputs[times <= start] = state
        interval = self._constant_interval if constant else self._interval
        for left, right in zip(points[:-1], points[1:], strict=True):
            state = interval(state, pulses, times, outputs, left, right)
        if constant:
            # A batch content has one carrier. Its moments stay in that frame
            # through pulse corners and ringdown; restore the model frame once
            # before combining with the incident field and demodulating.
            omega = (
                2
                * np.pi
                * (pulses[0].frequency_ghz - self.model.reference_frequency_ghz)
            )
            carrier = np.exp(-1j * omega * (times - start))
            outputs[:, :, 1:, :] *= carrier[:, None, None, None]
            state[:, 1:, :] *= np.exp(-1j * omega * (end - start))
            pulse = pulses[0]
            active = (times >= pulse.start_ns) & (times < pulse.end_ns)
            incident = np.zeros(len(times), dtype=np.complex128)
            incident[active] = pulse.amplitude * np.exp(
                -1j * omega * (times[active] - start) + 1j * pulse.phase_rad
            )
            if not np.isfinite(incident).all():
                raise ValueError("readout envelope must be finite")
        else:
            incident = np.array([self._incident(pulses, t) for t in times])
        self._validate_solution(state)
        moments = outputs[:, :, 1:, :].sum(axis=1)
        outgoing = incident[:, None] - np.einsum("r,kri->ki", self._scattering, moments)
        outgoing *= np.exp(2j * np.pi * self.model.reference_frequency_ghz * delay)
        return outgoing, state

    def _incident(self, pulses: tuple[ReadoutPulse, ...], t: float) -> complex:
        return sum(
            (p.incident(t, self.model.reference_frequency_ghz) for p in pulses), 0j
        )

    def _interval(
        self, state, pulses, times, outputs, left, right
    ) -> NDArray[np.complex128]:
        shape = state.shape

        def rhs(t, flat):
            t = min(t, float(np.nextafter(right, left)))
            current = flat.reshape(shape)
            derivative = np.zeros_like(current)
            derivative[:, 0, :] = self._q @ current[:, 0, :]
            derivative[:, 1:, :] = (
                np.einsum("st,tri->sri", self._q, current[:, 1:, :])
                - np.einsum("sru,sui->sri", self._a, current[:, 1:, :])
                + self._v[None, :, None]
                * self._incident(pulses, t)
                * current[:, 0:1, :]
            )
            return derivative.ravel()

        detunings = [
            abs(p.frequency_ghz - self.model.reference_frequency_ghz) for p in pulses
        ]
        fastest = max(
            [
                float(
                    np.max(
                        np.abs(
                            self.model.frequencies_ghz
                            - self.model.reference_frequency_ghz
                        )
                    )
                ),
                *detunings,
            ]
        )
        max_step = 1 / (16 * fastest) if fastest else np.inf
        solution = solve_ivp(
            rhs,
            (left, right),
            state.ravel(),
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
            max_step=max_step,
            dense_output=True,
        )
        if (
            not solution.success
            or solution.sol is None
            or not np.isfinite(solution.y).all()
        ):
            raise RuntimeError(f"readout solver failed: {solution.message}")
        selected = (times > left) & (times <= right)
        if np.any(selected):
            outputs[selected] = solution.sol(times[selected]).T.reshape((-1, *shape))
        return np.asarray(solution.y[:, -1].reshape(shape), dtype=np.complex128)

    def _constant_interval(
        self, state, pulses, times, outputs, left, right
    ) -> NDArray[np.complex128]:
        """Propagate one flat pulse exactly, retaining moments in its carrier frame.

        Population components do not rotate. Moment components rotate relative to
        the model reference by exp(+i*omega*(t-pulse.start_ns)), even in ringdown.
        The batch contract guarantees one pulse, so each interval has a constant
        generator in this frame regardless of carrier detuning.
        """
        n = len(self.model.state_labels)
        pulse = pulses[0]
        omega = 2 * np.pi * (pulse.frequency_ghz - self.model.reference_frequency_ghz)
        incident = pulse.incident(left, pulse.frequency_ghz)
        selected = np.flatnonzero((times > left) & (times <= right))
        targets = np.append(times[selected], right)
        durations = np.diff(np.concatenate(([left], targets)))
        propagators = self._constant_line.propagators(omega, incident, durations)
        populations = state[:, 0, :]
        moments = state[:, 1:, :].reshape(2 * n, n)
        result = state.copy()
        for step, dt in enumerate(durations):
            if dt:
                ep, em, driven = propagators[float(dt)]
                moments = em @ moments + driven @ populations
                populations = ep @ populations
            result[:, 0, :] = populations
            result[:, 1:, :] = moments.reshape(n, 2, n)
            if step < len(selected):
                outputs[selected[step]] = result
        return result

    def _calibrate(self, outgoing, grid: SampleGrid) -> RealArray:
        receiver = self.model.receiver
        oscillator = np.exp(
            2j
            * np.pi
            * (grid.demodulation_frequency_ghz - self.model.reference_frequency_ghz)
            * grid.times_ns
            - 1j * grid.phase_rad
        )
        signal = (
            np.conj(outgoing * oscillator[:, None])
            * receiver.gain
            * np.exp(1j * receiver.phase_rad)
        )
        return np.stack((signal.real.T, signal.imag.T), axis=-1) + receiver.offset

    def _validate_solution(self, state) -> None:
        probabilities = state[:, 0, :]
        if not np.isfinite(state).all():
            raise RuntimeError("readout produced nonfinite moments")
        if (
            np.max(np.abs(probabilities.imag)) > 1e-8
            or np.min(probabilities.real) < -1e-8
        ):
            raise RuntimeError("readout produced invalid probabilities")
        if not np.allclose(probabilities.sum(axis=0), 1, rtol=0, atol=1e-8):
            raise RuntimeError("readout probabilities do not sum to one")


class _ConstantLine:
    """Share homogeneous line propagation while keeping carrier shifts scalar.

    The block generator is [[Q, 0], [b*incident, M+i*omega]]. For a
    well-separated Sylvester solve (M+i*omega)X-XQ=b, its driven propagator
    is incident*(exp((M+i*omega)*dt)X-X*exp(Q*dt)). Only X and the scalar
    carrier phase depend on frequency. No diagonalizability assumption is made.
    """

    def __init__(self, q: RealArray, a: NDArray[np.complex128], v: RealArray):
        self._q = q
        n = len(q)
        self._m = np.asarray(np.kron(q, np.eye(2)), dtype=np.complex128)
        self._b = np.zeros((2 * n, n), dtype=np.complex128)
        for s in range(n):
            self._m[2 * s : 2 * s + 2, 2 * s : 2 * s + 2] -= a[s]
            self._b[2 * s : 2 * s + 2, s] = v
        self._homogeneous: dict[
            float, tuple[NDArray[np.complex128], NDArray[np.complex128]]
        ] = {}

    def propagators(self, omega: float, incident: complex, durations: RealArray):
        """Return population, moment and driven blocks for positive durations.

        Durations are ns, omega is the carrier/reference angular detuning in
        rad/ns, and incident is the flat drive in its carrier frame. Cache only
        frequency-independent exponentials, not every carrier's dense matrices.
        """
        shifted = self._m + 1j * omega * np.eye(len(self._m))
        coupling = self._coupling(shifted) if incident else None
        result = {}
        for dt in np.unique(durations[durations > 0]):
            dt = float(dt)
            if dt not in self._homogeneous:
                self._homogeneous[dt] = (
                    np.asarray(scipy.linalg.expm(self._q * dt), dtype=np.complex128),
                    np.asarray(scipy.linalg.expm(self._m * dt), dtype=np.complex128),
                )
            ep, base = self._homogeneous[dt]
            em = base * np.exp(1j * omega * dt)
            if not incident:
                driven = np.zeros_like(self._b)
            elif coupling is None:
                result[dt] = self._full_step(shifted, incident, dt)
                continue
            else:
                driven = incident * (em @ coupling - coupling @ ep)
            result[dt] = (ep, em, driven)
        return result

    def _coupling(self, shifted):
        try:
            coupling = scipy.linalg.solve_sylvester(shifted, -self._q, self._b)
        except np.linalg.LinAlgError:
            # A singular driven block still has a valid full exponential.
            return None
        # Near coincident spectra can make the difference of two large terms
        # lose precision. Use the original exponential rather than regularizing
        # rates, clipping populations or assuming a diagonalizable generator.
        roundoff = np.finfo(float).eps * np.linalg.norm(coupling, ord=np.inf)
        residual = shifted @ coupling - coupling @ self._q - self._b
        if (
            not np.isfinite(coupling).all()
            or roundoff > 1e-13
            or np.linalg.norm(residual, ord=np.inf) > 1e-13
        ):
            return None
        return coupling

    def _full_step(self, shifted, incident, dt):
        n = len(self._q)
        generator = np.zeros((3 * n, 3 * n), dtype=np.complex128)
        generator[:n, :n] = self._q
        generator[n:, n:] = shifted
        generator[n:, :n] = self._b * incident
        step = np.asarray(scipy.linalg.expm(generator * dt), dtype=np.complex128)
        return step[:n, :n], step[n:, n:], step[n:, :n]


def _model_key(model: ReadoutModel) -> tuple[object, ...]:
    """Lossless numeric content key; model identity alone may name many DC points."""
    receiver = model.receiver
    return (
        model.identity,
        model.state_labels,
        model.reference_frequency_ghz,
        model.frequencies_ghz.tobytes(),
        model.kappa_internal_per_ns.tobytes(),
        model.kappa_external_per_ns.tobytes(),
        model.rates_per_ns.tobytes(),
        model.mismatch_phase_rad.tobytes(),
        receiver.time_of_flight_ns,
        receiver.gain,
        receiver.phase_rad,
        receiver.offset.tobytes(),
        receiver.noise_covariance.tobytes(),
    )


def _pulse_key(pulses: tuple[ReadoutPulse, ...]) -> tuple[object, ...]:
    """Exact pulse parameters plus identity of retained, opaque envelope callbacks."""
    return tuple(
        (
            p.start_ns,
            p.end_ns,
            p.amplitude,
            p.frequency_ghz,
            p.phase_rad,
            id(p.envelope),
            p.breakpoints_ns,
        )
        for p in pulses
    )


def _grid_key(grids: tuple[SampleGrid, ...]) -> tuple[object, ...]:
    """Physical sample coordinates and demodulation, preserving read order."""
    return tuple(
        (g.times_ns.tobytes(), g.demodulation_frequency_ghz, g.phase_rad) for g in grids
    )
