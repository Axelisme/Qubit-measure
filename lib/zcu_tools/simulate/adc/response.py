"""Joint state-conditioned first moments for a shared two-resonator line."""

from collections import Counter

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ._constant import ConstantLine
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
        self._sums: dict[
            tuple[object, ...], tuple[tuple[ReadoutPulse, ...], RealArray]
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
        self._constant_line = ConstantLine(self._q, self._a, self._v, self._scattering)

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
        reuse the cache. Integrated flat pulses on uniform grids use guarded
        eigenbasis geometric sums, cached once per content without traces.
        Common-start flat mux contents sum independent carrier responses, with
        offset added once. Each rejected carrier retains the constant-field
        route, including full block exponentials for ill-conditioned coupling.
        Unsupported contents fall back individually to the field route. Trace
        requests always use the field route and retain their own cache, so an
        integrated-only result does not avoid a later trace solve. Constant
        fields use the closed-form matrix exponential in the pulse carrier frame,
        including detuned carriers. Frequency-independent Schur bases and sampled
        scattering projections are shared across contents; ill-conditioned
        driven blocks use the full matrix exponential.
        Common-start flat mux traces superpose vacuum-start carrier fields
        through the same ConstantLine before calibrating once. Other envelopes
        and different-start multi-pulse contents use the DOP853 moment ODE.
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
        times_by_content = [
            np.unique(np.concatenate([g.times_ns for g in grids]))
            for grids in batch.grids
        ]
        relative_keys = [
            np.maximum(
                times - batch.model.receiver.time_of_flight_ns - pulses[0].start_ns,
                0.0,
            ).tobytes()
            for pulses, times in zip(batch.pulses, times_by_content, strict=True)
        ]
        grid_uses = Counter(relative_keys)
        contents = []
        summed = []
        for pulses, grids, times, relative_key in zip(
            batch.pulses, batch.grids, times_by_content, relative_keys, strict=True
        ):
            key = (model_key, _pulse_key(pulses), _grid_key(grids))
            flat_mux = (
                len(pulses) > 1
                and all(p.envelope is None for p in pulses)
                and all(p.start_ns == pulses[0].start_ns for p in pulses)
            )
            if not batch.include_traces:
                if key in self._sums:
                    summed.append(self._sums[key][1])
                    continue
                if key not in self._responses and flat_mux:
                    sums = solver._mux_integrated_sums(pulses, grids, times, counts)
                    self._sums[key] = (pulses, sums)
                    summed.append(sums)
                    continue
                if (
                    key not in self._responses
                    and len(pulses) == 1
                    and pulses[0].envelope is None
                ):
                    direct = solver._constant_line.integrated_sums(
                        batch.model, pulses[0], grids
                    )
                    if direct is not None:
                        sums, populations = direct
                        solver._validate_solution(populations[:, None, :])
                        self._sums[key] = (pulses, sums)
                        summed.append(sums)
                        continue
            if key not in self._responses:
                constant = len(pulses) == 1 and pulses[0].envelope is None
                # Building a sampled exponential for every time only pays off
                # when enough contents reuse that exact relative grid.
                projected = grid_uses[relative_key] >= len(times)
                if flat_mux:
                    outgoing = solver._mux_fields(pulses, times)
                else:
                    outgoing, _ = solver._fields(
                        pulses, times, constant=constant, projected=projected
                    )
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
            traces = self._responses[key][1]
            contents.append(traces)
            summed.append(np.stack([trace.sum(axis=-2) for trace in traces]))
        sums = np.stack(summed)
        dense = (
            np.stack([np.stack(traces) for traces in contents])
            if batch.include_traces
            else None
        )
        return ConditionalSums(sums, counts, dense)

    def _mux_fields(
        self, pulses: tuple[ReadoutPulse, ...], times: RealArray
    ) -> NDArray[np.complex128]:
        """Superpose vacuum-start flat carriers before receiver calibration.

        Common starts give every carrier the same field-independent population
        trajectory. Homogeneous projections are shared by ConstantLine; only one
        carrier's temporary fields/moments are needed at a time. Its Sylvester
        guard retains full block exponential fallback, never a field ODE.
        """
        outgoing = np.zeros(
            (len(times), len(self.model.state_labels)), dtype=np.complex128
        )
        for pulse in pulses:
            part, _ = self._fields((pulse,), times, constant=True, projected=True)
            outgoing += part
        return outgoing

    def _mux_integrated_sums(
        self,
        pulses: tuple[ReadoutPulse, ...],
        grids: tuple[SampleGrid, ...],
        times: RealArray,
        counts: NDArray[np.int64],
    ) -> RealArray:
        """Sum flat common-origin carriers without duplicating receiver offset.

        Populations obey the same field-independent generator for each carrier.
        Common starts therefore give identical initial populations and make the
        moment equations linear in incident field, including unequal end times.
        Direct-sum guards remain per carrier; rejection never sends the entire
        mux content back to the ODE or relaxes the constant-line guards.
        """
        parts = []
        for pulse in pulses:
            direct = self._constant_line.integrated_sums(self.model, pulse, grids)
            if direct is not None:
                value, populations = direct
                self._validate_solution(populations[:, None, :])
            else:
                outgoing, _ = self._fields(
                    (pulse,), times, constant=True, projected=True
                )
                value = np.stack(
                    [
                        self._calibrate(
                            outgoing[np.searchsorted(times, g.times_ns)], g
                        ).sum(axis=-2)
                        for g in grids
                    ]
                )
            parts.append(value)
        # Every calibrated component already includes offset at every sample.
        return np.sum(parts, axis=0) - (
            (len(pulses) - 1) * counts[:, None, None] * self.model.receiver.offset
        )

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
        self,
        pulses: tuple[ReadoutPulse, ...],
        times: RealArray,
        *,
        constant: bool,
        projected: bool = False,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Solve one physical trajectory on the union of all requested read grids."""
        # Evaluate the emitted field at retarded time. Restoring the absolute
        # carrier's frame phase below is necessary even when f_ref == f_drive.
        delay = self.model.receiver.time_of_flight_ns
        times = times - delay
        start = min(p.start_ns for p in pulses) if pulses else 0.0
        end = max(start, float(times[-1]))
        if constant and projected:
            omega = (
                2
                * np.pi
                * (pulses[0].frequency_ghz - self.model.reference_frequency_ghz)
            )
            solution = self._constant_line.trajectory(pulses[0], times, omega)
            if solution is not None:
                outgoing, state = solution
                self._validate_solution(state)
                state[:, 1:, :] *= np.exp(-1j * omega * (end - start))
                outgoing *= np.exp(
                    2j * np.pi * self.model.reference_frequency_ghz * delay
                )
                return outgoing, state
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
