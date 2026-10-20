"""Continuous-time jump paths and vacuum-baseline cavity memory for traces."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .model import ReadoutModel, RealArray
from .request import PiecewiseLinearEnvelope, ReadoutPulse

ComplexArray = NDArray[np.complex128]


class PathFields:
    """Prepare no-jump vacuum fields once; propagate only memory per jump.

    model and pulse are validated ADC records. times is a nonempty increasing
    emitted-ns vector, before receiver delay. rtol/atol are positive solver
    tolerances supplied by ADC. The baseline integrates all states together,
    independent of transition rates. A jump changes the cavity generator, never
    the field itself. Dense baselines support arbitrary pulse envelopes without
    an ODE per shot or interpolation onto an artificial grid. Solver failures
    raise RuntimeError; invalid envelope evaluations propagate ValueError.
    """

    def __init__(
        self,
        model: ReadoutModel,
        pulse: ReadoutPulse,
        times: RealArray,
        rtol: float,
        atol: float,
    ) -> None:
        self.model, self.pulse, self.times = model, pulse, times
        self._v = np.sqrt(model.kappa_external_per_ns / 2)
        self._scattering = self._v * np.exp(-1j * model.mismatch_phase_rad)
        self._a = np.array(
            [
                np.diag(model.kappa_internal_per_ns / 2 + 2j * np.pi * detuning)
                + np.outer(self._v, self._v)
                for detuning in model.frequencies_ghz - model.reference_frequency_ghz
            ]
        )
        self._modes: list[tuple[ComplexArray, ComplexArray, ComplexArray] | None] = []
        for a in self._a:
            values, vectors = np.linalg.eig(-a)
            if np.linalg.cond(vectors) < 1e4:
                self._modes.append(
                    (
                        values,
                        vectors,
                        np.asarray(np.linalg.inv(vectors), dtype=np.complex128),
                    )
                )
            else:
                # Defective or nearly coalescent modes retain the matrix exponential.
                self._modes.append(None)
        self._segments: list[
            tuple[float, float, Callable[[RealArray], ComplexArray]]
        ] = []
        self._prepare_baselines(rtol, atol)
        self._baseline_samples = self.baseline(times)
        self._incident = np.array(
            [pulse.incident(float(t), model.reference_frequency_ghz) for t in times]
        )

    def _prepare_baselines(self, rtol: float, atol: float) -> None:
        pulse = self.pulse
        start = pulse.start_ns
        end = max(start, float(self.times[-1]))
        corners = {start, end} | {
            float(t)
            for t in (
                pulse.end_ns,
                *pulse.breakpoints_ns,
                *(
                    pulse.start_ns + pulse.envelope.knots_ns
                    if isinstance(pulse.envelope, PiecewiseLinearEnvelope)
                    else ()
                ),
            )
            if start < t < end
        }
        points = sorted(corners)
        fields = np.zeros((len(self._a), 2), dtype=np.complex128)
        fastest = max(
            float(
                np.max(
                    np.abs(
                        self.model.frequencies_ghz - self.model.reference_frequency_ghz
                    )
                )
            ),
            abs(pulse.frequency_ghz - self.model.reference_frequency_ghz),
        )
        for left, right in zip(points[:-1], points[1:], strict=True):

            def rhs(
                t: float,
                flat: ComplexArray,
                stop: float = float(np.nextafter(right, left)),
            ) -> ComplexArray:
                t = min(t, stop)
                field = flat.reshape((-1, 2))
                incident = pulse.incident(t, self.model.reference_frequency_ghz)
                return (
                    -np.einsum("sru,su->sr", self._a, field) + self._v * incident
                ).ravel()

            solution = solve_ivp(
                rhs,
                (left, right),
                fields.ravel(),
                method="DOP853",
                rtol=rtol,
                atol=atol,
                max_step=1 / (16 * fastest) if fastest else np.inf,
                dense_output=True,
            )
            if (
                not solution.success
                or solution.sol is None
                or not np.isfinite(solution.y).all()
            ):
                raise RuntimeError(f"readout path baseline failed: {solution.message}")
            self._segments.append((left, right, solution.sol))
            fields = solution.y[:, -1].reshape((-1, 2))

    def baseline(self, times: RealArray) -> ComplexArray:
        """Evaluate vacuum fields at emitted ns, shaped (time,state,cavity).

        times is a finite vector within the prepared time support. No sampling
        occurs; earlier lookback times return zero. Output owns its storage.
        """
        fields = np.zeros((len(times), len(self._a), 2), dtype=np.complex128)
        for left, right, evaluate in self._segments:
            selected = (times > left) & (times <= right)
            if np.any(selected):
                fields[selected] = evaluate(times[selected]).T.reshape(
                    (-1, len(self._a), 2)
                )
        return fields

    def _memory(
        self, state: int, elapsed: RealArray, field: ComplexArray
    ) -> ComplexArray:
        modes = self._modes[state]
        if modes is None:
            return np.array(
                [expm(-self._a[state] * t) @ field for t in elapsed],
                dtype=np.complex128,
            ).reshape((-1, 2))
        values, vectors, inverse = modes
        return (np.exp(elapsed[:, None] * values) * (inverse @ field)) @ vectors.T

    def _shot(self, initial: int, rng: np.random.Generator) -> ComplexArray:
        state = initial
        left = self.pulse.start_ns
        end = max(left, float(self.times[-1]))
        memory = np.zeros(2, dtype=np.complex128)
        fields = np.zeros((len(self.times), 2), dtype=np.complex128)
        while left < end:
            rates = self.model.rates_per_ns[:, state]
            total = float(rates.sum())
            jump = left + float(rng.exponential(1 / total)) if total else np.inf
            selected = (self.times >= left) & (self.times < jump)
            fields[selected] = self._baseline_samples[selected, state] + self._memory(
                state, self.times[selected] - left, memory
            )
            if jump >= end:
                break
            target = int(rng.choice(len(rates), p=rates / total))
            baseline = self.baseline(np.array([jump]))[0]
            memory = (
                baseline[state]
                + self._memory(state, np.array([jump - left]), memory)[0]
                - baseline[target]
            )
            state, left = target, jump
        return self._incident - fields @ self._scattering

    def draw(self, states: NDArray[np.int64], rng: np.random.Generator) -> ComplexArray:
        """Draw CTMC paths; return complex emitted fields shaped (sample,shot).

        states is a vector of valid initial indices in model order. rng is the
        caller-owned local generator; initial draws have already been consumed.
        This consumes waiting-time and target-state draws, without receiver noise.
        """
        outgoing = np.empty((len(self.times), len(states)), dtype=np.complex128)
        for shot, state in enumerate(states):
            outgoing[:, shot] = self._shot(int(state), rng)
        return outgoing
