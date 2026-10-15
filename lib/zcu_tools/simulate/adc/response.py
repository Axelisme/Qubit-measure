"""Joint state-conditioned first moments for a shared two-resonator line."""

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .model import ReadoutModel, RealArray
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
        self._v = np.sqrt(model.kappa_external_per_ns / 2)
        self._q = model.rates_per_ns - np.diag(model.rates_per_ns.sum(axis=0))
        detuning = model.frequencies_ghz - model.reference_frequency_ghz
        self._a = np.array(
            [
                np.diag(model.kappa_internal_per_ns / 2 + 2j * np.pi * f)
                + np.outer(self._v, self._v)
                for f in detuning
            ]
        )

    def response(
        self, pulses: tuple[ReadoutPulse, ...], grid: SampleGrid
    ) -> ConditionalResponse:
        """Solve all initial-state conditional mean fields once on the ADC grid.

        pulses are finite coherent readout drives in ns/GHz. grid owns physical
        times and demodulation. Return conditional traces and final populations
        in model.state_labels order, including pre-pulse/ringup/ringdown. No
        initial-state or receiver-noise draws occur. State-conditioned moments
        obey dm_s=ΣQ_st*m_t-A_s*m_s+v*b*p_s, not a mean-frequency field equation.
        ValueError signals invalid input/callbacks; RuntimeError signals failed
        integration or invalid probability/moment output. No density Hilbert
        space, jump-time sampling, clipping or renormalization is introduced.
        """
        start = min(p.start_ns for p in pulses) if pulses else 0.0
        end = max(start, float(grid.times_ns[-1]))
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
        outputs = np.zeros((len(grid.times_ns), n, 3, n), dtype=np.complex128)
        outputs[grid.times_ns <= start] = state
        for left, right in zip(points[:-1], points[1:], strict=True):
            state = self._interval(state, pulses, grid.times_ns, outputs, left, right)
        self._validate_solution(state)
        moments = outputs[:, :, 1:, :].sum(axis=1)
        outgoing = np.array([self._incident(pulses, t) for t in grid.times_ns])[
            :, None
        ] - np.einsum("r,kri->ki", self._v, moments)
        iq = self._calibrate(outgoing, grid)
        return ConditionalResponse(self.model.state_labels, iq, state[:, 0, :].real.T)

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
            outgoing
            * oscillator[:, None]
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
