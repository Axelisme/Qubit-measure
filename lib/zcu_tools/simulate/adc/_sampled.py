"""Vacuum-start affine-input propagation without dense knot histories."""

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from .model import RealArray
from .request import PiecewiseLinearEnvelope, ReadoutPulse

ComplexArray = NDArray[np.complex128]


class SampledLine:
    """Carrier-local affine convolution workspace, discarded after one trajectory.

    q, m, b and output are validated population/moment/coupling/projection arrays
    from ConstantLine. omega is carrier/reference angular detuning in rad/ns.
    The augmented (moments, t*populations, populations) exponential provides
    exp(Q*dt), exp((M+iω)*dt), and both forcing convolutions without spectral
    division, even for singular/defective generators. Cache at most 128 blocks;
    retain no dense knot history. Nonfinite blocks request ODE fallback.
    """

    def __init__(
        self,
        q: RealArray,
        m: ComplexArray,
        b: ComplexArray,
        output: ComplexArray,
        omega: float,
    ):
        n = len(q)
        self._generator = np.zeros((4 * n, 4 * n), dtype=np.complex128)
        self._generator[: 2 * n, : 2 * n] = m + 1j * omega * np.eye(2 * n)
        self._generator[: 2 * n, 2 * n : 3 * n] = b
        self._generator[2 * n : 3 * n, 2 * n : 3 * n] = q
        self._generator[2 * n : 3 * n, 3 * n :] = np.eye(n)
        self._generator[3 * n :, 3 * n :] = q
        self._blocks: dict[
            float, tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]
        ] = {}

        self._output, self._omega = output, omega

    def _advance(self, populations, moments, dt, amplitude, slope):
        n = len(populations)
        if not dt:
            return populations, moments
        if dt not in self._blocks:
            step = np.asarray(
                scipy.linalg.expm(self._generator * dt), dtype=np.complex128
            )
            if not np.isfinite(step).all():
                return None
            # Bound temporary storage even on fully irregular knot/sample grids.
            if len(self._blocks) == 128:
                self._blocks.pop(next(iter(self._blocks)))
            self._blocks[dt] = (
                step[3 * n :, 3 * n :].copy(),
                step[: 2 * n, : 2 * n].copy(),
                step[: 2 * n, 2 * n : 3 * n].copy(),
                step[: 2 * n, 3 * n :].copy(),
            )
        ep, em, c0, c1 = self._blocks[dt]
        return ep @ populations, em @ moments + (
            amplitude * c0 + slope * c1
        ) @ populations

    def trajectory(
        self, pulse: ReadoutPulse, times: RealArray
    ) -> tuple[ComplexArray, ComplexArray] | None:
        """Emitted fields/final carrier-frame state, or ODE fallback.

        pulse supplies a PiecewiseLinearEnvelope and finite support. times are
        retarded absolute ns. ADC evaluations do not restart knot propagation.
        """
        envelope = pulse.envelope
        if not isinstance(envelope, PiecewiseLinearEnvelope):
            return None
        n = self._generator.shape[0] // 4
        omega, output = self._omega, self._output
        start = pulse.start_ns
        end = max(start, float(times[-1]))
        corners = np.unique(
            np.concatenate(
                (
                    [start, end],
                    (start + envelope.knots_ns)[
                        (start + envelope.knots_ns > start)
                        & (start + envelope.knots_ns < min(end, pulse.end_ns))
                    ],
                    [pulse.end_ns] if start < pulse.end_ns < end else [],
                )
            )
        )
        populations = np.eye(n, dtype=np.complex128)
        moments = np.zeros((2 * n, n), dtype=np.complex128)
        outgoing = np.zeros((len(times), n), dtype=np.complex128)
        scale = pulse.amplitude * np.exp(1j * pulse.phase_rad)
        for left, right in zip(corners[:-1], corners[1:], strict=True):
            active = left < pulse.end_ns
            amplitude = scale * envelope(left - start) if active else 0j
            slope = (
                scale
                * (envelope(right - start) - envelope(left - start))
                / (right - left)
                if active
                else 0j
            )
            selected = np.flatnonzero((times > left) & (times <= right))
            # ADC evaluations do not interrupt knot propagation or retain history.
            for index in selected:
                dt = float(times[index] - left)
                value = self._advance(populations, moments, dt, amplitude, slope)
                if value is None:
                    return None
                _, sampled = value
                incident = (
                    scale * envelope(float(times[index] - start))
                    if times[index] < pulse.end_ns
                    else 0j
                )
                outgoing[index] = (incident - output @ sampled) * np.exp(
                    -1j * omega * (times[index] - start)
                )
            value = self._advance(
                populations, moments, float(right - left), amplitude, slope
            )
            if value is None:
                return None
            populations, moments = value
        at_start = times == start
        if pulse.end_ns > start:
            outgoing[at_start] = scale * envelope(0.0)
        state = np.empty((n, 3, n), dtype=np.complex128)
        state[:, 0, :] = populations
        state[:, 1:, :] = moments.reshape(n, 2, n)
        if not np.isfinite(state).all() or not np.isfinite(outgoing).all():
            return None
        return outgoing, state
