"""Calibrated discrete sums of the constant shared-line moment solution.

Diagonalize M=Um diag(mu) Um^-1 and Q=Uq diag(lambda) Uq^-1. The
Sylvester equation (M+i*omega)X-XQ=B gives Y=Um^-1 X Uq with
Y_ab=(Um^-1 B Uq)_ab/(mu_a+i*omega-lambda_b). Project the driven
solution exp((M+i*omega)t)X-X exp(Qt) onto the outgoing field.
Demodulation shifts each exponent before summing its geometric progression.
The finite pulse is an on-step minus an off-step; the latter acts on exp(Q*T)
to preserve population transfer during the pulse. No continuous quadrature or
sample-period factor enters these discrete ADC sums.

Defective, poorly reconstructed or poorly separated generators keep the field
route, including its full-exponential fallback for Sylvester cancellation.
"""

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from .model import ReadoutModel, RealArray
from .request import ReadoutPulse, SampleGrid

ComplexArray = NDArray[np.complex128]


def _geometric(
    rates: ComplexArray, start: float, step: float, count: int
) -> ComplexArray:
    """Sum exp(rate * (start + k*step)), including the zero-rate limit."""
    z = rates * step
    # Reduce scalar phase before expm1, including the sampled alias limit.
    phase = np.where(
        abs(z.imag) > np.pi, (z.imag + np.pi) % (2 * np.pi) - np.pi, z.imag
    )
    z = z.real + 1j * phase
    den = np.expm1(z)
    ratio = np.full_like(z, count, dtype=np.complex128)
    np.divide(np.expm1(count * z), den, out=ratio, where=den != 0)
    return np.asarray(np.exp(rates * start) * ratio, dtype=np.complex128)


class IntegratedLine:
    """Spectral workspace for one validated constant-line model.

    q: population generator (states, states), 1/ns.
    m: homogeneous moment generator (cavity*states, cavity*states), 1/ns.
    b: drive coupling (cavity*states, states).
    output: scattered-field projection (cavity*states,).
    Arrays remain caller-owned. valid is False for ill-conditioned eigenbases
    or reconstruction residuals above 1e-13; sums then requests fallback.
    """

    def __init__(
        self, q: RealArray, m: ComplexArray, b: ComplexArray, output: ComplexArray
    ):
        self._q, self._m, self._b = q, m, b
        moment_eigen = scipy.linalg.eig(m)
        rate_eigen = scipy.linalg.eig(q)
        lm, um = moment_eigen[0], moment_eigen[1]
        lq, uq = rate_eigen[0], rate_eigen[1]
        self._lm = np.asarray(lm, dtype=np.complex128)
        self._lq = np.asarray(lq, dtype=np.complex128)
        self._um = np.asarray(um, dtype=np.complex128)
        self._uq = np.asarray(uq, dtype=np.complex128)
        self._valid = max(np.linalg.cond(um), np.linalg.cond(uq)) < 1e4
        # Do not invert a defective basis. Unused placeholders retain shapes.
        self._im = np.asarray(
            scipy.linalg.inv(um) if self._valid else um, dtype=np.complex128
        )
        self._iq = np.asarray(
            scipy.linalg.inv(uq) if self._valid else uq, dtype=np.complex128
        )
        for a, u, values, inv in ((m, um, lm, self._im), (q, uq, lq, self._iq)):
            self._valid = bool(
                self._valid and np.linalg.norm(a - (u * values) @ inv, np.inf) < 1e-13
            )
        self._output = output @ um
        self._drive = self._im @ b @ uq

    def sums(
        self, model: ReadoutModel, pulse: ReadoutPulse, grids: tuple[SampleGrid, ...]
    ) -> tuple[RealArray, ComplexArray] | None:
        """Return calibrated (reads, states, IQ) sums and final populations.

        Only flat, uniformly sampled grids are supported. Return None for a
        nonuniform grid, ill-conditioned eigenbasis, near-coincident spectra,
        or the original Sylvester cancellation/residual guard. Populations refer
        to the last retarded sample, no earlier than pulse start. No traces,
        quadrature, sample-period factor or noise are constructed.
        """
        if not self._valid or pulse.envelope is not None:
            return None
        omega = 2 * np.pi * (pulse.frequency_ghz - model.reference_frequency_ghz)
        gap = self._lm[:, None] + 1j * omega - self._lq[None, :]
        if np.min(abs(gap)) < 1e-10:
            return None
        y = self._drive / gap
        x = self._um @ y @ self._iq
        residual = (
            (self._m + 1j * omega * np.eye(len(self._m))) @ x - x @ self._q - self._b
        )
        if (
            not np.isfinite(x).all()
            or np.finfo(float).eps * np.linalg.norm(x, np.inf) > 1e-13
            or np.linalg.norm(residual, np.inf) > 1e-13
        ):
            return None
        output_x = self._output @ y
        results = []
        for grid in grids:
            result = self._grid_sum(model, pulse, grid, omega, y, output_x)
            if result is None:
                return None
            results.append(result)
        elapsed = max(
            0.0,
            max(float(g.times_ns[-1]) for g in grids)
            - model.receiver.time_of_flight_ns
            - pulse.start_ns,
        )
        populations = (self._uq * np.exp(self._lq * elapsed)) @ self._iq
        return np.stack(results), np.asarray(populations, dtype=np.complex128)

    def batch_sums(
        self,
        model: ReadoutModel,
        pulses: tuple[ReadoutPulse, ...],
        grids: tuple[tuple[SampleGrid, ...], ...],
    ) -> list[tuple[RealArray, ComplexArray] | None]:
        """Sum a tile sharing exact start/end and per-read time coordinates.

        Carrier, amplitude, phase and demodulation may vary. Return None for
        each rejected content, retaining the scalar spectral/field fallback.
        Caller limits the tile to 64 contents. Guards are evaluated per carrier;
        population evolution and uniform-grid checks are shared, not relaxed.
        """
        results: list[tuple[RealArray, ComplexArray] | None] = [None] * len(pulses)
        if not self._valid:
            return results
        omega = (
            2
            * np.pi
            * (
                np.array([p.frequency_ghz for p in pulses])
                - model.reference_frequency_ghz
            )
        )
        gap = (
            self._lm[None, :, None]
            + 1j * omega[:, None, None]
            - self._lq[None, None, :]
        )
        separated = np.min(abs(gap), axis=(1, 2)) >= 1e-10
        # Rejected gaps never enter a division or contaminate other contents.
        indices = np.flatnonzero(separated)
        if not len(indices):
            return results
        y = self._drive[None, :, :] / gap[indices]
        x = (self._um[None, :, :] @ y) @ self._iq
        residual = (
            (
                self._m[None, :, :]
                + 1j * omega[indices, None, None] * np.eye(len(self._m))
            )
            @ x
            - x @ self._q
            - self._b
        )
        accepted = (
            np.isfinite(x).all(axis=(1, 2))
            & (np.finfo(float).eps * np.linalg.norm(x, np.inf, axis=(1, 2)) <= 1e-13)
            & (np.linalg.norm(residual, np.inf, axis=(1, 2)) <= 1e-13)
        )
        indices, y = indices[accepted], y[accepted]
        if not len(indices):
            return results
        output_x = np.einsum("m,fmn->fn", self._output, y)
        reads = []
        for read in range(len(grids[0])):
            selected_grids = tuple(grids[i][read] for i in indices)
            value = self._batch_grid_sum(
                model, pulses[0], selected_grids, omega[indices], y, output_x
            )
            if value is None:
                return results
            reads.append(value)
        values = np.stack(reads, axis=1)
        amplitude = np.array([p.amplitude * np.exp(1j * p.phase_rad) for p in pulses])[
            indices
        ]
        values *= amplitude[:, None, None]
        receiver = model.receiver
        values *= np.exp(
            2j * np.pi * model.reference_frequency_ghz * receiver.time_of_flight_ns
        )
        background = np.array(
            [receiver.background(pulses[i].frequency_ghz) for i in indices]
        )
        calibrated = (
            np.conj(values)
            * background[:, None, None]
            * receiver.gain
            * np.exp(1j * receiver.phase_rad)
        )
        iq = np.stack((calibrated.real, calibrated.imag), axis=-1)
        counts = np.array([len(g.times_ns) for g in grids[0]])
        iq += counts[None, :, None, None] * receiver.offset
        elapsed = max(
            0.0,
            max(float(g.times_ns[-1]) for g in grids[0])
            - receiver.time_of_flight_ns
            - pulses[0].start_ns,
        )
        populations = np.asarray(
            (self._uq * np.exp(self._lq * elapsed)) @ self._iq, dtype=np.complex128
        )
        for row, index in enumerate(indices):
            results[index] = (iq[row], populations)
        return results

    def _batch_grid_sum(self, model, pulse, grids, omega, y, output_x):
        times = grids[0].times_ns
        step = (
            float((times[-1] - times[0]) / (len(times) - 1)) if len(times) > 1 else 1.0
        )
        tolerance = 4 * abs(np.spacing(np.max(abs(times)))) + 8 * np.finfo(
            float
        ).eps * abs(times[-1] - times[0])
        if np.any(abs(times - (times[0] + np.arange(len(times)) * step)) > tolerance):
            return None
        delay = model.receiver.time_of_flight_ns
        elapsed = times - delay - pulse.start_ns
        duration = pulse.end_ns - pulse.start_ns
        nu = (
            2
            * np.pi
            * (
                np.array([g.demodulation_frequency_ghz for g in grids])
                - model.reference_frequency_ghz
            )
        )
        result = np.zeros((len(omega), len(self._q)), dtype=np.complex128)
        active = elapsed[(elapsed >= 0) & (elapsed < duration)]
        if len(active):
            result += _geometric(
                1j * (nu - omega)[:, None], float(active[0]), step, len(active)
            )

        def scatter(ts):
            gm = _geometric(
                self._lm[None, :] + 1j * nu[:, None], float(ts[0]), step, len(ts)
            )
            gq = _geometric(
                self._lq[None, :] + 1j * (nu - omega)[:, None],
                float(ts[0]),
                step,
                len(ts),
            )
            return (
                np.einsum("fm,fmn->fn", self._output[None, :] * gm, y) - output_x * gq
            ) @ self._iq

        causal = elapsed[elapsed >= 0]
        if len(causal):
            result -= scatter(causal)
        after = elapsed[elapsed > duration] - duration
        if len(after):
            ep = (self._uq * np.exp(self._lq * duration)) @ self._iq
            result += np.exp(1j * (nu - omega) * duration)[:, None] * (
                scatter(after) @ ep
            )
        phase = np.exp(
            1j * nu * (delay + pulse.start_ns)
            - 1j * np.array([g.phase_rad for g in grids])
        )
        return result * phase[:, None]

    def _scattered_sum(
        self,
        times: RealArray,
        step: float,
        nu: float,
        omega: float,
        y: ComplexArray,
        output_x: ComplexArray,
    ) -> ComplexArray:
        gm = _geometric(self._lm + 1j * nu, float(times[0]), step, len(times))
        gq = _geometric(self._lq + 1j * (nu - omega), float(times[0]), step, len(times))
        return np.asarray(
            ((self._output * gm) @ y - output_x * gq) @ self._iq, dtype=np.complex128
        )

    def _grid_sum(
        self,
        model: ReadoutModel,
        pulse: ReadoutPulse,
        grid: SampleGrid,
        omega: float,
        y: ComplexArray,
        output_x: ComplexArray,
    ) -> RealArray | None:
        times = grid.times_ns
        step = (
            float((times[-1] - times[0]) / (len(times) - 1)) if len(times) > 1 else 1.0
        )
        # Compare coordinates to their endpoint-fitted lattice, not adjacent
        # differences to a fixed absolute ns threshold. Four coordinate ULPs
        # cover quantization at late absolute starts; the span term covers the
        # lattice arithmetic itself. Meaningful nonuniformity still falls back.
        lattice = times[0] + np.arange(len(times)) * step
        tolerance = 4 * abs(np.spacing(np.max(abs(times)))) + 8 * np.finfo(
            float
        ).eps * abs(times[-1] - times[0])
        if np.any(abs(times - lattice) > tolerance):
            return None
        receiver = model.receiver
        delay = receiver.time_of_flight_ns
        elapsed = times - delay - pulse.start_ns
        duration = pulse.end_ns - pulse.start_ns
        nu = (
            2
            * np.pi
            * (grid.demodulation_frequency_ghz - model.reference_frequency_ghz)
        )
        phase = np.exp(1j * nu * (delay + pulse.start_ns) - 1j * grid.phase_rad)
        summed = np.zeros(len(self._q), dtype=np.complex128)
        active = elapsed[(elapsed >= 0) & (elapsed < duration)]
        if len(active):
            summed += _geometric(
                np.array([1j * (nu - omega)]), float(active[0]), step, len(active)
            )[0]
        causal = elapsed[elapsed >= 0]
        if len(causal):
            summed -= self._scattered_sum(causal, step, nu, omega, y, output_x)
        after = elapsed[elapsed > duration] - duration
        if len(after):
            ep_duration = (self._uq * np.exp(self._lq * duration)) @ self._iq
            summed += np.exp(1j * (nu - omega) * duration) * (
                self._scattered_sum(after, step, nu, omega, y, output_x) @ ep_duration
            )
        summed *= pulse.amplitude * np.exp(1j * pulse.phase_rad) * phase
        summed *= np.exp(2j * np.pi * model.reference_frequency_ghz * delay)
        calibrated = (
            np.conj(summed)
            * receiver.background(pulse.frequency_ghz)
            * receiver.gain
            * np.exp(1j * receiver.phase_rad)
        )
        iq = np.stack((calibrated.real, calibrated.imag), axis=-1)
        return iq + len(times) * receiver.offset
