"""Reusable arbitrary-envelope vacuum fields for physical cavity paths."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .model import ReadoutModel, RealArray
from .request import PiecewiseLinearEnvelope, ReadoutPulse

ComplexArray = NDArray[np.complex128]


def line_parameters(model: ReadoutModel) -> tuple[ComplexArray, RealArray]:
    """Return fixed-branch line blocks (state,cavity,cavity), 1/ns, and input coupling.

    Coupling is (cavity,) sqrt(1/ns). Population rates and receiver calibration do
    not enter these vacuum-field equations.
    """
    v = np.sqrt(model.kappa_external_per_ns / 2)
    a = np.array(
        [
            np.diag(model.kappa_internal_per_ns / 2 + 2j * np.pi * detuning)
            + np.outer(v, v)
            for detuning in model.frequencies_ghz - model.reference_frequency_ghz
        ]
    )
    return a, v


class PathFields:
    """Prepare no-jump vacuum fields once for arbitrary-envelope path observers.

    model and pulse are validated ADC records. times is a nonempty increasing
    emitted-ns vector, before receiver delay. rtol/atol are positive solver
    tolerances supplied by ADC. The baseline integrates all states together,
    independent of transition rates. A jump changes the cavity generator, never
    the field itself. Dense baselines support arbitrary pulse envelopes without
    an ODE per shot or interpolation onto an artificial grid. Solver failures
    raise RuntimeError; invalid envelope evaluations propagate ValueError.
    start_ns optionally limits vacuum propagation to pulse history at/after that
    emitted-ns time, without changing pulse-local coordinates. None retains the
    complete pulse history. This does not alter shot observers' default behavior.
    """

    def __init__(
        self,
        model: ReadoutModel,
        pulse: ReadoutPulse,
        times: RealArray,
        rtol: float,
        atol: float,
        *,
        start_ns: float | None = None,
    ) -> None:
        self.model, self.pulse, self.times = model, pulse, times
        self._start_ns = start_ns
        self._a, self._v = line_parameters(model)
        self._scattering = self._v * np.exp(-1j * model.mismatch_phase_rad)
        self._segments: list[
            tuple[float, float, Callable[[RealArray], ComplexArray]]
        ] = []
        self._prepare_baselines(rtol, atol)

    def _prepare_baselines(self, rtol: float, atol: float) -> None:
        pulse = self.pulse
        start = (
            pulse.start_ns
            if self._start_ns is None
            else max(pulse.start_ns, self._start_ns)
        )
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
        corners = sorted(corners)
        kappa = float(
            np.max(self.model.kappa_internal_per_ns + self.model.kappa_external_per_ns)
        )
        points = [corners[0]]
        for left, right in zip(corners[:-1], corners[1:], strict=True):
            count = max(1, int(np.ceil((right - left) * kappa)))
            points.extend(np.linspace(left, right, count + 1)[1:])
        fields = np.zeros((len(self._a), self.model.cavity_count), dtype=np.complex128)
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
                field = flat.reshape((-1, self.model.cavity_count))
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
            fields = solution.y[:, -1].reshape((-1, self.model.cavity_count))

    def baseline(self, times: RealArray) -> ComplexArray:
        """Evaluate vacuum fields at emitted ns, shaped (time,state,cavity).

        times is a finite vector within the prepared time support. No sampling
        occurs; earlier lookback times return zero. Output owns its storage.
        """
        fields = np.zeros(
            (len(times), len(self._a), self.model.cavity_count), dtype=np.complex128
        )
        for left, right, evaluate in self._segments:
            selected = (times > left) & (times <= right)
            if np.any(selected):
                fields[selected] = evaluate(times[selected]).T.reshape(
                    (-1, len(self._a), self.model.cavity_count)
                )
        return fields
