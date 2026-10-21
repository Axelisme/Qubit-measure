"""Causal delayed multi-exponential source lines before reduced-flux mixing.

Each source filters its signed real lab-frame signal before mapping to n loops.
Constant complex-envelope pulses use analytic integrals; arbitrary continuous
envelopes use quadrature with the caller's tolerances. No tail truncation or
shot memory, density propagation or conditional-field solution occurs here.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from .evolution import ControlPulse, ControlSequence, quantize_ns


@dataclass(frozen=True)
class DistortionTerm:
    """One exponential contribution to a source's causal step response.

    amplitude: finite signed dimensionless step weight; negative values and sums
        outside [0, 1] are allowed for overshoot/undershoot calibration.
    tau_ns: finite strictly positive decay time in ns.
    Invalid parameters raise ValueError at construction.
    """

    amplitude: float
    tau_ns: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.amplitude, self.tau_ns]).all() or self.tau_ns <= 0:
            raise ValueError(
                "distortion amplitude must be finite and tau positive finite"
            )


@dataclass(frozen=True)
class FluxLine:
    """Calibration for one source, applied before reduced-flux loop mixing.

    source: nonempty numeric source name matching ControlPulse.source.
    coefficients: n >= 1 finite signed reduced-flux-per-gain coefficients, in the
        same loop order as FilteredFlux.dc_flux (FTF uses F1/C/F2, n=3).
    delay_ns: finite nonnegative causal delay in ns.
    tau_ns: finite nonnegative base lowpass time in ns; zero is instantaneous.
    distortion_terms: exponential step weights A_k and positive decay times tau_k.
        Empty retains the base lowpass exactly. After delay the unit-step response
        is 1 - (1-sum(A_k))*exp(-t/tau_ns) - sum(A_k*exp(-t/tau_k)) for t>=0,
        zero before t=0. For tau_ns=0 its exponential is zero, including at t=0;
        the instantaneous feedthrough is 1-sum(A_k). Tails are never truncated.
    Invalid fields raise ValueError at construction.
    """

    source: str
    coefficients: tuple[float, ...]
    delay_ns: float = 0.0
    tau_ns: float = 0.0
    distortion_terms: tuple[DistortionTerm, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.source
            or not self.coefficients
            or not np.isfinite((*self.coefficients, self.delay_ns, self.tau_ns)).all()
        ):
            raise ValueError("flux line requires named finite coefficients/delay/tau")
        if min(self.delay_ns, self.tau_ns) < 0:
            raise ValueError("source delay/tau must be nonnegative")


class FilteredFlux:
    """Explicit independent-shot reduced flux from DC and source controls.

    dc_flux: finite shape (n,), n>=1, already offset+M_DC*live values; copied at
        construction, exposed as an owned array that callers must not modify.
    controls: complete signed envelope/carrier history for this independent shot.
    lines: unique source calibrations, each with n coefficients; unmapped controls
        do not contribute. Empty lines return DC unchanged.
    rtol/atol: positive finite scalar convolution quadrature tolerances.
    Invalid dimensions, repeated source names or tolerances raise ValueError at
    construction. Each evaluation reconstructs the causal shot response with zero
    pulse contribution before activity; there is no cross-shot state or settling
    cutoff. Public controls, lines, rtol and atol retain the constructor inputs.
    """

    def __init__(
        self,
        dc_flux: NDArray[np.float64],
        controls: ControlSequence,
        lines: tuple[FluxLine, ...],
        *,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        self.dc_flux = np.array(dc_flux, dtype=float, copy=True)
        if (
            self.dc_flux.ndim != 1
            or not self.dc_flux.size
            or not np.isfinite(self.dc_flux).all()
        ):
            raise ValueError("DC flux must be finite shape (n,) with n >= 1")
        if any(len(line.coefficients) != self.dc_flux.size for line in lines):
            raise ValueError(
                "flux line coefficient dimensions must match DC flux shape"
            )
        if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
            raise ValueError("line quadrature tolerances must be positive finite")
        if len({line.source for line in lines}) != len(lines):
            raise ValueError("flux line source names must be unique")
        self.controls, self.lines, self.rtol, self.atol = controls, lines, rtol, atol

    @property
    def event_times_ns(self) -> tuple[float, ...]:
        """Return delayed absolute source events for numeric model partitioning.

        Includes quantized pulse support, envelope breakpoints and phase changes
        before each line's delay is applied, matching at(). Lowpass tails remain
        continuous and untruncated; these events impose no integration step size.
        The evolution owner rounds their delayed absolute coordinates to 2 ns.
        """
        events: set[float] = set()
        for line in self.lines:
            for pulse in self.controls.pulses:
                if pulse.source == line.source:
                    events.update(
                        quantize_ns(t) + line.delay_ns
                        for t in (pulse.start_ns, pulse.end_ns, *pulse.breakpoints_ns)
                    )
            events.update(
                quantize_ns(event.time_ns) + line.delay_ns
                for event in self.controls.phase_events
                if event.source == line.source
            )
        return tuple(sorted(events))

    def at(self, time_ns: float) -> NDArray[np.float64]:
        """Return actual flux shape (n,) at absolute finite ns, including tails.

        Nonfinite time_ns raises ValueError; failed quadrature raises RuntimeError.
        The returned array is owned by the caller; changing it does not affect DC.

        Pulse gain, sign and quadrature enter before delay/lowpass. Passive tails
        continue after pulse end without cutoff. No readout freezing is done by
        this utility; that approximation belongs to environment composition.
        """
        if not np.isfinite(time_ns):
            raise ValueError("flux evaluation time must be finite")
        result = self.dc_flux.copy()
        for line in self.lines:
            value = 0.0
            for pulse in self.controls.pulses:
                if pulse.source != line.source:
                    continue
                time = time_ns - line.delay_ns
                base = self._pulse(pulse, time, line.tau_ns)
                if not line.distortion_terms:
                    value += base
                    continue
                value += (1 - sum(t.amplitude for t in line.distortion_terms)) * base
                value += sum(
                    term.amplitude * self._pulse(pulse, time, term.tau_ns)
                    for term in line.distortion_terms
                )
            result += np.array(line.coefficients) * value
        return result

    def _pulse(self, pulse: ControlPulse, time: float, tau: float) -> float:
        start, end = quantize_ns(pulse.start_ns), quantize_ns(pulse.end_ns)
        if time < start or end <= start:
            return 0.0
        if tau == 0:
            return self.controls.amplitude(pulse, time).real if time < end else 0.0
        upper = min(time, end)
        if pulse.envelope.constant is not None and not self.controls.phase_events:
            omega = 2 * np.pi * pulse.frequency_ghz
            amplitude = self.controls.amplitude(pulse, start)
            lam = 1 / tau + 1j * omega
            # Difference factored at the upper endpoint avoids overflow for long
            # pulses and expm1 preserves short intervals without subtractive loss.
            value = (
                amplitude
                * np.exp(1j * omega * (upper - start))
                * np.exp(-(time - upper) / tau)
                * (-np.expm1(-lam * (upper - start)))
                / (tau * lam)
            )
            return float(value.real)
        points = sorted(
            {start, upper, *(t for t in pulse.breakpoints_ns if start < t < upper)}
        )
        total = 0.0
        for lo, hi in zip(points[:-1], points[1:], strict=True):
            output = quad(
                lambda t: (
                    np.exp(-(time - t) / tau)
                    / tau
                    * self.controls.amplitude(pulse, t).real
                ),
                lo,
                hi,
                epsabs=self.atol,
                epsrel=self.rtol,
                full_output=1,
            )
            if len(output) != 3 or not np.isfinite(output[0]):
                raise RuntimeError("source line convolution quadrature failed")
            total += output[0]
        return total
