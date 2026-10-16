"""Causal delay/first-order source lines, before reduced-flux loop mixing.

The filter is the scalar convolution of the signed real lab-frame source with
exp(-t/tau)/tau. It is not a density or conditional-field solver. Constant
complex-envelope pulses use analytic integrals; arbitrary continuous envelopes
use quadrature with the caller's tolerances. No tail truncation or shot memory.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from .evolution import ControlPulse, ControlSequence
from .evolution.controls import quantize_ns


@dataclass(frozen=True)
class FluxLine:
    """source: numeric source name matching ControlPulse.source. coefficients:

    three finite signed F1/C/F2 reduced-flux per gain coefficients. delay_ns and
    tau_ns: finite nonnegative source delay/time constant; tau=0 is instantaneous.
    Response is applied per source before this loop mapping, not after mixing.
    """

    source: str
    coefficients: tuple[float, float, float]
    delay_ns: float = 0.0
    tau_ns: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.source
            or len(self.coefficients) != 3
            or not np.isfinite((*self.coefficients, self.delay_ns, self.tau_ns)).all()
        ):
            raise ValueError("flux line requires named finite coefficients/delay/tau")
        if min(self.delay_ns, self.tau_ns) < 0:
            raise ValueError("source delay/tau must be nonnegative")


class FilteredFlux:
    """Explicit independent-shot reduced flux from DC and source controls.

    dc_flux: finite F1/C/F2 shape (3,), already offset+M_DC*live values. controls:
    complete signed envelope/carrier history; lines: unique source calibrations.
    rtol/atol: positive finite scalar convolution quadrature tolerances. Each
    evaluation reconstructs that shot's filter, with zero pulse contribution
    before source activity; no hidden cross-shot state or guessed settling time.
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
        if self.dc_flux.shape != (3,) or not np.isfinite(self.dc_flux).all():
            raise ValueError("DC flux must be finite shape (3,)")
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
        """Return actual flux shape (3,) at absolute finite ns, including tail.

        Pulse gain, sign and quadrature enter before delay/lowpass. Passive tails
        continue after pulse end without cutoff. No readout freezing is done by
        this utility; that approximation belongs to environment composition.
        """
        if not np.isfinite(time_ns):
            raise ValueError("flux evaluation time must be finite")
        result = self.dc_flux.copy()
        for line in self.lines:
            value = sum(
                self._pulse(pulse, time_ns - line.delay_ns, line.tau_ns)
                for pulse in self.controls.pulses
                if pulse.source == line.source
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
