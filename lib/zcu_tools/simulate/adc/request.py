"""Explicit readout pulse, sample and sampling contracts in ns/GHz."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import SamplingIdentity

from .model import RealArray, readonly_array
from .populations import validate_populations


@dataclass(frozen=True, eq=False)
class PiecewiseLinearEnvelope:
    """Content-defined dimensionless complex envelope on pulse-local ns.

    knots_ns: owned read-only finite nonempty increasing float64 vector, starting
        at zero. values: owned read-only complex128 vector of the same shape.
    Interpolate linearly and hold the last value through pulse end. Pulse support
    belongs to ReadoutPulse. Calling with negative/nonfinite ns raises ValueError.
    content_key is a lossless pair of bytes for numeric cache reuse. Only this
    envelope type enables closed-form sampled propagation in responses(); arbitrary
    callbacks retain the ODE. Invalid coordinates/values/shapes raise ValueError.
    """

    knots_ns: RealArray
    values: NDArray[np.complex128]

    def __post_init__(self) -> None:
        knots = readonly_array(self.knots_ns, "envelope knots")
        values = np.array(self.values, dtype=np.complex128, copy=True)
        if (
            knots.ndim != 1
            or not len(knots)
            or knots[0] != 0
            or np.any(np.diff(knots) <= 0)
            or values.shape != knots.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "envelope requires finite matching vectors and increasing knots from zero"
            )
        values.setflags(write=False)
        object.__setattr__(self, "knots_ns", knots)
        object.__setattr__(self, "values", values)

    @property
    def content_key(self) -> tuple[bytes, bytes]:
        """Lossless numeric coordinates and amplitudes, independent of identity."""
        return self.knots_ns.tobytes(), self.values.tobytes()

    def __call__(self, local_ns: float) -> complex:
        """Evaluate finite nonnegative pulse-local ns, holding the last value."""
        if not np.isfinite(local_ns) or local_ns < 0:
            raise ValueError("envelope time must be finite and nonnegative")
        return complex(np.interp(local_ns, self.knots_ns, self.values))


@dataclass(frozen=True)
class ReadoutPulse:
    """Finite coherent incident pulse in the model's rotating coordinates.

    start_ns/end_ns: absolute ns support [start,end), no ADC-side quantization.
        Environment supplies the already-quantized control coordinates.
    amplitude: signed complex b_in in sqrt(photons/ns), before envelope/carrier.
    frequency_ghz: absolute carrier cycles/ns; input in the model reference frame
        is amplitude*envelope(t-start)*exp(-2πi*(f-f_ref)*(t-start)+i*phase).
    phase_rad: carrier phase at pulse start, in radians.
    envelope: optional pure dimensionless complex callable of pulse-local ns;
        None is flat one. PiecewiseLinearEnvelope enables closed-form sampled
        propagation in responses(); arbitrary callbacks retain the ODE and are
        caller-owned and must remain deterministic.
        It is not normalized or sampled onto an artificial grid.
    breakpoints_ns: absolute ns semantic envelope corners for integration, inside
        pulse support. Other sample/event coordinates never restart the envelope.
    Invalid/nonfinite values raise ValueError. Coincident pulses add coherently.
    """

    start_ns: float
    end_ns: float
    amplitude: complex
    frequency_ghz: float
    phase_rad: float = 0.0
    envelope: Callable[[float], complex] | None = None
    breakpoints_ns: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(
            [
                self.start_ns,
                self.end_ns,
                self.amplitude,
                self.frequency_ghz,
                self.phase_rad,
                *self.breakpoints_ns,
            ]
        ).all():
            raise ValueError("readout pulse values must be finite")
        if self.end_ns < self.start_ns:
            raise ValueError("readout pulse endpoints must be ordered")
        if any(not self.start_ns <= t <= self.end_ns for t in self.breakpoints_ns):
            raise ValueError("readout breakpoints must lie inside pulse")

    def incident(self, time_ns: float, reference_frequency_ghz: float) -> complex:
        """Return b_in in sqrt(photons/ns) at absolute ns, zero outside support.

        reference_frequency_ghz is the explicit numeric rotating frame. No
        state-dependent frequency averaging is involved. Nonfinite envelope
        evaluations raise ValueError.
        """
        if not self.start_ns <= time_ns < self.end_ns:
            return 0j
        local = time_ns - self.start_ns
        value = 1.0 if self.envelope is None else self.envelope(local)
        result = complex(
            self.amplitude
            * value
            * np.exp(
                -2j * np.pi * (self.frequency_ghz - reference_frequency_ghz) * local
                + 1j * self.phase_rad
            )
        )
        if not np.isfinite(result):
            raise ValueError("readout envelope must be finite")
        return result


@dataclass(frozen=True)
class SampleGrid:
    """One receiver's physical sample coordinates and demodulation oscillator.

    times_ns: owned/read-only finite strictly increasing (sample,) absolute ns;
        at least one sample. Negative/pre-pulse times model physical lookback.
        No hidden sample-period multiplication occurs in sample sums.
    demodulation_frequency_ghz: absolute oscillator cycles/ns. Rotating output is
        multiplied by exp(+2πi*(f_demod-f_ref)*t-i*phase_rad), then conjugated
        into microwave I+iQ before receiver gain/phase/offset and added noise.
    phase_rad: demod oscillator phase at absolute origin zero, radians.
    Empty/nonfinite/unordered grids raise ValueError.
    """

    times_ns: RealArray
    demodulation_frequency_ghz: float
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        times = readonly_array(self.times_ns, "sample times")
        if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError("sample grid must be nonempty and strictly increasing")
        if not np.isfinite([self.demodulation_frequency_ghz, self.phase_rad]).all():
            raise ValueError("demodulation settings must be finite")
        object.__setattr__(self, "times_ns", times)


@dataclass(frozen=True)
class PopulationInput:
    """Classical joint-state mixture, independently usable without evolution.

    state_labels: explicit order matching ReadoutModel.state_labels exactly.
    values: owned/read-only dimensionless (state,) probabilities, nonnegative
        and summing to one within 1e-6 (inclusive float64 endpoints).
        The caller supplies an explicitly normalized distribution; this record
        only validates and copies it. No clipping or renormalization occurs.
        Convert density explicitly with evolution.populations if appropriate;
        ADC does not infer a basis or turn coherences into probabilities.
        These describe the earliest readout pulse start_ns, or origin zero when
        pulses are empty. Earlier lookback samples do not advance the mixture.
    Invalid values/order at the ADC seam raise ValueError.
    """

    state_labels: tuple[str, ...]
    values: RealArray

    def __post_init__(self) -> None:
        values = readonly_array(self.values, "populations")
        if (
            values.shape != (len(self.state_labels),)
            or not len(values)
            or len(set(self.state_labels)) != len(values)
        ):
            raise ValueError("population vector must match unique state labels")
        validate_populations(values)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ConditionalResponse:
    """Deterministic state-conditioned readout, before initial-state/noise draws.

    state_labels: initial/final state order, matching ReadoutModel.
    traces: owned/read-only float64 (initial_state,sample,2) ADC sample units,
        last axis microwave [I,Q], after delay/demodulation/conjugation and
        gain/phase/offset, without noise.
    final_populations: owned/read-only float64 (initial_state,final_state) at
        the final sample's emitted time (ADC time minus time_of_flight_ns),
        no earlier than readout origin. Retains solver output without repair.
    These are conditional means, not jump-time trajectories or single-shot tails.
    """

    state_labels: tuple[str, ...]
    traces: RealArray
    final_populations: RealArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "traces", readonly_array(self.traces, "conditional traces")
        )
        object.__setattr__(
            self,
            "final_populations",
            readonly_array(self.final_populations, "conditional populations"),
        )


@dataclass(frozen=True)
class ADCSamples:
    """One realization of float64 samples and their unnormalized discrete sums.

    traces: owned/read-only (shot,sample,2) ADC sample units, final axis [I,Q].
    sums: owned/read-only (shot,2), exact sum of traces over sample axis. No ns
        factor, length normalization, QICK rounding/int64 or rep averaging.
    initial_states: owned/read-only (shot,) int64 state indices in model order,
        one initial joint-state draw per shot, useful for explicit diagnostics.
    identity: immutable identity used for the shared state/noise realization.
    """

    traces: RealArray
    sums: RealArray
    initial_states: NDArray[np.int64]
    identity: SamplingIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", readonly_array(self.traces, "ADC traces"))
        object.__setattr__(self, "sums", readonly_array(self.sums, "ADC sums"))
        states = np.array(self.initial_states, dtype=np.int64, copy=True)
        states.setflags(write=False)
        object.__setattr__(self, "initial_states", states)
