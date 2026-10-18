"""Continuous controls on an absolute event grid, independent of QICK."""

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def quantize_ns(time_ns: float) -> float:
    """Round finite absolute ns to the nearest 2 ns, ties to even, from origin zero.

    Negative times are allowed. Non-finite time_ns raises ValueError.
    """
    if not np.isfinite(time_ns):
        raise ValueError("event time must be finite")
    return float(2 * np.rint(time_ns / 2))


@dataclass(frozen=True)
class Envelope:
    """Continuous complex envelope description in pulse-local ns.

    value: pure scalar callable of local time in ns, returning signed complex
        dimensionless amplitude. Caller owns callable lifetime and immutability.
    constant: constant value, or None for time-varying envelopes. This is used
        for exact static/periodic algorithm selection and must match value.
        Prefer the constructors rather than setting this optimization hint.
    identity: reproducible callable/content descriptor required for persistent
        direct records; built-in constructors provide it. Never use callable repr.
    No amplitude normalization occurs. Nonfinite evaluations raise ValueError.
    """

    value: Callable[[float], complex]
    constant: complex | None = None
    identity: str | None = None

    def at(self, local_ns: float) -> complex:
        """Evaluate the unchanged continuous envelope at pulse-local ns."""
        result = complex(self.value(local_ns))
        if not np.isfinite(result):
            raise ValueError("envelope must be finite")
        return result

    @classmethod
    def flat(cls, amplitude: complex = 1.0) -> "Envelope":
        """Return a constant signed dimensionless envelope without normalization."""
        if not np.isfinite(amplitude):
            raise ValueError("envelope amplitude must be finite")
        return cls(lambda t: amplitude, amplitude, f"flat:{complex(amplitude)!s}")

    @classmethod
    def gaussian(cls, *, sigma_ns: float, center_ns: float) -> "Envelope":
        """Return exp(-0.5*((local_ns-center_ns)/sigma_ns)**2), untruncated.

        Pulse boundaries define support. sigma_ns must be positive and both
        parameters finite; invalid values raise ValueError.
        """
        if not np.isfinite([sigma_ns, center_ns]).all() or sigma_ns <= 0:
            raise ValueError("Gaussian sigma must be positive and parameters finite")
        return cls(
            lambda t: complex(np.exp(-0.5 * ((t - center_ns) / sigma_ns) ** 2)),
            identity=f"gaussian:{float(sigma_ns).hex()}:{float(center_ns).hex()}",
        )

    @classmethod
    def sampled(
        cls,
        times_ns: NDArray[np.float64],
        values: NDArray[np.complex128],
        *,
        outside: Literal["zero", "hold"] = "zero",
    ) -> "Envelope":
        """Interpolate complex samples linearly on their original local ns axis.

        outside: "zero" returns zero beyond the samples; "hold" extends endpoint
        values (for program linear playback). Own copies of increasing finite
        (sample,) times and matching signed complex values. Extension beyond the
        sample range follows outside. The pulse should
        declare semantic breakpoints if sample corners require solver boundaries;
        samples are not resampled onto the 2 ns event grid.
        """
        if outside not in ("zero", "hold"):
            raise ValueError("unsupported sampled envelope extension")
        times = np.array(times_ns, dtype=float, copy=True)
        samples = np.array(values, dtype=complex, copy=True)
        if (
            times.ndim != 1
            or len(times) < 2
            or samples.shape != times.shape
            or not np.isfinite(times).all()
            or not np.isfinite(samples).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "sampled envelope requires matching finite increasing arrays"
            )
        times.setflags(write=False)
        samples.setflags(write=False)
        identity = sha256(
            times.astype("<f8").tobytes()
            + samples.astype("<c16").tobytes()
            + outside.encode()
        ).hexdigest()
        left, right = (0j, 0j) if outside == "zero" else (samples[0], samples[-1])
        return cls(
            lambda t: complex(np.interp(t, times, samples, left=left, right=right)),
            identity=f"sampled:{identity}",
        )


@dataclass(frozen=True)
class ControlPulse:
    """One source's continuous pulse on an absolute ns timeline.

    source: model DriveSource name; no channel or device lookup occurs here.
    start_ns/end_ns: absolute pulse support [start,end), rounded independently
        from origin zero to nearest 2 ns, ties even. Collapsed pulses are empty.
    envelope: dimensionless signed complex function of time since the quantized
        pulse start; another source's breakpoint never restarts this coordinate.
    frequency_ghz: signed carrier cycles/ns. No frequency adjustment is made.
    phase_rad: pulse-local additive carrier phase in radians, not a frame update.
    phase_mode: reset anchors carrier phase to zero at this pulse start;
        continuous inherits the previous pulse's carrier, which runs through idle
        at its last frequency. With no predecessor it runs from origin zero.
    breakpoints_ns: absolute semantic boundaries, rounded on the same origin.
        They partition integration without staircasing the envelope.
    Invalid/nonfinite parameters raise ValueError. Overlapping pulses on the same
    source are rejected by ControlSequence (use distinct sources for overlap).
    """

    source: str
    start_ns: float
    end_ns: float
    envelope: Envelope = field(default_factory=Envelope.flat)
    frequency_ghz: float = 0.0
    phase_rad: float = 0.0
    phase_mode: Literal["reset", "continuous"] = "reset"
    breakpoints_ns: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.source
            or not np.isfinite(
                [
                    self.start_ns,
                    self.end_ns,
                    self.frequency_ghz,
                    self.phase_rad,
                    *self.breakpoints_ns,
                ]
            ).all()
        ):
            raise ValueError("pulse source and finite parameters required")
        if self.end_ns < self.start_ns or self.phase_mode not in (
            "reset",
            "continuous",
        ):
            raise ValueError(
                "pulse requires ordered endpoints and supported phase mode"
            )
        if any(not self.start_ns <= t <= self.end_ns for t in self.breakpoints_ns):
            raise ValueError("pulse breakpoints must lie inside support")


@dataclass(frozen=True)
class PhaseEvent:
    """Instantaneous virtual-Z/frame update; does not rotate density directly.

    source: model source whose later pulse phases receive the update.
    time_ns: absolute ns, nearest-2-ns quantized from origin zero.
    delta_rad: additive frame phase in radians, persistent even across carrier
        resets. All coincident events accumulate in input order, not discarded.
    """

    source: str
    time_ns: float
    delta_rad: float

    def __post_init__(self) -> None:
        if not self.source or not np.isfinite([self.time_ns, self.delta_rad]).all():
            raise ValueError("phase event requires source and finite time/phase")


@dataclass(frozen=True)
class ControlSequence:
    """Complete absolute controls, reusable across explicit shot fragments.

    pulses: immutable pulse descriptions, independent across sources. Same-source
        supports must not overlap after quantization. Input order need not be
        chronological; original events are retained at coincident coordinates.
    phase_events: additive source frame updates, including events before an
        evolve call's start. Pass the same complete sequence to each fragment so
        carrier/frame state is reconstructed without implicit instance memory.
    All temporal quantities use ns and all phases use radians.
    """

    pulses: tuple[ControlPulse, ...] = ()
    phase_events: tuple[PhaseEvent, ...] = ()

    def __post_init__(self) -> None:
        ends: dict[str, float] = {}
        for pulse in sorted(self.pulses, key=lambda p: p.start_ns):
            start, end = quantize_ns(pulse.start_ns), quantize_ns(pulse.end_ns)
            if start < ends.get(pulse.source, -np.inf):
                raise ValueError("same-source pulses must not overlap")
            ends[pulse.source] = end

    def amplitude(self, pulse: ControlPulse, time_ns: float) -> complex:
        """Evaluate an active pulse's envelope, carrier and persistent frame.

        time_ns is absolute physical ns and may be within a solver interval.
        Caller selects active support; the function also evaluates endpoint limits
        from within that interval. The sequence, not prior evolve calls, owns
        the complete phase history. No density or mutable phase is stored.
        """
        previous_time, previous_phase, previous_frequency = (
            0.0,
            0.0,
            pulse.frequency_ghz,
        )
        anchor = 0.0
        for item in sorted(self.pulses, key=lambda p: p.start_ns):
            if item.source != pulse.source:
                continue
            begin = quantize_ns(item.start_ns)
            anchor = (
                0.0
                if item.phase_mode == "reset"
                else previous_phase
                + 2 * np.pi * previous_frequency * (begin - previous_time)
            )
            if item is pulse:
                break
            previous_time, previous_phase, previous_frequency = (
                begin,
                anchor,
                item.frequency_ghz,
            )
        local = time_ns - quantize_ns(pulse.start_ns)
        frame = sum(
            event.delta_rad
            for event in self.phase_events
            if event.source == pulse.source and quantize_ns(event.time_ns) <= time_ns
        )
        phase = (
            anchor + 2 * np.pi * pulse.frequency_ghz * local + pulse.phase_rad + frame
        )
        return pulse.envelope.at(local) * complex(np.exp(1j * phase))
