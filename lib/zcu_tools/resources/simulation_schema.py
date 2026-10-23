"""Immutable physical and numeric YAML values shared by simulation settings.

No devices or numeric models are created. All records reject unknown fields
and nonfinite numbers. Energies are H/h in GHz, physical time is ns.
"""

from collections.abc import Mapping
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False, strict=True)]
Positive = Annotated[Finite, Field(gt=0)]
Nonnegative = Annotated[Finite, Field(ge=0)]
Count = Annotated[int, Field(gt=0, strict=True)]
Index = Annotated[int, Field(ge=0, strict=True)]
Name = Annotated[str, Field(min_length=1)]


def validation_message(error: ValidationError, *, root: str) -> str:
    """Render simulation settings errors as ordered dotted-path diagnostics.

    error is a Pydantic validation failure. root labels errors with no location;
    pass an empty string to leave that location empty. Validator messages with a
    'field: message' prefix append field to the location. The returned string
    joins diagnostics with '; ' in their original order, without raising or
    changing the supplied exception.
    """
    messages = []
    for item in error.errors():
        path = ".".join(str(part) for part in item["loc"])
        message = item["msg"].removeprefix("Value error, ")
        if ": " in message:
            field, message = message.split(": ", 1)
            path = f"{path}.{field}" if path else field
        messages.append(f"{path or root}: {message}")
    return "; ".join(messages)


class Values(BaseModel):
    """Frozen extra-forbidden YAML record; nested records share this policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RoundPacing(Values):
    """Environment pacing policy shared by FTF and fluxonium numeric settings.

    time_scale: finite nonnegative physical-time multiplier, default 1.0.
    One follows physical duration, 0.1 runs ten times faster, zero skips waiting.
    The removed fixed-duration field raises with the replacement field name.
    """

    time_scale: Nonnegative = 1.0

    @model_validator(mode="before")
    @classmethod
    def reject_fixed_duration(cls, value: object) -> object:
        """Reject old fixed seconds rather than interpreting them as a multiplier."""
        if isinstance(value, Mapping) and "min_round_duration_s" in value:
            raise ValueError("min_round_duration_s: removed; use time_scale")
        return value


class Fluxonium(Values):
    """Fluxonium physical values, with no truncation or convergence guarantee.

    ej_ghz/ec_ghz/el_ghz: finite positive H/h GHz. q_cap/q_ind: finite positive
    dimensionless constant quality factors; omitted/None disables that loss.
    xqp: finite nonnegative quasiparticle fraction, default 0 (gap 3.4e-4 eV).
    aphi: finite nonnegative independent-loop Phi0/sqrt(Hz) noise amplitude,
    default 0. Zero xqp/aphi disables the corresponding loss/noise channel.
    """

    ej_ghz: Positive
    ec_ghz: Positive
    el_ghz: Positive
    q_cap: Positive | None = None
    q_ind: Positive | None = None
    xqp: Nonnegative = 0.0
    aphi: Nonnegative = 0.0


class ComplexCoefficient(Values):
    """real/imag: finite signed parts; the containing calibration defines units."""

    real: Finite
    imag: Finite


class ReceiverBackground(Values):
    """Fixed complex transmission background, applied per absolute carrier.

    ripple_depth: finite reflected amplitude in [0,1), default zero.
    ripple_period_mhz: finite positive standing-wave period in MHz, default 1000.
    ripple_phase_deg: finite standing-wave phase in degrees, default zero.
    slope_db_per_ghz: finite amplitude slope in dB/GHz, default zero.
    pivot_ghz: finite absolute GHz where the slope factor is unity, default zero.
    Zero depth and slope give a flat unity background at every frequency.
    """

    ripple_depth: Annotated[Finite, Field(ge=0, lt=1)] = 0.0
    ripple_period_mhz: Positive = 1000.0
    ripple_phase_deg: Finite = 0.0
    slope_db_per_ghz: Finite = 0.0
    pivot_ghz: Finite = 0.0


class Receiver(Values):
    """ro_ch: nonnegative integer logical ADC index, independent of generators.

    gain: finite signed ADC sample units / sqrt(photons/ns), may be zero.

    phase_deg: additive receiver phase (default 0). offset_adc: per-sample I/Q
    offset (default zero). noise_psd_adc2_ns: required symmetric PSD 2x2 spectral
    density ADC²*ns, two-sided white noise convention Cov(sample)=PSD/dt_ns.
    Zero/singular matrices are supported. This is not per-sample covariance.
    time_of_flight_ns: finite nonnegative total readout-line delay in ns, default
    zero. Delays the waveform and gives exp(-2πi*f*delay) at absolute frequency f;
    there is no independent electronic-delay parameter. Calibration/noise follow
    the demodulated field's conversion to microwave I+iQ by conjugation.
    background: optional fixed frequency-dependent transmission before gain/phase;
    omitted or empty gives unity. Does not change noise, offset or cavity fields.
    """

    ro_ch: Index
    gain: Finite
    phase_deg: Finite = 0.0
    time_of_flight_ns: Nonnegative = 0.0
    background: ReceiverBackground = Field(default_factory=ReceiverBackground)
    offset_adc: tuple[Finite, Finite] = (0.0, 0.0)
    noise_psd_adc2_ns: tuple[tuple[Finite, Finite], tuple[Finite, Finite]]

    @model_validator(mode="after")
    def covariance(self) -> Self:
        """Return this receiver, or reject a nonsymmetric/non-PSD noise matrix.

        Singular and zero PSDs are valid. ValueError names noise_psd_adc2_ns;
        eigenvalues within relative floating-point roundoff count as zero.
        No array is stored or modified.
        """
        matrix = np.array(self.noise_psd_adc2_ns)
        # Rank-one PSD matrices can acquire a negative eigenvalue of roundoff
        # size. This is a relative machine-precision test, not a noise floor.
        roundoff = 8 * np.finfo(float).eps * float(np.max(np.abs(matrix)))
        if (
            not np.array_equal(matrix, matrix.T)
            or np.linalg.eigvalsh(matrix).min() < -roundoff
        ):
            raise ValueError(
                "noise_psd_adc2_ns: must be symmetric positive semidefinite"
            )
        return self


class Solver(Values):
    """rtol/atol: finite positive relative/absolute DOP853 local tolerances.

    max_step_ns: finite positive integration step cap in ns. bath_segment_ns:
    finite positive fixed bath-update interval in ns. All four are explicit user
    choices, with no convergence or physical-applicability guarantee.
    """

    rtol: Positive
    atol: Positive
    max_step_ns: Positive
    bath_segment_ns: Positive
