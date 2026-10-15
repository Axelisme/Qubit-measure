"""Named immutable FTF YAML values; no numeric-simulator dependency.

All energies use H/h GHz, reduced flux is Phi/Phi0. Integer truncations and
solver accuracy are user choices, not convergence certification. Unknown fields
and nonfinite numbers fail at this boundary; no expressions or QICK dicts.
"""

from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False, strict=True)]
Positive = Annotated[Finite, Field(gt=0)]
Nonnegative = Annotated[Finite, Field(ge=0)]
Count = Annotated[int, Field(gt=0, strict=True)]
Index = Annotated[int, Field(ge=0, strict=True)]
Triple = tuple[Finite, Finite, Finite]


class Values(BaseModel):
    """Frozen extra-forbidden YAML record; nested records share this policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Fluxonium(Values):
    """Fluxonium physical and numeric values.

    ej_ghz/ec_ghz/el_ghz: positive GHz. native_cutoff: oscillator dimension >=2;
    levels: local retained dimension. q_cap/q_ind: positive constant Q, omitted
    disables. xqp: nonnegative fraction, default 0 (gap 3.4e-4 eV). aphi:
    nonnegative independent-loop Phi0/sqrt(Hz) noise amplitude, default 0.
    """

    ej_ghz: Positive
    ec_ghz: Positive
    el_ghz: Positive
    native_cutoff: Annotated[int, Field(ge=2, strict=True)]
    levels: Count
    q_cap: Positive | None = None
    q_ind: Positive | None = None
    xqp: Nonnegative = 0.0
    aphi: Nonnegative = 0.0

    @model_validator(mode="after")
    def dimensions(self) -> Self:
        if self.levels > self.native_cutoff:
            raise ValueError("levels exceeds native_cutoff")
        return self


class Squid(Values):
    """Fixed-theta SQUID values; no fluxonium inductive/qp mechanism.

    ejmax_ghz/ec_ghz: positive GHz; asymmetry in [0,1]. charge_cutoff: ncut>=1;
    levels<=2*ncut+1. ng: Cooper-pair offset (default 0). q_cap: optional cap Q.
    resistance_ohm: optional charge-line real impedance. flux_resistance_ohm and
    mutual_phi0_per_a: paired flux-line parameters, default disabled. aphi:
    nonnegative independent-loop Phi0/sqrt(Hz), default 0.
    """

    ejmax_ghz: Positive
    ec_ghz: Positive
    asymmetry: Annotated[Finite, Field(ge=0, le=1)]
    charge_cutoff: Count
    levels: Count
    ng: Finite = 0.0
    q_cap: Positive | None = None
    resistance_ohm: Positive | None = None
    flux_resistance_ohm: Positive | None = None
    mutual_phi0_per_a: Finite | None = None
    aphi: Nonnegative = 0.0

    @model_validator(mode="after")
    def consistency(self) -> Self:
        if self.levels > 2 * self.charge_cutoff + 1:
            raise ValueError("levels exceeds native charge dimension")
        if (self.flux_resistance_ohm is None) != (self.mutual_phi0_per_a is None):
            raise ValueError("flux_resistance_ohm and mutual_phi0_per_a must be paired")
        return self


class Model(Values):
    """f1/coupler/f2: local circuits; g1c/g2c/g12_ghz: full signed n*n GHz.

    retained_levels: M<=local product. temperature_k: common nonnegative Kelvin.
    """

    f1: Fluxonium
    coupler: Squid
    f2: Fluxonium
    g1c_ghz: Finite
    g2c_ghz: Finite
    g12_ghz: Finite
    retained_levels: Count
    temperature_k: Nonnegative

    @model_validator(mode="after")
    def dimension(self) -> Self:
        if self.retained_levels > self.f1.levels * self.coupler.levels * self.f2.levels:
            raise ValueError("retained_levels exceeds local product")
        return self


class Channels(Values):
    """generators/readouts: positive channel counts in fixed mock QICK profile.

    No clock, DDS, memory, channel type or firmware override is accepted.
    """

    generators: Count
    readouts: Count


class DCSource(Values):
    """name: unique live device name. unit: V or A, fixed at construction.

    limits: finite ordered inclusive (min,max), in unit. initial: startup-only
    value inside limits; rampstep: positive increment in the same unit.
    """

    name: Annotated[str, Field(min_length=1)]
    unit: Literal["V", "A"]
    limits: tuple[Finite, Finite]
    initial: Finite
    rampstep: Positive

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if self.limits[0] >= self.limits[1]:
            raise ValueError("limits must be increasing")
        if not self.limits[0] <= self.initial <= self.limits[1]:
            raise ValueError("initial: must lie within limits")
        return self


class DCColumn(Values):
    """source: defined DC device name. unit: explicit Phi0/V or Phi0/A.

    coefficients: three signed F1/C/F2 reduced-flux per source-unit coefficients.
    """

    source: str
    unit: Literal["Phi0/V", "Phi0/A"]
    coefficients: Triple


class DCMapping(Values):
    """offset: F1/C/F2 reduced-flux offsets; columns: unique named source columns."""

    offset: Triple
    columns: tuple[DCColumn, ...]


class FluxSource(Values):
    """name: unique pulse source. gen_ch: unique bound generator index.

    coefficients_phi0_per_gain: signed F1/C/F2 pulse calibration, independent of
    DC. delay_ns/tau_ns: nonnegative source-line delay/first-order lowpass, default
    zero, applied BEFORE loop mapping. Signed real waveform including carrier
    drives this line. No per-pulse normalization, no cross-shot filter memory.
    """

    name: Annotated[str, Field(min_length=1)]
    gen_ch: Index
    coefficients_phi0_per_gain: Triple
    delay_ns: Nonnegative = 0.0
    tau_ns: Nonnegative = 0.0


class MicrowaveSource(Values):
    """name: unique source; gen_ch: unique generator. coefficients_ghz_per_gain:

    F1/C/F2 (real,imag) complex lab-frame coefficients in GHz per program gain.
    Explicit zero pairs disable crosstalk. Uses Re[c*g*w*exp(i carrier)]*n, no
    hidden n01 normalization or extra RWA factor in the input.
    """

    name: Annotated[str, Field(min_length=1)]
    gen_ch: Index
    coefficients_ghz_per_gain: tuple[
        tuple[Finite, Finite], tuple[Finite, Finite], tuple[Finite, Finite]
    ]


class ReadoutSource(Values):
    """gen_ch: one generator serving both resonators; input_per_gain: signed

    finite incident amplitude sqrt(photons/ns) per program gain. Carrier/envelope
    and phase come only from the program, not this configuration.
    """

    gen_ch: Index
    input_per_gain: Finite


class Receiver(Values):
    """ro_ch: single ADC. gain: signed field-to-ADC sample gain, may be zero.

    phase_deg: additive receiver phase (default 0). offset_adc: per-sample I/Q
    offset (default zero). noise_psd_adc2_ns: required symmetric PSD 2x2 spectral
    density ADC²*ns, two-sided white noise convention Cov(sample)=PSD/dt_ns.
    Zero/singular matrices are supported. This is not per-sample covariance.
    """

    ro_ch: Index
    gain: Finite
    phase_deg: Finite = 0.0
    offset_adc: tuple[Finite, Finite] = (0.0, 0.0)
    noise_psd_adc2_ns: tuple[tuple[Finite, Finite], tuple[Finite, Finite]]

    @model_validator(mode="after")
    def covariance(self) -> Self:
        matrix = np.array(self.noise_psd_adc2_ns)
        if not np.array_equal(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() < 0:
            raise ValueError(
                "noise_psd_adc2_ns: must be symmetric positive semidefinite"
            )
        return self


class Resonator(Values):
    """frequency_ghz: positive bare GHz. charge_coefficients_ghz: F1/C/F2 full n

    coupling GHz, never hidden g01. kappa_external_mhz/kappa_internal_mhz:
    nonnegative linewidth kappa/(2*pi) MHz, total positive. thermal_occupation:
    nonnegative residual nth; omitted uses Bose(common Temp,bare f), explicit 0
    disables. Both resonators share one reciprocal co-located two-port line.
    """

    frequency_ghz: Positive
    charge_coefficients_ghz: Triple
    kappa_external_mhz: Nonnegative
    kappa_internal_mhz: Nonnegative
    thermal_occupation: Nonnegative | None = None

    @model_validator(mode="after")
    def linewidth(self) -> Self:
        if self.kappa_external_mhz + self.kappa_internal_mhz <= 0:
            raise ValueError("total linewidth must be positive")
        return self


class Solver(Values):
    """rtol/atol: positive DOP853 local tolerances; max_step_ns: positive finite

    integration cap; bath_segment_ns: positive fixed bath update interval. All
    four are explicit user choices, no convergence or applicability guarantee.
    """

    rtol: Positive
    atol: Positive
    max_step_ns: Positive
    bath_segment_ns: Positive


class FTFSettings(Values):
    """Complete trusted FTF parameter snapshot, not a generic environment DSL.

    format='zcu.ftf', version=1. channels selects counts only. model supplies
    physical/truncation data. devices and dc_mapping supply startup and live DC
    calibration. flux_sources/microwave_sources bind distinct generators (empty
    disables that control family). readout_source/receiver serve exactly two
    resonators. solver controls accuracy. seed: optional nonnegative root seed;
    None generates once at environment startup. No pulses or model truth are
    exposed by normal GUI/MCP metadata. Unknown fields are always rejected.
    """

    format: Literal["zcu.ftf"]
    version: Literal[1]
    channels: Channels
    model: Model
    devices: tuple[DCSource, ...]
    dc_mapping: DCMapping
    flux_sources: tuple[FluxSource, ...] = ()
    microwave_sources: tuple[MicrowaveSource, ...] = ()
    readout_source: ReadoutSource
    receiver: Receiver
    resonators: tuple[Resonator, Resonator]
    solver: Solver
    seed: Index | None = None
