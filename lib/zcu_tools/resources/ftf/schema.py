"""Named immutable FTF YAML values; no numeric-simulator dependency.

All energies use H/h GHz, reduced flux is Phi/Phi0. Integer truncations and
solver accuracy are user choices, not convergence certification. Unknown fields
and nonfinite numbers fail at this boundary; no expressions or QICK dicts.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Generic, Literal, Self, TypeVar

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
    field_validator,
    model_validator,
)

Finite = Annotated[float, Field(allow_inf_nan=False, strict=True)]
Positive = Annotated[Finite, Field(gt=0)]
Nonnegative = Annotated[Finite, Field(ge=0)]
Count = Annotated[int, Field(gt=0, strict=True)]
Index = Annotated[int, Field(ge=0, strict=True)]
Name = Annotated[str, Field(min_length=1)]
LoopValue = TypeVar("LoopValue")

SourceValue = TypeVar("SourceValue")


def _immutable_sources(value: Mapping[str, SourceValue]) -> Mapping[str, SourceValue]:
    return MappingProxyType(dict(value))


def _serialize_sources(value: Mapping[str, SourceValue]) -> dict[str, SourceValue]:
    return dict(value)


# Both named source families must stay immutable while retaining JSON replay.
SourceMap = Annotated[
    Mapping[Name, SourceValue],
    AfterValidator(_immutable_sources),
    PlainSerializer(_serialize_sources),
]


class Values(BaseModel):
    """Frozen extra-forbidden YAML record; nested records share this policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Fluxonium(Values):
    """Fluxonium physical values.

    ej_ghz/ec_ghz/el_ghz: positive GHz. q_cap/q_ind: positive constant Q, omitted
    disables. xqp: nonnegative fraction, default 0 (gap 3.4e-4 eV). aphi:
    nonnegative independent-loop Phi0/sqrt(Hz) noise amplitude, default 0.
    """

    ej_ghz: Positive
    ec_ghz: Positive
    el_ghz: Positive
    q_cap: Positive | None = None
    q_ind: Positive | None = None
    xqp: Nonnegative = 0.0
    aphi: Nonnegative = 0.0


class Squid(Values):
    """Fixed-theta SQUID values; no fluxonium inductive/qp mechanism.

    ejmax_ghz/ec_ghz: positive GHz; asymmetry in [0,1]. ng: Cooper-pair
    offset (default 0). q_cap: optional cap Q.
    resistance_ohm: optional charge-line real impedance. flux_resistance_ohm and
    mutual_phi0_per_a: paired flux-line parameters, default disabled. aphi:
    nonnegative independent-loop Phi0/sqrt(Hz), default 0.
    """

    ejmax_ghz: Positive
    ec_ghz: Positive
    asymmetry: Annotated[Finite, Field(ge=0, le=1)]
    ng: Finite = 0.0
    q_cap: Positive | None = None
    resistance_ohm: Positive | None = None
    flux_resistance_ohm: Positive | None = None
    mutual_phi0_per_a: Finite | None = None
    aphi: Nonnegative = 0.0

    @model_validator(mode="after")
    def consistency(self) -> Self:
        if (self.flux_resistance_ohm is None) != (self.mutual_phi0_per_a is None):
            raise ValueError("flux_resistance_ohm and mutual_phi0_per_a must be paired")
        return self


class LoopMap(Values, Generic[LoopValue]):
    """Required f1/coupler/f2 values, keyed by the affected circuit, not by line.

    The value type defines units; explicit zeros represent an unaffected loop.
    """

    f1: LoopValue
    coupler: LoopValue
    f2: LoopValue

    def as_tuple(self) -> tuple[LoopValue, LoopValue, LoopValue]:
        """Return fixed F1/coupler/F2 order for numeric boundaries."""
        return self.f1, self.coupler, self.f2


class Chip(Values):
    """f1/coupler/f2 physical circuits; signed full-charge couplings in GHz.

    g_f1_coupler_ghz/g_f2_coupler_ghz/g_f1_f2_ghz multiply full n*n.
    temperature_k is common nonnegative Kelvin. Truncations live in numerics.
    """

    f1: Fluxonium
    coupler: Squid
    f2: Fluxonium
    g_f1_coupler_ghz: Finite
    g_f2_coupler_ghz: Finite
    g_f1_f2_ghz: Finite
    temperature_k: Nonnegative


class DCSource(Values):
    """The map key is the live device name. unit: V or A, fixed at construction.

    limits: finite ordered inclusive (min,max), in unit. initial: startup-only
    value inside limits; ramp_step: positive increment in the same unit.
    flux_per_unit: signed Phi0/V or Phi0/A, matching unit, for all three loops.
    """

    unit: Literal["V", "A"]
    limits: tuple[Finite, Finite]
    initial: Finite
    ramp_step: Positive
    flux_per_unit: LoopMap[Finite]

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if self.limits[0] >= self.limits[1]:
            raise ValueError("limits must be increasing")
        if not self.limits[0] <= self.initial <= self.limits[1]:
            raise ValueError("initial: must lie within limits")
        return self


class DCBias(Values):
    """offset_phi0: reduced-flux offsets. sources: named fixed-unit DC devices.

    Each source contributes its live value times its flux_per_unit loop map.
    Names are nonempty and unique across DC sources and drive lines.
    """

    offset_phi0: LoopMap[Finite]
    sources: SourceMap[DCSource]


class ComplexCoefficient(Values):
    """Finite signed real/imag coefficient; the containing map defines units."""

    real: Finite
    imag: Finite


class DriveLine(Values):
    """One generator with charge or fixed-DC small-signal flux calibration.

    gen_ch: unique nonnegative generator. operator: charge (default) or flux.
    charge_drive_ghz_per_gain: required only for charge, multiplies full n.
    flux_drive_phi0_per_gain: required only for flux, multiplies d(H/h)/df at DC
    in GHz/Phi0, where f = Phi/Phi0. The unused map must be absent, even if null.
    Both maps require real/imag for F1/C/F2; explicit zeros disable coupling.
    Uses Re[c*g*w*exp(i carrier)]*operator, without hidden n01 or RWA factors.
    Flux is a linearized drive, not a time-dependent large-amplitude H(Phi(t)).
    """

    gen_ch: Index
    operator: Literal["charge", "flux"] = "charge"
    charge_drive_ghz_per_gain: LoopMap[ComplexCoefficient] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    flux_drive_phi0_per_gain: LoopMap[ComplexCoefficient] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    def coefficients(self) -> LoopMap[ComplexCoefficient]:
        """Return the selected calibration, in GHz/gain or Phi0/gain.

        Reject a missing/null active map or any supplied inactive map with
        ValueError naming the field. This is also the schema pairing invariant.
        """
        if self.operator == "charge":
            selected = self.charge_drive_ghz_per_gain
            required = "charge_drive_ghz_per_gain"
            forbidden = "flux_drive_phi0_per_gain"
        else:
            selected = self.flux_drive_phi0_per_gain
            required = "flux_drive_phi0_per_gain"
            forbidden = "charge_drive_ghz_per_gain"
        if selected is None:
            raise ValueError(f"{required}: required for operator {self.operator}")
        if forbidden in self.model_fields_set:
            raise ValueError(f"{forbidden}: forbidden for operator {self.operator}")
        return selected

    @model_validator(mode="after")
    def calibration(self) -> Self:
        """Validate operator/calibration pairing and return this frozen line."""
        self.coefficients()
        return self


class ReadoutSource(Values):
    """kind: pulse (default) or mux fixed compile capability.\n\n    gen_ch: one generator serving both resonators; input_per_gain: signed

    finite incident amplitude sqrt(photons/ns) per program gain. Carrier/envelope
    and phase come only from the program, not this configuration.
    """

    # Omitting the pulse default preserves existing settings/replay identities.
    kind: Literal["pulse", "mux"] = Field(
        default="pulse", exclude_if=lambda value: value == "pulse"
    )
    gen_ch: Index
    input_per_gain: Finite


class Receiver(Values):
    """ro_ch: one logical ADC. gain: signed field-to-ADC sample gain, may be zero.

    phase_deg: additive receiver phase (default 0). offset_adc: per-sample I/Q
    offset (default zero). noise_psd_adc2_ns: required symmetric PSD 2x2 spectral
    density ADC²*ns, two-sided white noise convention Cov(sample)=PSD/dt_ns.
    Zero/singular matrices are supported. This is not per-sample covariance.
    time_of_flight_ns: finite nonnegative total readout-line delay in ns, default
    zero. Delays the waveform and gives exp(-2πi*f*delay) at absolute frequency f;
    there is no independent electronic-delay parameter. Calibration/noise follow
    the demodulated field's conversion to microwave I+iQ by conjugation.
    """

    ro_ch: Index
    gain: Finite
    phase_deg: Finite = 0.0
    time_of_flight_ns: Nonnegative = 0.0
    offset_adc: tuple[Finite, Finite] = (0.0, 0.0)
    noise_psd_adc2_ns: tuple[tuple[Finite, Finite], tuple[Finite, Finite]]

    @model_validator(mode="after")
    def covariance(self) -> Self:
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


class Resonator(Values):
    """frequency_ghz: positive bare GHz. charge_coefficients_ghz: named F1/C/F2 full n

    coupling GHz, never hidden g01. kappa_external_mhz/kappa_internal_mhz:
    nonnegative linewidth kappa/(2*pi) MHz, total positive. thermal_occupation:
    nonnegative residual nth; omitted uses Bose(common Temp,bare f), explicit 0
    disables. Both resonators share one reciprocal co-located two-port line.
    mismatch_phase_rad: finite phase in radians, default zero (symmetric line).
    Rotates this resonator's scattering relative to direct transmission, with
    the hanger model's +phi sign in microwave IQ; linewidth/dynamics stay fixed.
    """

    frequency_ghz: Positive
    charge_coefficients_ghz: LoopMap[Finite]
    kappa_external_mhz: Nonnegative
    kappa_internal_mhz: Nonnegative
    thermal_occupation: Nonnegative | None = None
    mismatch_phase_rad: Finite = 0.0

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


class Resonators(Values):
    """r1/r2: exactly two resonators sharing one reciprocal readout line."""

    r1: Resonator
    r2: Resonator


class Readout(Values):
    """drive: shared incident field and pulse/mux compile capability.

    resonators: the physical pair, independent of tone count. adc: one receiver
    object for pulse, or a nonempty list of at most eight receivers for mux.
    Channel indices must be unique; program tone order determines output order.
    """

    drive: ReadoutSource
    resonators: Resonators
    adc: Receiver | tuple[Receiver, ...]

    @field_validator("adc", mode="before")
    @classmethod
    def parse_receivers(cls, value: object) -> Receiver | tuple[Receiver, ...]:
        """Select the external container before validation to retain field paths.

        A record validates as one receiver, sequences as a receiver tuple. The
        kind validator then enforces which shape is legal for this readout.
        """
        if isinstance(value, (Mapping, Receiver)):
            return Receiver.model_validate(value)
        return TypeAdapter(tuple[Receiver, ...]).validate_python(value)

    @model_validator(mode="after")
    def receiver_shape(self) -> Self:
        """Reject kind/ADC shape mismatches and duplicate mux receiver routes."""
        if self.drive.kind == "pulse":
            if not isinstance(self.adc, Receiver):
                raise ValueError("adc: pulse requires one receiver object")
        else:
            if not isinstance(self.adc, tuple) or not 1 <= len(self.adc) <= 8:
                raise ValueError("adc: mux requires a list of 1..8 receivers")
            if len({receiver.ro_ch for receiver in self.adc}) != len(self.adc):
                raise ValueError("adc: duplicate receiver channel")
        return self

    @property
    def receivers(self) -> tuple[Receiver, ...]:
        """Return all bound receivers as an immutable tuple, including pulse."""
        return (self.adc,) if isinstance(self.adc, Receiver) else self.adc


class FluxoniumTruncation(Values):
    """native_cutoff: oscillator dimension >=2; levels: local retained dimension."""

    native_cutoff: Annotated[int, Field(ge=2, strict=True)]
    levels: Count

    @model_validator(mode="after")
    def dimensions(self) -> Self:
        """Reject local levels larger than the oscillator basis."""
        if self.levels > self.native_cutoff:
            raise ValueError("levels exceeds native_cutoff")
        return self


class SquidTruncation(Values):
    """charge_cutoff: ncut>=1; levels: local dimension <=2*ncut+1."""

    charge_cutoff: Count
    levels: Count

    @model_validator(mode="after")
    def dimensions(self) -> Self:
        """Reject local levels larger than the native charge basis."""
        if self.levels > 2 * self.charge_cutoff + 1:
            raise ValueError("levels exceeds native charge dimension")
        return self


class Truncation(Values):
    """Local bases for f1/coupler/f2; retained_levels: system dimension <=product."""

    f1: FluxoniumTruncation
    coupler: SquidTruncation
    f2: FluxoniumTruncation
    retained_levels: Count

    @model_validator(mode="after")
    def dimension(self) -> Self:
        """Reject a system dimension exceeding the product of local levels."""
        if self.retained_levels > self.f1.levels * self.coupler.levels * self.f2.levels:
            raise ValueError("retained_levels exceeds local product")
        return self


class Numerics(Values):
    """truncation: basis sizes; solver: accuracy; seed: nonnegative root RNG seed.

    Omitted/None seed generates once at environment startup, not once per round.
    """

    truncation: Truncation
    solver: Solver
    seed: Index | None = None


class FTFSettings(Values):
    """Complete FTF v2 domain snapshot, not a generic environment DSL.

    format='zcu.ftf', version=2 (integer only). chip supplies physical values;
    dc_bias supplies startup/live DC calibration; drive_lines binds charge
    or small-signal flux generators (omitted/empty disables). readout serves two
    resonators with one drive and one or more ADCs. numerics supplies
    truncations/solver/seed.
    Channel counts derive from maximum configured indices plus one. Unknown
    fields, including large-amplitude flux controls and v1 aliases, are rejected.
    No model truth enters ordinary GUI/MCP metadata.
    """

    format: Literal["zcu.ftf"]
    version: Literal[2]
    chip: Chip
    dc_bias: DCBias
    drive_lines: SourceMap[DriveLine] = Field(
        default_factory=dict, validate_default=True
    )
    readout: Readout
    numerics: Numerics

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        """Reject noninteger YAML versions before checking the supported value."""
        if type(value) is not int:
            raise ValueError("must be integer 2")
        return value
