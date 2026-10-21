"""Named immutable FTF YAML values; no numeric-simulator dependency.

All energies use H/h GHz, reduced flux is Phi/Phi0. Integer truncations and
solver accuracy are user choices, not convergence certification. Unknown fields
and nonfinite numbers fail at this boundary; no expressions or QICK dicts.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Generic, Literal, Self, TypeVar

from pydantic import (
    AfterValidator,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from zcu_tools.resources.simulation_schema import (
    ComplexCoefficient,
    Count,
    Finite,
    Fluxonium,
    Index,
    Name,
    Nonnegative,
    Positive,
    Receiver,
    Solver,
    Values,
)

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


class FluxoniumDesign(Values):
    """Complete author-supplied predictor guess, independent of model truth.

    ej_ghz/ec_ghz/el_ghz are positive nominal energies in GHz. flux_unit is the
    flux-source device's native A or V, checked against entry wiring by startup.
    flux_half is the zero-bias half-flux device value; flux_period is the signed
    nonzero device-value change per quantum; flux_bias is an additive device-value
    correction. All affine inputs are finite and use flux_unit. Every field is
    required; neither chip nor DC calibration supplies defaults.
    """

    ej_ghz: Positive
    ec_ghz: Positive
    el_ghz: Positive
    flux_unit: Literal["A", "V"]
    flux_half: Finite
    flux_period: Finite
    flux_bias: Finite

    @field_validator("flux_period")
    @classmethod
    def nonzero_period(cls, value: float) -> float:
        """Reject a zero nominal period, which cannot map device values to flux."""
        if value == 0:
            raise ValueError("must be nonzero")
        return value


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
    """One incident source serving both physical resonators.

    gen_ch: nonnegative integer generator index, with no fixed upper bound.
    input_per_gain: signed finite incident amplitude sqrt(photons/ns) per program
    gain. Carrier, envelope and phase come only from the program. The containing
    readout field selects pulse or mux capability; kind has been removed.
    """

    gen_ch: Index
    input_per_gain: Finite

    @model_validator(mode="before")
    @classmethod
    def reject_removed_kind(cls, value: object) -> object:
        """Reject the removed kind selector before accepting source fields."""
        if isinstance(value, Mapping) and "kind" in value:
            raise ValueError("kind: removed; configure mux separately in readout.mux")
        return value


class MuxReadoutSource(ReadoutSource):
    """Independent mux source and its bound receiver calibrations.

    gen_ch: nonnegative integer generator, distinct from pulse and drive lines.
    input_per_gain: signed finite incident sqrt(photons/ns) per program gain.
    adc: 1..8 receivers with unique nonnegative ro_ch, disjoint from pulse ADC.
    Indices have no fixed upper bound. Program tones select receiver subsets and
    their output order; the physical resonator pair is shared with pulse.
    """

    adc: Annotated[tuple[Receiver, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def unique_receivers(self) -> Self:
        """Reject duplicate receiver channels within the mux source."""
        if len({receiver.ro_ch for receiver in self.adc}) != len(self.adc):
            raise ValueError("adc: duplicate receiver channel")
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


class Resonators(Values):
    """r1/r2: exactly two resonators sharing one reciprocal readout line."""

    r1: Resonator
    r2: Resonator


class Readout(Values):
    """Pulse readout with an optional independent mux path.

    drive: pulse incident source. adc: its single receiver object. resonators:
    the physical pair shared by both paths, independent of tone count. mux:
    optional mux source with 1..8 receivers; omission/None disables only mux.
    Generator bindings and all receiver channels must be unique across paths.
    """

    drive: ReadoutSource
    resonators: Resonators
    adc: Receiver
    mux: MuxReadoutSource | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def separate_receivers(self) -> Self:
        """Reject receiver channels shared by the pulse and mux paths."""
        if self.mux is not None and any(
            receiver.ro_ch == self.adc.ro_ch for receiver in self.mux.adc
        ):
            raise ValueError("mux.adc: duplicate receiver channel with readout.adc")
        return self

    @property
    def receivers(self) -> tuple[Receiver, ...]:
        """Return all bound ADCs, pulse first followed by configured mux order."""
        return (self.adc,) + (() if self.mux is None else self.mux.adc)


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


class TelemetrySettings(Values):
    """Local routing logging policy, separate from numerical accuracy.

    enabled: write one acquire summary, default True. directory: optional output
    path; None uses project-root .sim_telemetry. Relative paths resolve against
    the working directory when the environment is built. No I/O occurs here.
    """

    enabled: bool = True
    directory: Path | None = None


class Numerics(Values):
    """truncation: basis sizes; solver: accuracy; seed: nonnegative root RNG seed.

    Omitted/None seed generates once at environment startup, not once per round.
    min_round_duration_s: finite nonnegative wall-time floor per successful binding
    round, in seconds. Default 0.2 gives live plotting time; zero disables pacing.
    telemetry: local acquire-summary policy; omitted settings enable logging.
    """

    truncation: Truncation
    solver: Solver
    seed: Index | None = None
    min_round_duration_s: Nonnegative = 0.2
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)


class FTFSettings(Values):
    """Complete FTF v2 domain snapshot, not a generic environment DSL.

    format='zcu.ftf', version=2 (integer only). chip supplies physical values;
    dc_bias supplies startup/live DC calibration; drive_lines binds charge
    or small-signal flux generators (omitted/empty disables). readout serves two
    resonators with a pulse source/receiver and an optional mux source/receivers.
    numerics supplies truncations/solver/seed. design optionally maps entry qubit
    names to complete independent predictor guesses; omitted/empty creates none.
    Channel counts derive from maximum configured indices plus one. Unknown
    fields, including large-amplitude flux controls and v1 aliases, are rejected.
    No model truth enters ordinary GUI/MCP metadata.
    """

    format: Literal["zcu.ftf"]
    version: Literal[2]
    chip: Chip
    design: SourceMap[FluxoniumDesign] = Field(
        default_factory=dict, validate_default=True
    )
    dc_bias: DCBias
    drive_lines: SourceMap[DriveLine] = Field(
        default_factory=dict, validate_default=True
    )
    readout: Readout
    numerics: Numerics

    def with_telemetry(self, *, enabled: bool, directory: Path | None = None) -> Self:
        """Return a detached settings copy with an explicit local logging policy.

        enabled=False disables all routing timers and I/O for profiles/benchmarks.
        directory optionally overrides the output path; None preserves the current
        directory. Does not mutate this settings value, perform I/O or change
        numerical accuracy, round pacing or RNG settings.
        """
        telemetry = TelemetrySettings(
            enabled=enabled,
            directory=self.numerics.telemetry.directory
            if directory is None
            else directory,
        )
        return self.model_copy(
            update={
                "numerics": self.numerics.model_copy(update={"telemetry": telemetry})
            }
        )

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        """Reject noninteger YAML versions before checking the supported value."""
        if type(value) is not int:
            raise ValueError("must be integer 2")
        return value
