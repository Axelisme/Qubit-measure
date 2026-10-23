"""Immutable single-qubit physical settings, units and wiring constraints.

These values describe one fluxonium, not a generic topology. Basis size and
solver accuracy are user choices; successful validation does not certify
convergence or the physical validity of a dispersive/Markov approximation.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from zcu_tools.resources.simulation_schema import (
    ComplexCoefficient,
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


class Chip(Values):
    """qubit: one fluxonium's energies/loss parameters; temperature_k: common K >=0.

    The qubit's full charge operator defines both microwave and resonator
    couplings. No coupler, second qubit or inter-qubit coupling is supported.
    """

    qubit: Fluxonium
    temperature_k: Nonnegative


class DCSource(Values):
    """One startup/live DC source; name is the nonempty registered device name.

    unit: A or V, fixed for its lifetime. limits: increasing inclusive bounds
    in that unit; initial: startup value within limits. ramp_step: positive
    increment in the same unit. flux_per_unit: signed finite Phi0/A or Phi0/V,
    matching unit. The instantaneous contribution is live value * flux_per_unit.
    """

    name: Name
    unit: Literal["A", "V"]
    limits: tuple[Finite, Finite]
    initial: Finite
    ramp_step: Positive
    flux_per_unit: Finite

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.limits[0] >= self.limits[1]:
            raise ValueError("limits: must be increasing")
        if not self.limits[0] <= self.initial <= self.limits[1]:
            raise ValueError("initial: must lie within limits")
        return self


class DCBias(Values):
    """offset_phi0: finite reduced-flux offset; source: the single DC device.

    DC reduced flux equals offset_phi0 + source.flux_per_unit * live value.
    Initial is applied only at startup; it is not a per-round reset.
    """

    offset_phi0: Finite
    source: DCSource


class DriveLine(Values):
    """One microwave generator with a full-charge calibration.

    name: nonempty unique source name. gen_ch: nonnegative unique generator
    index. charge_drive_ghz_per_gain: finite complex GHz/gain multiplying full n
    as Re[c * gain * envelope * exp(i carrier)] * n. No hidden n01 normalization
    or rotating-wave factor is included. Program supplies carrier and phase.
    """

    name: Name
    gen_ch: Index
    charge_drive_ghz_per_gain: ComplexCoefficient


class DistortionTerm(Values):
    """amplitude: finite signed dimensionless exponential step-response weight.

    tau_ns: strictly positive finite decay time in ns. Multiple terms add;
    neither individual amplitudes nor their sum are constrained to [0,1].
    """

    amplitude: Finite
    tau_ns: Positive


class FluxLine(Values):
    """One fast-flux generator with causal delay and exponential response.

    name: nonempty unique source name. gen_ch: nonnegative unique generator.
    flux_phi0_per_gain: signed finite reduced flux per program gain. delay_ns:
    nonnegative physical delay, default 0. tau_ns: nonnegative base response
    time, default 0 (instantaneous). distortion_terms: additional signed
    exponential weights/time constants, default empty. After the delay, the
    unit-step response is 1-(1-sum(A))*exp(-t/tau_ns)-sum(A*exp(-t/tau_k));
    tau_ns=0 means its exponential vanishes for t>=0. No shot memory or gain
    applicability range is inferred from these calibrations.
    """

    name: Name
    gen_ch: Index
    flux_phi0_per_gain: Finite
    delay_ns: Nonnegative = 0.0
    tau_ns: Nonnegative = 0.0
    distortion_terms: tuple[DistortionTerm, ...] = ()


class ReadoutSource(Values):
    """gen_ch: nonnegative generator index reserved for incident readout.

    input_per_gain: finite signed incident field sqrt(photons/ns)/gain.
    Program supplies pulse envelope, carrier and phase; no implicit readout.
    The environment planner chooses pulse or coherent mux interpretation.
    """

    gen_ch: Index
    input_per_gain: Finite


class Resonator(Values):
    """One dispersive linear readout cavity coupled to the qubit's full n.

    frequency_ghz: positive bare H/h GHz. charge_coefficient_ghz: finite signed
    coupling GHz (not g01). kappa_external_mhz: positive external linewidth
    kappa/(2*pi) MHz; kappa_internal_mhz: nonnegative internal linewidth in the
    same convention. thermal_occupation: optional nonnegative residual photons;
    omitted/None uses Bose(temperature_k, bare frequency), zero disables.
    mismatch_phase_rad: finite scattering phase in radians, default zero;
    rotates the scattering contribution without changing cavity dynamics.
    """

    frequency_ghz: Positive
    charge_coefficient_ghz: Finite
    kappa_external_mhz: Positive
    kappa_internal_mhz: Nonnegative
    thermal_occupation: Nonnegative | None = None
    mismatch_phase_rad: Finite = 0.0


class Readout(Values):
    """drive: one incident generator route; resonator: one physical cavity.

    adc: one logical receiver with gain/phase/offset and two-sided noise PSD.
    ADC and generator indices are separate namespaces; equal indices are valid.
    """

    drive: ReadoutSource
    resonator: Resonator
    adc: Receiver


class Truncation(Values):
    """native_cutoff: integer harmonic-oscillator basis dimension >=2.

    levels: retained lowest eigenstates at the DC reference flux, integer
    2..native_cutoff. This same fixed projected space is used for fast flux;
    it is an accuracy parameter, not just a display count. Increase it to check
    projection error away from DC. Neither field has an implicit default.
    """

    native_cutoff: Annotated[int, Field(ge=2, strict=True)]
    levels: Annotated[int, Field(ge=2, strict=True)]

    @model_validator(mode="after")
    def _dimensions(self) -> Self:
        if self.levels > self.native_cutoff:
            raise ValueError("levels: exceeds native_cutoff")
        return self


class Numerics(Values):
    """truncation: oscillator/retained sizes; solver: explicit numerical accuracy.

    seed: nonnegative integer root RNG seed or None (default: choose once at
    startup, not per round). slow_envelope_segments_per_period: strict positive
    integer midpoint subdivisions per carrier period, default 20.
    min_round_duration_s: finite nonnegative successful
    round pacing floor in seconds, default 0.2; zero disables pacing. These
    values do not guarantee convergence and do not specify any pulse sequence.
    """

    truncation: Truncation
    solver: Solver
    slow_envelope_segments_per_period: Annotated[int, Field(ge=1, strict=True)] = 20
    seed: Index | None = None
    min_round_duration_s: Nonnegative = 0.2


class SingleFluxoniumSettings(Values):
    """Complete immutable single-fluxonium v1 physical/configuration snapshot.

    format: exactly 'zcu.single-fluxonium'; version: strict integer 1. chip:
    single circuit and temperature. dc_bias: one offset and live DC source.
    drive_line: one microwave line. flux_line: one large-amplitude fast-flux
    line, or None/omitted for a DC-only circuit with no pulsed flux route.
    readout: one pulse generator/cavity/ADC. numerics: explicit basis and
    solver choices, optional seed/pacing. All sections except flux_line are required.

    Unknown fields, nonfinite numbers, colliding source names or generator
    indices are rejected. Counts derive from maximum bound channel index + 1;
    index holes are not bound sources. ADC indices do not collide with DACs.
    No devices are created and no model truth is published to GUI metadata.
    """

    format: Literal["zcu.single-fluxonium"]
    version: Literal[1]
    chip: Chip
    dc_bias: DCBias
    drive_line: DriveLine
    flux_line: FluxLine | None = None
    readout: Readout
    numerics: Numerics

    @field_validator("version", mode="before")
    @classmethod
    def _integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("must be integer 1")
        return value

    @model_validator(mode="after")
    def _wiring(self) -> Self:
        names = {self.dc_bias.source.name}
        channels = {self.readout.drive.gen_ch}
        for field, line in (
            ("drive_line", self.drive_line),
            ("flux_line", self.flux_line),
        ):
            if line is None:
                continue
            if line.name in names:
                raise ValueError(f"{field}.name: duplicate source name")
            if line.gen_ch in channels:
                raise ValueError(f"{field}.gen_ch: duplicate generator binding")
            names.add(line.name)
            channels.add(line.gen_ch)
        return self
