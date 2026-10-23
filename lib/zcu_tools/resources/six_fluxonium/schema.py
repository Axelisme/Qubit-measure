"""Six independent circuits with local DC currents and one shared coil."""

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from zcu_tools.resources.simulation_schema import Finite, Index, Name, Positive, Values
from zcu_tools.resources.single_fluxonium.schema import (
    Chip,
    DriveLine,
    Numerics,
    Readout,
)


class CurrentSource(Values):
    """name: addressed driver name; unit: A. limits: inclusive bounds in A.

    initial: startup current within increasing limits. ramp_step: positive A
    increment. SixFluxoniumSettings requires coil_bias or the matching Qn_bias.
    Per-qubit local and coil conversion lives on CircuitBias.
    """

    name: Name
    unit: Literal["A"]
    limits: tuple[Finite, Finite]
    initial: Finite
    ramp_step: Positive

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.limits[0] >= self.limits[1]:
            raise ValueError("limits: must be increasing")
        if not self.limits[0] <= self.initial <= self.limits[1]:
            raise ValueError("initial: must lie within limits")
        return self


class CircuitBias(Values):
    """source: this qubit's local current driver, Qn_bias, in A.

    offset/period: local zero-flux current and signed nonzero current per Phi0,
    both in A. coil_offset/coil_period: corresponding coil calibration in A.
    Reduced flux is (I_local - offset)/period + (I_coil - coil_offset)/coil_period.
    Each driver affects only its circuit, except coil_bias, which affects all six.
    There is no pulsed flux input or local-line crosstalk.
    """

    source: CurrentSource
    offset: Finite
    period: Finite
    coil_offset: Finite
    coil_period: Finite

    @field_validator("period", "coil_period")
    @classmethod
    def _nonzero(cls, value: float) -> float:
        if value == 0:
            raise ValueError("must be nonzero")
        return value


class SixFluxoniumQubit(Values):
    """One circuit's material with local and coil DC inputs, no fast-flux route.

    format/version: single-fluxonium, strict integer 1. chip: circuit and K.
    dc_bias: local driver and both current-to-flux calibrations. drive_line: microwave
    calibration. readout: bare cavity, incident-field and ADC calibrations.
    numerics: truncation, solver and seed/pacing choices, as on single settings.
    """

    format: Literal["zcu.single-fluxonium"]
    version: Literal[1]
    chip: Chip
    dc_bias: CircuitBias
    drive_line: DriveLine
    readout: Readout
    numerics: Numerics

    @field_validator("version", mode="before")
    @classmethod
    def _integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("must be an integer")
        return value


class SixQubits(Values):
    """Q1..Q6: complete independent materials in physical cavity/channel order.

    Additional and missing members are rejected. Input mapping order has no
    effect on the physical association between circuit and channel.
    """

    Q1: SixFluxoniumQubit
    Q2: SixFluxoniumQubit
    Q3: SixFluxoniumQubit
    Q4: SixFluxoniumQubit
    Q5: SixFluxoniumQubit
    Q6: SixFluxoniumQubit

    def ordered(self) -> tuple[SixFluxoniumQubit, ...]:
        """Return Q1..Q6 in channel order, independent of input mapping order."""
        return (self.Q1, self.Q2, self.Q3, self.Q4, self.Q5, self.Q6)


class SixFluxoniumNumerics(Values):
    """Cache residency allowances in bytes, all strict nonnegative integers.

    prepare_cache_bytes: environment-wide prepare arrays and estimated metadata,
        default 256 MiB. result_memo_bytes: environment-wide exact-result arrays
        and metadata, default 64 MiB. routing_cache_bytes: environment-wide
        routing facts and metadata, default 4 MiB. Each total is divided equally
        between six independent sources, rounding down (at most five bytes unused).
    adc_cache_bytes: allowance per source for deterministic ADC responses,
        default 8 MiB, not divided. Zero disables the corresponding residency.
    No fixed upper bound is imposed. Negative/noninteger values fail validation.
    These allowances exclude active work, models and process RSS.
    """

    prepare_cache_bytes: Index = 256 * 1024**2
    result_memo_bytes: Index = 64 * 1024**2
    routing_cache_bytes: Index = 4 * 1024**2
    adc_cache_bytes: Index = 8 * 1024**2


class SixFluxoniumSettings(Values):
    """Immutable six-independent-fluxonium v1 physical/configuration snapshot.

    format: zcu.six-fluxonium; version: strict integer 1. qubits: Q1..Q6 material.
    dc_source: shared coil_bias driver in A. mux_gen_ch: integer 0. pulse_gen_ch: integer
    13; pulse_ro_ch: integer 6, dedicated to single-tone frequency sweeps.
    Q_i requires microwave gen i, mux readout gen 0 and mux ADC i-1. Microwave
    source names differ from all seven DC names. Q_i's local source is Q{i}_bias.
    Tones/pulse frequencies select the nearest bare cavity at each point;
    equal-distance routing fails at acquisition. No joint Hilbert space is used.
    Q1's seed selects the environment root stream. The maximum of the six
    min_round_duration_s values sets successful-round pacing. Per-qubit solver
    tolerances, truncations and physical calibrations remain independent.
    numerics: environment cache residency allowances, default SixFluxoniumNumerics.
    No devices are created. Validation does not certify physical convergence.
    """

    format: Literal["zcu.six-fluxonium"]
    version: Literal[1]
    qubits: SixQubits
    numerics: SixFluxoniumNumerics = Field(default_factory=SixFluxoniumNumerics)
    dc_source: CurrentSource
    mux_gen_ch: Literal[0]
    pulse_gen_ch: Literal[13]
    pulse_ro_ch: Literal[6]

    @field_validator(
        "version", "mux_gen_ch", "pulse_gen_ch", "pulse_ro_ch", mode="before"
    )
    @classmethod
    def _integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("must be an integer")
        return value

    @model_validator(mode="after")
    def _wiring(self) -> Self:
        if self.dc_source.name != "coil_bias":
            raise ValueError("dc_source.name: must equal coil_bias")
        names = {"coil_bias", *(f"Q{i}_bias" for i in range(1, 7))}
        for index, qubit in enumerate(self.qubits.ordered(), 1):
            if qubit.dc_bias.source.name != f"Q{index}_bias":
                raise ValueError(
                    f"qubits.Q{index}.dc_bias.source.name: must equal Q{index}_bias"
                )
            bindings = (
                ("drive_line.gen_ch", qubit.drive_line.gen_ch, index),
                ("readout.drive.gen_ch", qubit.readout.drive.gen_ch, self.mux_gen_ch),
                ("readout.adc.ro_ch", qubit.readout.adc.ro_ch, index - 1),
            )
            for field, actual, expected in bindings:
                if actual != expected:
                    raise ValueError(f"qubits.Q{index}.{field}: must equal {expected}")
            name: Name = qubit.drive_line.name
            if name in names:
                raise ValueError(
                    f"qubits.Q{index}.drive_line.name: duplicate source name"
                )
            names.add(name)
        return self
