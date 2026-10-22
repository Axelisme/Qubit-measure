"""Six independent circuits sharing one live DC current source."""

from typing import Literal, Self

from pydantic import field_validator, model_validator

from zcu_tools.resources.simulation_schema import Finite, Name, Positive, Values
from zcu_tools.resources.single_fluxonium.schema import (
    Chip,
    DriveLine,
    Numerics,
    Readout,
)


class GlobalDCSource(Values):
    """name: flux_bias; unit: A. limits: increasing inclusive bounds in A.

    initial: startup current within limits. ramp_step: positive A increment.
    Only one driver is created. Per-qubit flux conversion lives on GlobalBias.
    """

    name: Literal["flux_bias"]
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


class GlobalBias(Values):
    """offset_phi0: reduced-flux offset; flux_per_unit: signed finite Phi0/A.

    This qubit's flux equals offset_phi0 + flux_per_unit * live global current.
    No independent source or pulsed flux input is defined.
    """

    offset_phi0: Finite
    flux_per_unit: Finite


class SixFluxoniumQubit(Values):
    """One circuit's material without a local DC driver or fast-flux route.

    format/version: single-fluxonium, strict integer 1. chip: circuit and K.
    dc_bias: this circuit's global-current conversion. drive_line: microwave
    calibration. readout: bare cavity, incident-field and ADC calibrations.
    numerics: truncation, solver and seed/pacing choices, as on single settings.
    """

    format: Literal["zcu.single-fluxonium"]
    version: Literal[1]
    chip: Chip
    dc_bias: GlobalBias
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


class SixFluxoniumSettings(Values):
    """Immutable six-independent-fluxonium v1 physical/configuration snapshot.

    format: zcu.six-fluxonium; version: strict integer 1. qubits: Q1..Q6 material.
    dc_source: one shared A driver. mux_gen_ch: integer 0. pulse_gen_ch: integer
    13; pulse_ro_ch: integer 6, dedicated to single-tone frequency sweeps.
    Q_i requires microwave gen i, mux readout gen 0 and mux ADC i-1. Microwave
    source names are globally unique and differ from the shared DC source name.
    Tones/pulse frequencies select the nearest bare cavity at each point;
    equal-distance routing fails at acquisition. No joint Hilbert space is used.
    Q1's seed selects the environment root stream. The maximum of the six
    min_round_duration_s values sets successful-round pacing. Per-qubit solver
    tolerances, truncations and physical calibrations remain independent.
    No devices are created. Validation does not certify physical convergence.
    """

    format: Literal["zcu.six-fluxonium"]
    version: Literal[1]
    qubits: SixQubits
    dc_source: GlobalDCSource
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
        names: set[str] = {self.dc_source.name}
        for index, qubit in enumerate(self.qubits.ordered(), 1):
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
