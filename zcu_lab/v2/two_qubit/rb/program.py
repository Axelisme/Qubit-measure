"""Lower two-qubit logical gates to ordinary pulses with sequence-local frames."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Self

from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import Module, Pulse, PulseCfg

from .._drive import DualLineCfg, dual_line_drive
from .._readout import JointRoles
from .sequence import TwoQubitGate


class CZPhasePair(ConfigBase):
    """Component-named logical Z corrections applied after every physical CZ.

    f1/f2 are the ordered source component names, not aliases or tone labels.
    correction_f1_deg/correction_f2_deg are finite logical rotation angles in
    degrees, typically minus the measured local-Z slope. They have no defaults:
    absent calibration must not silently become zero. This pair does not alter
    the physical charge pulses' constructive relative phase.
    """

    f1: Annotated[
        str,
        FieldMeta(
            label="F1 component",
            description="填入 local Z 校準所用的 F1 component name。",
        ),
    ] = Field(min_length=1)
    f2: Annotated[
        str,
        FieldMeta(
            label="F2 component",
            description="填入 local Z 校準所用的 F2 component name。",
        ),
    ] = Field(min_length=1)
    correction_f1_deg: Annotated[
        float,
        FieldMeta(
            label="F1 Z correction",
            unit="deg",
            description="填入 local_z 實驗取得的 F1 邏輯 Z correction。",
        ),
    ]
    correction_f2_deg: Annotated[
        float,
        FieldMeta(
            label="F2 Z correction",
            unit="deg",
            description="填入 local_z 實驗取得的 F2 邏輯 Z correction。",
        ),
    ]

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        """Reject identical components and nonfinite logical corrections."""
        if self.f1 == self.f2 or not all(
            math.isfinite(v) for v in (self.correction_f1_deg, self.correction_f2_deg)
        ):
            raise ValueError(
                "CZ phase pair needs distinct components and finite angles"
            )
        return self

    def validate_roles(self, roles: JointRoles) -> None:
        """Reject a pair not authored for these ordered F1/F2 names, including swaps.

        roles is the run's captured mapping. No capture, pulse, or pair is changed.
        """
        if (self.f1, self.f2) != (roles.f1, roles.f2):
            raise ValueError(
                "CZ phase pair identities must match source F1/F2 in order"
            )


def sequence_modules(
    gates: Sequence[TwoQubitGate],
    *,
    x90_f1: PulseCfg,
    x90_f2: PulseCfg,
    drive: DualLineCfg,
    phase_pair: CZPhasePair,
) -> tuple[Module, ...]:
    """Translate playback-ordered gates, starting both virtual frames at zero.

    Q0 uses x90_f1, Q1 uses x90_f2. Pulses must be calibrated positive X90s;
    their numeric phase (degrees) defines each role's physical X axis. drive is
    the ordinary synchronous CZ pair. Caller validates phase_pair identities
    against its captured roles with validate_roles before compiling.

    Logical Z(theta)=exp(-i*theta*Z/2) advances a pending logical angle theta.
    A later X pulse uses programmed phase base-theta, matching the existing RB
    convention (positive logical Z90 turns a following X90 into phase -90).
    The physical rotation axis is cos(phase) X + sin(phase) Y. Thus logical
    product = terminal Rz(theta) times emitted pulse product. Terminal Z frames
    need no hardware pulse before computational-basis readout.

    CZ commutes with both pending Z frames. Emit its two unchanged physical
    phases, then advance each pending logical angle by its named correction.
    No hardware Z pulse, reset, readout, merging, or mutation occurs. Return
    ordinary Pulse/Join modules with unique names. Reject unknown gates and
    nonfinite/nonscalar X90 phases; compilation/route failures remain the
    program owner's responsibility. Each call is independent of every other.
    """
    pulses = (x90_f1, x90_f2)
    if any(
        not isinstance(p.phase, (int, float)) or not math.isfinite(p.phase)
        for p in pulses
    ):
        raise ValueError("X90 phase must be a finite scalar in degrees")
    frames = [0.0, 0.0]
    modules: list[Module] = []
    for index, gate in enumerate(gates):
        if gate in (TwoQubitGate.Z90_Q0, TwoQubitGate.Z90_Q1):
            role = 0 if gate == TwoQubitGate.Z90_Q0 else 1
            frames[role] = (frames[role] + 90.0) % 360.0
        elif gate in (TwoQubitGate.X90_Q0, TwoQubitGate.X90_Q1):
            role = 0 if gate == TwoQubitGate.X90_Q0 else 1
            pulse = pulses[role]
            modules.append(
                Pulse(
                    f"gate_{index}_x90_f{role + 1}",
                    pulse.with_updates(phase=(pulse.phase - frames[role]) % 360.0),
                )
            )
        elif gate == TwoQubitGate.CZ:
            modules.append(
                dual_line_drive(
                    drive, frequency=float(drive.f1.freq), name=f"gate_{index}_cz"
                )
            )
            frames[0] = (frames[0] + phase_pair.correction_f1_deg) % 360.0
            frames[1] = (frames[1] + phase_pair.correction_f2_deg) % 360.0
        else:
            raise ValueError(f"Unknown two-qubit gate: {gate!r}")
    return tuple(modules)
