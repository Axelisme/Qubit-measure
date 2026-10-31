"""Named logical CZ corrections shared by paired RB configurations."""

from __future__ import annotations

import math
from typing import Annotated, Self

from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta

from .._readout import JointRoles


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

