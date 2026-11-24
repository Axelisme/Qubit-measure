"""Logical CZ corrections shared by paired RB and CZ optimization."""

from typing import Annotated

from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta


class CZPhasePair(ConfigBase):
    """Ordered finite logical Z corrections applied after every physical CZ.

    Each angle is in degrees and is usually minus the measured local-Z slope.
    Both are required. They advance logical frames without altering the
    constructive relative phase of the physical CZ pulses.
    """

    correction_qubit1_deg: Annotated[
        float,
        FieldMeta(
            label="Qubit1 Z correction",
            unit="deg",
            description="填入 local_z 實驗取得的 Qubit1 邏輯 Z correction。",
        ),
    ] = Field(allow_inf_nan=False)
    correction_qubit2_deg: Annotated[
        float,
        FieldMeta(
            label="Qubit2 Z correction",
            unit="deg",
            description="填入 local_z 實驗取得的 Qubit2 邏輯 Z correction。",
        ),
    ] = Field(allow_inf_nan=False)
