"""Readout and reset declarations shared by two-qubit experiments."""

from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field
from zcu_tools.program.v2.modules import MuxHeraldResetCfg, ResetCfg

POINT_ALIAS_DESCRIPTION = (
    "The GUI uses the point aliases $qubit1 and $qubit2. "
    "Notebook's built-in qubit1 and qubit2 bindings are separate and do not set GUI aliases."
)

# ADR-0074: select shot masking through the existing reset field.
MuxResetCfg: TypeAlias = Annotated[
    ResetCfg | MuxHeraldResetCfg, Field(discriminator="type")
]
