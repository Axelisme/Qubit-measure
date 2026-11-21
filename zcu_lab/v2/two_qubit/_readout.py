"""Readout reset configuration shared by two-qubit runs."""

from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field
from zcu_tools.program.v2.modules import MuxHeraldResetCfg, ResetCfg

# ADR-0074: select shot masking through the existing reset field.
MuxResetCfg: TypeAlias = Annotated[
    ResetCfg | MuxHeraldResetCfg, Field(discriminator="type")
]
