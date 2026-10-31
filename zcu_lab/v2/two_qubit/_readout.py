"""Captured full-tone FTF role binding shared by two-qubit runs."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta, MuxReadoutCfg


class JointRoles(ConfigBase):
    """Fixed captured component names and original full-readout tone positions.

    f1/f2 name distinct fluxonium components, coupler names their transmon,
    and mux names the MuxGroup. f1_tone/f2_tone are their original zero-based
    positions in 0..tone_count-1. All fields remain visible and readonly in
    Forms, derived from captured MuxGroup.qubits, never labels or ADCs.
    Core callers supply that same resolved mapping and tone_count, the complete
    captured group size in 1..8. Names persist with the run; no
    live alias/library lookup occurs during acquisition or offline analysis.
    """

    f1: Annotated[str, FieldMeta(readonly=True)] = Field(min_length=1)
    f2: Annotated[str, FieldMeta(readonly=True)] = Field(min_length=1)
    coupler: Annotated[str, FieldMeta(readonly=True)] = Field(min_length=1)
    mux: Annotated[str, FieldMeta(readonly=True)] = Field(min_length=1)
    f1_tone: Annotated[int, FieldMeta(readonly=True)] = Field(ge=0)
    f2_tone: Annotated[int, FieldMeta(readonly=True)] = Field(ge=0)
    tone_count: Annotated[int, FieldMeta(readonly=True)] = Field(ge=1, le=8)

    @model_validator(mode="after")
    def validate_distinct(self) -> Self:
        """Reject identical F1/F2 components or tone positions."""
        if self.f1 == self.f2 or self.f1_tone == self.f2_tone:
            raise ValueError("F1 and F2 must be distinct components and tones")
        return self

    def validate_readout(self, readout: MuxReadoutCfg) -> None:
        """Require all captured tones and in-range F1/F2 original positions.

        readout is the complete mux configuration, never a selected subset.
        Raise ValueError for a count mismatch or out-of-range role index;
        neither mapping nor readout is modified.
        """
        if self.tone_count != len(readout.tones):
            raise ValueError("Readout must retain every tone of the captured mux group")
        if max(self.f1_tone, self.f2_tone) >= len(readout.tones):
            raise ValueError("Captured F1/F2 tones exceed the complete mux readout")
