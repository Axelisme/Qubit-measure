"""Full-tone FTF binding and one-round mux collection shared by FTF runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, MuxReadoutCfg
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules import MuxReadout


class JointRoles(ConfigBase):
    """Fixed captured component names and original full-readout tone positions.

    f1/f2 name distinct fluxonium components, coupler names their transmon,
    and mux names the MuxGroup. f1_tone/f2_tone are their original zero-based
    positions in 0..tone_count-1. Form derives them from MuxGroup.qubits, never
    labels or ADCs. Core
    callers supply that same resolved mapping and tone_count, the complete
    captured group size in 1..8. Names persist with the run; no
    live alias/library lookup occurs during acquisition or offline analysis.
    """

    f1: str = Field(min_length=1)
    f2: str = Field(min_length=1)
    coupler: str = Field(min_length=1)
    mux: str = Field(min_length=1)
    f1_tone: int = Field(ge=0)
    f2_tone: int = Field(ge=0)
    tone_count: int = Field(ge=1, le=8)

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


def collect_joint_shots(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    """Convert this initialized program's completed round, not averaged IQ.

    acquired is the Schedule conversion callback input; it is intentionally not
    interpreted. Read raw solely through the named MuxReadout's public contract.
    Return (reps,N) in unchanged loop/tone order; reject nonfinite acquired IQ.
    This has no accumulation side effect, so Schedule's final conversion simply
    overwrites the same round slot instead of appending duplicate shots.
    """
    del acquired
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_raw_iq(prog, read_index=0)
            shots = iq[..., 0] + 1j * iq[..., 1]
            if shots.shape != (prog.cfg_model.reps, len(module.cfg.tones)):
                raise ValueError(
                    "Joint readout requires only the reps loop and one mux read"
                )
            if not np.all(np.isfinite(shots)):
                raise ValueError(
                    "Acquired joint IQ must be finite; NaN is only an unwritten sentinel"
                )
            return shots
    raise ValueError("Joint program is missing its named readout MuxReadout")
