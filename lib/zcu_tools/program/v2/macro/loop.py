from __future__ import annotations

import logging

from qick.asm_v2 import CondJump, IncReg, Jump, Label, Macro, WriteReg

from .meta import MetaMacro

logger = logging.getLogger(__name__)


class OpenInnerLoop(Macro):
    """Register-driven counterpart of qick's OpenLoop."""

    # fields: name (str), counter_reg (str), n (str | int)
    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"

        trip_count = self.n if isinstance(self.n, int) else None

        return [
            MetaMacro(
                type="LOOP_START", name=self.name, args={"trip_count": trip_count}
            ),
            Label(label=start),
            MetaMacro(type="LOOP_SECTION", name="initial"),
            WriteReg(dst=self.counter_reg, src=0),
            MetaMacro(type="LOOP_SECTION", name="stop_check"),
            CondJump(label=end, arg1=self.counter_reg, test="NS", op="-", arg2=self.n),
            MetaMacro(type="LOOP_SECTION", name="body"),
        ]


class CloseInnerLoop(Macro):
    """Register-driven counterpart of qick's CloseLoop.

    Emits: counter increment, unconditional jump back to loop start, end label.
    Must be paired with an OpenLoopReg sharing the same ``name``.
    """

    # fields: name (str), counter_reg (str)
    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"
        return [
            MetaMacro(type="LOOP_SECTION", name="update"),
            IncReg(dst=self.counter_reg, src=1),
            MetaMacro(type="LOOP_SECTION", name="jump_back"),
            Jump(label=start),
            Label(label=end),
            MetaMacro(type="LOOP_END", name=self.name),
        ]
