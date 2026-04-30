from __future__ import annotations

import logging

from qick.asm_v2 import CondJump, IncReg, Jump, Label, Macro, WriteReg

logger = logging.getLogger(__name__)


class MetaMacro(Macro):
    """A macro that emits a meta instruction for the IR builder."""

    # fields: type (str), name (str)
    def translate(self, prog):  # type: ignore[override]
        # Emit a meta instruction without incrementing the instruction address.
        prog._add_asm(
            {
                "CMD": "__META__",
                "TYPE": getattr(self, "type"),
                "NAME": getattr(self, "name"),
            },
            0,
        )


class OpenInnerLoop(Macro):
    """Register-driven counterpart of qick's OpenLoop."""

    # fields: name (str), counter_reg (str), n (str | int)
    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"
        return [
            MetaMacro(type="LOOP_START", name=self.name),
            WriteReg(dst=self.counter_reg, src=0),
            Label(label=start),
            CondJump(label=end, arg1=self.counter_reg, test="NS", op="-", arg2=self.n),
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
            IncReg(dst=self.counter_reg, src=1),
            Jump(label=start),
            Label(label=end),
            MetaMacro(type="LOOP_END", name=self.name),
        ]
