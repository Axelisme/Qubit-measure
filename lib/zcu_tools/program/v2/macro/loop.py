from __future__ import annotations

import logging

from qick.asm_v2 import CondJump, IncReg, Jump, Label, Macro, WriteReg

logger = logging.getLogger(__name__)


class OpenLoopReg(Macro):
    """Register-driven counterpart of qick's OpenLoop.

    Emits: skip-if-negative guard, counter init, loop-start label, and
    exit-condition check. The loop body (whatever macros the caller appends
    next) runs between this and the matching CloseLoopReg.
    Does NOT participate in qick's sweep machinery (no loop_stack / loop_dict).
    """

    # fields: name (str), n_reg (str)
    def preprocess(self, prog) -> None:  # type: ignore[override]
        # allocate the counter register, matching qick's OpenLoop pattern
        prog.add_reg(name=self.name)

    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"
        return [
            # skip the loop entirely if n_reg is negative
            CondJump(label=end, arg1=self.n_reg, test="S", op=None, arg2=None),
            WriteReg(dst=self.name, src=0),
            Label(label=start),
            # exit when counter reaches n_reg (n_reg - count == 0)
            CondJump(label=end, arg1=self.n_reg, test="Z", op="-", arg2=self.name),
        ]


class CloseLoopReg(Macro):
    """Register-driven counterpart of qick's CloseLoop.

    Emits: counter increment, unconditional jump back to loop start, end label.
    Must be paired with an OpenLoopReg sharing the same ``name``.
    """

    # fields: name (str)
    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"
        return [
            IncReg(dst=self.name, src=1),
            Jump(label=start),
            Label(label=end),
        ]
