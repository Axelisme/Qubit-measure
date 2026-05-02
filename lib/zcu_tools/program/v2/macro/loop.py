from __future__ import annotations

import logging

from qick.asm_v2 import CondJump, IncReg, Jump, Label, Macro, WriteReg

from .meta import MetaMacro

logger = logging.getLogger(__name__)


class OpenInnerLoop(Macro):
    """Register-driven counterpart of qick's OpenLoop."""

    # fields: name (str), counter_reg (str), n (str | int)
    def __init__(
        self,
        name: str,
        counter_reg: str,
        n: int | str,
        *,
        range_hint: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(name=name, counter_reg=counter_reg, n=n, range_hint=range_hint)

    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"

        mapped_counter = prog._get_reg(self.counter_reg)
        mapped_n = prog._get_reg(self.n) if isinstance(self.n, str) else self.n

        return [
            MetaMacro(
                type="LOOP_START",
                name=self.name,
                info=dict(
                    counter_reg=mapped_counter, n=mapped_n, range_hint=self.range_hint
                ),
            ),
            WriteReg(dst=self.counter_reg, src=0),
            Label(label=start),
            CondJump(label=end, arg1=self.counter_reg, test="NS", op="-", arg2=self.n),
            MetaMacro(type="LOOP_BODY_START", name=self.name),
        ]


class CloseInnerLoop(Macro):
    """Register-driven counterpart of qick's CloseLoop.

    Emits: counter increment, unconditional jump back to loop start, end label.
    Must be paired with an OpenLoopReg sharing the same ``name``.
    """

    # fields: name (str), counter_reg (str)
    def __init__(self, name: str, counter_reg: str) -> None:
        super().__init__(name=name, counter_reg=counter_reg)

    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"
        return [
            IncReg(dst=self.counter_reg, src=1),
            MetaMacro(type="LOOP_BODY_END", name=self.name),
            Jump(label=start),
            Label(label=end),
            MetaMacro(type="LOOP_END", name=self.name),
        ]
