from __future__ import annotations

import logging
from typing import Union

from qick.asm_v2 import AsmInst, Label, Macro, WriteLabel, WriteReg

from ..ir.hw_semantics import needs_big_jump
from .meta import MetaMacro
from .write_reg import WriteRegOp, format_alu_op

logger = logging.getLogger(__name__)


def _emit_cond_jump(
    prog,
    *,
    label: str,
    if_cond: str,
    op: str,
) -> list[Macro]:
    """Emit a TEST followed by a JUMP to properly evaluate flags.

    On pmem_size > 2**11, lowers to `WriteLabel + TEST + JUMP ADDR=s15`
    so the semantics survive the s15 indirection.
    """
    if needs_big_jump(prog.tproccfg.get("pmem_size")):
        return [
            WriteLabel(label=label),
            AsmInst(
                inst={"CMD": "TEST", "OP": op, "UF": "1"},
                addr_inc=1,
            ),
            AsmInst(
                inst={"CMD": "JUMP", "IF": if_cond, "ADDR": "s15"},
                addr_inc=1,
            ),
        ]
    return [
        AsmInst(
            inst={"CMD": "TEST", "OP": op, "UF": "1"},
            addr_inc=1,
        ),
        AsmInst(
            inst={"CMD": "JUMP", "IF": if_cond, "LABEL": label},
            addr_inc=1,
        ),
    ]


class OpenInnerLoop(Macro):
    """Register-driven counterpart of qick's OpenLoop.

    Emitted shape (do-while + guard):

        [n is dynamic]   JUMP end -if(Z) -op(n - #0)   ; guard skip when n == 0
        REG_WR counter imm #0                          ; counter := 0
        start:
        <body>
        ...
    """

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

        out: list[Macro] = [
            MetaMacro(
                type="LOOP_START",
                name=self.name,
                info=dict(
                    counter_reg=mapped_counter, n=mapped_n, range_hint=self.range_hint
                ),
            ),
        ]

        # Guard: skip the entire loop when a runtime n equals zero. Constant
        # n is assumed positive (callers reject n <= 0).
        if isinstance(self.n, str):
            out.extend(
                _emit_cond_jump(
                    prog,
                    label=end,
                    if_cond="Z",
                    op=f"{mapped_n} - #0",
                )
            )

        out.extend(
            [
                WriteReg(dst=self.counter_reg, src=0),
                Label(label=start),
                MetaMacro(type="LOOP_BODY_START", name=self.name),
            ]
        )
        return out


class CloseInnerLoop(Macro):
    """Register-driven counterpart of qick's CloseLoop.

    Emitted shape (do-while + guard):

        ...
        <body>
        REG_WR counter op (counter + #1)               ; counter += 1
        LOOP_BODY_END
        JUMP start -if(S) -op(counter - n)             ; counter < n: loop back
        end:

    Must be paired with an :class:`OpenInnerLoop` sharing the same ``name``.
    The ``n`` argument must match the value passed to ``OpenInnerLoop``.
    """

    # fields: name (str), counter_reg (str), n (str | int)
    def __init__(self, name: str, counter_reg: str, n: int | str) -> None:
        super().__init__(name=name, counter_reg=counter_reg, n=n)

    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"

        op_str = format_alu_op(prog, self.counter_reg, "-", self.n)

        return [
            WriteRegOp(dst=self.counter_reg, lhs=self.counter_reg, op="+", rhs=1),
            MetaMacro(type="LOOP_BODY_END", name=self.name),
            *_emit_cond_jump(prog, label=start, if_cond="S", op=op_str),
            Label(label=end),
            MetaMacro(type="LOOP_END", name=self.name),
        ]
