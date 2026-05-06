from __future__ import annotations

import logging
from numbers import Integral
from typing import Union

from qick.asm_v2 import AsmInst, Label, Macro, WriteLabel, WriteReg

from .meta import MetaMacro
from .write_reg import WriteRegOp

logger = logging.getLogger(__name__)


def _needs_big_jump(prog) -> bool:
    tproccfg = getattr(prog, "tproccfg", None)
    if not isinstance(tproccfg, dict):
        return False
    pmem_size = tproccfg.get("pmem_size")
    return isinstance(pmem_size, int) and pmem_size > 2**11


def _emit_cond_jump(
    prog,
    *,
    label: str,
    if_cond: str,
    op: str,
) -> list[Macro]:
    """Emit a condensed `JUMP -if(...) -op(...)` (single word) with big-jump fallback.

    On pmem_size > 2**11, lowers to `WriteLabel + JUMP IF=... ADDR=s15 OP=...`
    so the same condensed-IF/OP semantics survive the s15 indirection. This
    matches the IR layer's `_emit_label_jump` behavior.
    """
    if _needs_big_jump(prog):
        return [
            WriteLabel(label=label),
            AsmInst(
                inst={"CMD": "JUMP", "IF": if_cond, "OP": op, "ADDR": "s15"},
                addr_inc=1,
            ),
        ]
    return [
        AsmInst(
            inst={"CMD": "JUMP", "IF": if_cond, "OP": op, "LABEL": label},
            addr_inc=1,
        )
    ]


def _format_op(prog, lhs_reg: str, op: str, rhs: Union[int, str]) -> str:
    lhs = prog._get_reg(lhs_reg)
    if isinstance(rhs, Integral):
        return f"{lhs} {op} #{int(rhs)}"
    if isinstance(rhs, str):
        return f"{lhs} {op} {prog._get_reg(rhs)}"
    raise RuntimeError(f"invalid rhs: {rhs!r}")


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
        n: Union[int, str],
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
        JUMP start -if(NS) -op(counter - n)            ; counter < n: loop back
        end:

    Must be paired with an :class:`OpenInnerLoop` sharing the same ``name``.
    The ``n`` argument must match the value passed to ``OpenInnerLoop``.
    """

    # fields: name (str), counter_reg (str), n (str | int)
    def __init__(self, name: str, counter_reg: str, n: Union[int, str]) -> None:
        super().__init__(name=name, counter_reg=counter_reg, n=n)

    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"

        op_str = _format_op(prog, self.counter_reg, "-", self.n)

        return [
            MetaMacro(type="LOOP_BODY_END", name=self.name),
            WriteRegOp(dst=self.counter_reg, lhs=self.counter_reg, op="+", rhs=1),
            *_emit_cond_jump(prog, label=start, if_cond="NS", op=op_str),
            Label(label=end),
            MetaMacro(type="LOOP_END", name=self.name),
        ]
