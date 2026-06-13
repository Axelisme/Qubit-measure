from __future__ import annotations

import logging
from numbers import Integral

from qick.asm_v2 import AsmInst, Macro

logger = logging.getLogger(__name__)


def format_alu_op(prog, lhs_reg: str, op: str, rhs: int | str | None) -> str:
    """Format an ALU `-op()` expression, resolving register names at expand time.

    rhs may be an int literal (`#N`), a register name (resolved via _get_reg),
    or None for a plain register copy (just the resolved lhs).
    """
    lhs = prog._get_reg(lhs_reg)
    if rhs is None:
        return lhs
    if isinstance(rhs, Integral):
        return f"{lhs} {op} #{int(rhs)}"
    if isinstance(rhs, str):
        return f"{lhs} {op} {prog._get_reg(rhs)}"
    raise RuntimeError(f"invalid rhs: {rhs!r}")


class WriteRegOp(Macro):
    """REG_WR dst = <lhs op rhs>, resolving register names at expand time.

    Fills the gap in qick's WriteReg, which only supports imm or plain copy.
    lhs is always a register name; rhs is an int literal, a register name,
    or None (for a plain register copy).
    """

    # fields: dst (str), lhs (str), op (str), rhs (int | str | None)
    def expand(self, prog):  # type: ignore[override]
        dst = prog._get_reg(self.dst)
        op_str = format_alu_op(prog, self.lhs, self.op, self.rhs)
        return [
            AsmInst(
                inst={"CMD": "REG_WR", "DST": dst, "SRC": "op", "OP": op_str},
                addr_inc=1,
            )
        ]
