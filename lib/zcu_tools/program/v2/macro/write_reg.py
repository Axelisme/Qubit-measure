from __future__ import annotations

import logging
from numbers import Integral

from qick.asm_v2 import AsmInst, Macro

logger = logging.getLogger(__name__)


class WriteRegOp(Macro):
    """REG_WR dst = <lhs op rhs>, resolving register names at expand time.

    Fills the gap in qick's WriteReg, which only supports imm or plain copy.
    lhs is always a register name; rhs is an int literal, a register name,
    or None (for a plain register copy).
    """

    # fields: dst (str), lhs (str), op (str), rhs (int | str | None)
    def expand(self, prog):  # type: ignore[override]
        dst = prog._get_reg(self.dst)
        lhs = prog._get_reg(self.lhs)
        if self.rhs is None:
            op_str = lhs
        elif isinstance(self.rhs, Integral):
            op_str = f"{lhs} {self.op} #{int(self.rhs)}"
        elif isinstance(self.rhs, str):
            op_str = f"{lhs} {self.op} {prog._get_reg(self.rhs)}"
        else:
            raise RuntimeError(f"invalid rhs: {self.rhs!r}")
        return [
            AsmInst(
                inst={"CMD": "REG_WR", "DST": dst, "SRC": "op", "OP": op_str},
                addr_inc=1,
            )
        ]
