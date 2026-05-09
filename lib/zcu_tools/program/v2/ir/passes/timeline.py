from __future__ import annotations

import dataclasses
from typing import cast

from ..instructions import BaseInst, NopInst, TimeInst
from ..node import BasicBlockNode
from ..operands import Literal
from ..pipeline import AbsLinearPass


def _is_zero_ref_increment(inst: BaseInst) -> bool:
    if not isinstance(inst, TimeInst):
        return False
    if inst.c_op != "inc_ref":
        return False
    if inst.r1 is not None:
        return False
    if not isinstance(inst.lit, Literal):
        return False
    lit = inst.lit.value
    if not lit.startswith("#"):
        return False
    try:
        return int(lit[1:]) == 0
    except ValueError:
        return False


def _is_lit_time(inst: BaseInst) -> bool:
    """True for TIME inc_ref #N with N > 0 (no register operand)."""
    if not isinstance(inst, TimeInst):
        return False
    if (
        inst.c_op != "inc_ref"
        or inst.r1 is not None
        or not isinstance(inst.lit, Literal)
    ):
        return False
    if not inst.lit.value.startswith("#"):
        return False
    try:
        return int(inst.lit.value[1:]) > 0
    except ValueError:
        return False


def _get_lit_time_value(inst: TimeInst) -> int:
    return int(cast(Literal, inst.lit).value[1:])


def _is_reg_time(inst: BaseInst) -> bool:
    """True for TIME inc_ref rX (register-driven increment)."""
    return isinstance(inst, TimeInst) and inst.c_op == "inc_ref" and inst.r1 is not None


def _is_anchored_timed(inst: BaseInst) -> bool:
    """True when inst has a time field of the form '@N' with N a plain integer."""
    t = getattr(inst, "time", None)
    if not isinstance(t, Literal) or not t.value.startswith("@"):
        return False
    # If it's a Literal, it doesn't contain registers by definition of get_read_regs()
    return not t.get_read_regs()


def _adjust_time_field(inst: BaseInst, delta: int) -> BaseInst:
    """Return a copy of inst with time adjusted by +delta (precondition: _is_anchored_timed)."""
    old = int(cast(Literal, getattr(inst, "time")).value[1:])
    return dataclasses.replace(inst, time=Literal(f"@{old + delta}"))  # type: ignore[call-overload]


class ZeroDelayDCELinear(AbsLinearPass):
    """Remove TIME inc_ref #0 instructions from a BasicBlockNode."""

    def process_block(self, block: BasicBlockNode) -> None:
        if block.fix_addr_size:
            return
        block.insts = [
            inst for inst in block.insts if not _is_zero_ref_increment(inst)
        ]


class TimedMergeLinear(AbsLinearPass):
    """Aggressive TIME inc_ref optimisation pass."""

    def process_block(self, block: BasicBlockNode) -> None:
        if block.fix_addr_size:
            return
        self._merge_free(block)

    def _merge_free(self, block: BasicBlockNode) -> None:
        pending_lit: int = 0
        result: list[BaseInst] = []

        for inst in block.insts:
            if _is_lit_time(inst):
                pending_lit += _get_lit_time_value(cast(TimeInst, inst))
            elif _is_anchored_timed(inst):
                result.append(
                    _adjust_time_field(inst, pending_lit) if pending_lit > 0 else inst
                )
            elif _is_reg_time(inst):
                if pending_lit > 0:
                    result.append(
                        TimeInst(c_op="inc_ref", lit=Literal(f"#{pending_lit}"))
                    )
                    pending_lit = 0
                result.append(inst)
            else:
                result.append(inst)

        if pending_lit > 0:
            result.append(TimeInst(c_op="inc_ref", lit=Literal(f"#{pending_lit}")))

        block.insts = result
