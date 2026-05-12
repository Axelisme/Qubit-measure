from __future__ import annotations

import dataclasses
from typing import cast

from ..analysis import reads_implicit_time_base
from ..instructions import BaseInst, TimeInst
from ..node import BasicBlockNode
from ..operands import Immediate, TimeOffset
from ..pipeline import AbsChunkPass, ChunkList, PipeLineContext


def _is_zero_ref_increment(inst: BaseInst) -> bool:
    if not isinstance(inst, TimeInst):
        return False
    if inst.c_op != "inc_ref":
        return False
    if inst.r1 is not None:
        return False
    return inst.lit == Immediate(0)


def _is_lit_time(inst: BaseInst) -> bool:
    """True for TIME inc_ref #N with N > 0 (no register operand)."""
    if not isinstance(inst, TimeInst):
        return False
    if (
        inst.c_op != "inc_ref"
        or inst.r1 is not None
        or not isinstance(inst.lit, Immediate)
    ):
        return False
    return inst.lit.value > 0


def _get_lit_time_value(inst: TimeInst) -> int:
    return cast(Immediate, inst.lit).value


def _is_reg_time(inst: BaseInst) -> bool:
    """True for TIME inc_ref rX (register-driven increment)."""
    return isinstance(inst, TimeInst) and inst.c_op == "inc_ref" and inst.r1 is not None


def _is_anchored_timed(inst: BaseInst) -> bool:
    """True when inst has a time field of the form '@N' with N a plain integer."""
    t = getattr(inst, "time", None)
    return isinstance(t, TimeOffset)


def _adjust_time_field(inst: BaseInst, delta: int) -> BaseInst:
    """Return a copy of inst with time adjusted by +delta (precondition: _is_anchored_timed)."""
    old = cast(TimeOffset, getattr(inst, "time")).value
    return dataclasses.replace(inst, time=TimeOffset(old + delta))  # type: ignore[call-overload]


class ZeroDelayDCEPass(AbsChunkPass):
    """Remove TIME inc_ref #0 instructions from BasicBlockNode chunks."""

    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]:
        _ = ctx
        changed = False
        for chunk in chunks:
            if not isinstance(chunk, BasicBlockNode):
                continue
            changed |= self._process_block(chunk)
        return chunks, changed

    def _process_block(self, block: BasicBlockNode) -> bool:
        if block.fix_addr_size:
            return False
        before = list(block.insts)
        block.insts = [inst for inst in block.insts if not _is_zero_ref_increment(inst)]
        return before != block.insts


class TimedMergePass(AbsChunkPass):
    """Aggressive TIME inc_ref optimisation pass."""

    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]:
        _ = ctx
        changed = False
        for chunk in chunks:
            if not isinstance(chunk, BasicBlockNode):
                continue
            changed |= self._process_block(chunk)
        return chunks, changed

    def _process_block(self, block: BasicBlockNode) -> bool:
        if block.fix_addr_size:
            return False
        before = list(block.insts)
        self._merge_free(block)
        return before != block.insts

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
                    result.append(TimeInst(c_op="inc_ref", lit=Immediate(pending_lit)))
                    pending_lit = 0
                result.append(inst)
            elif reads_implicit_time_base(inst):
                # s14-reading instruction that we cannot fold into (no literal
                # @T, or @T is a register).  Flush pending TIME inc_ref before
                # the instruction so its emission time stays anchored.
                if pending_lit > 0:
                    result.append(TimeInst(c_op="inc_ref", lit=Immediate(pending_lit)))
                    pending_lit = 0
                result.append(inst)
            else:
                result.append(inst)

        if pending_lit > 0:
            result.append(TimeInst(c_op="inc_ref", lit=Immediate(pending_lit)))

        block.insts = result
