"""TimedMergePass: fold TIME inc_ref increments into anchored timestamps.

Purpose
-------
The tProc v2 hardware supports two ways to schedule port writes: via an
explicit ``TIME inc_ref #N`` that advances the reference clock, and via an
``@T`` anchored timestamp field on instructions that encode an absolute
offset from the current reference.  This pass absorbs pending literal
``TIME inc_ref #N`` delays into downstream ``@T`` fields, eliminating the
separate TIME instruction and reducing pmem usage.

Example
-------
Before::

    TIME inc_ref #100
    PORT_WR 2 @50 ...    ; @50 relative to current ref

After::

    PORT_WR 2 @150 ...   ; @50 + 100 absorbed; no separate TIME needed

QICK Hardware Notes
-------------------
- ``s14`` is the implicit time base register (``TIMED_BASE_REG``).  Any
  instruction that *reads* ``s14`` observes the current accumulated reference.
  Before such an instruction, all pending literal ``TIME inc_ref`` must be
  flushed to ensure the instruction sees the correct reference value.
- ``@N`` (``TimeOffset``) is an *anchored* absolute offset from the current
  reference.  A pending ``TIME inc_ref #P`` can be absorbed by replacing
  ``@N`` with ``@(N + P)`` on the instruction.
- Register-driven ``TIME inc_ref rX`` cannot be folded into ``@N`` because the
  increment amount is unknown at compile time.  A pending literal delay is
  flushed before a register-driven increment.
- Only ``TIME inc_ref`` (``c_op == "inc_ref"``) instructions are handled.
  Other ``TIME`` variants are not affected.

Decision Notes
--------------
Folding is greedy: the pass accumulates all contiguous literal increments
into ``pending_lit`` and applies them at the first opportunity (next anchored
instruction or end of block).  This is correct because ``TIME inc_ref`` is
the only instruction that modifies ``s14`` in the literal path, so the
accumulated delta is always the exact pending advance.
"""

from __future__ import annotations

import dataclasses
from typing import cast

from ...hw_semantics import TIMED_BASE_REG
from ...instructions import BaseInst, TimeInst
from ...node import BasicBlockNode
from ...operands import Immediate, TimeOffset
from ...pipeline import AbsChunkPass, ChunkList, PipeLineContext


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
            elif TIMED_BASE_REG in inst.reg_read:
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
