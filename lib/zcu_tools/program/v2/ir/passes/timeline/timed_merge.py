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
- ``@N`` (``TimeOffset``) is an *anchored* absolute offset from the current
  reference.  A pending ``TIME inc_ref #P`` can be absorbed by replacing
  ``@N`` with ``@(N + P)`` on the instruction.
- Register-driven ``TIME inc_ref rX`` cannot be folded into ``@N`` because the
  increment amount is unknown at compile time.  Pending is flushed before it.
- ``TIME set_ref`` / ``updt`` / ``rst`` write s14 to an absolute value,
  invalidating any accumulated pending delta.  Pending is flushed before them.
- Transparent instructions (pending is NOT flushed): ``PortWriteInst``,
  ``DportWriteInst``, ``DmemReadInst``, ``DmemWriteInst``, ``WmemWriteInst``,
  and ``RegWriteInst`` whose ``dst`` is not a system register (``sN``).
  Any other instruction flushes pending before it.

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

from ...hw_semantics import TIMED_BASE_REG
from ...instructions import (
    BaseInst,
    DmemReadInst,
    DmemWriteInst,
    DportWriteInst,
    PortWriteInst,
    RegWriteInst,
    TimeInst,
    WmemWriteInst,
)
from ...node import BasicBlockNode
from ...operands import Immediate, TimeOffset
from ..base import BlockChunkPass

# conservative safe limit for TIME inc_ref and @T fields
TIMED_LIT_MAX: int = (1 << 20) - 1


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


def _flush(result: list[BaseInst], pending_lit: int) -> int:
    if pending_lit > 0:
        result.append(TimeInst(c_op="inc_ref", lit=Immediate(pending_lit)))
    return 0


class TimedMergePass(BlockChunkPass):
    """Aggressive TIME inc_ref optimisation pass."""

    def _process_block(self, block: BasicBlockNode) -> bool:
        if block.disable_opt:
            return False
        before = list(block.insts)
        self._merge_free(block)
        return before != block.insts

    def _merge_free(self, block: BasicBlockNode) -> None:
        pending_lit: int = 0
        result: list[BaseInst] = []

        for inst in block.insts:
            if _is_lit_time(inst):
                assert isinstance(inst, TimeInst) and isinstance(inst.lit, Immediate)
                delta = inst.lit.value
                if pending_lit + delta > TIMED_LIT_MAX:
                    pending_lit = _flush(result, pending_lit)
                    if delta > TIMED_LIT_MAX:
                        result.append(TimeInst(c_op="inc_ref", lit=Immediate(delta)))
                    else:
                        pending_lit = delta
                else:
                    pending_lit += delta
            elif isinstance(getattr(inst, "time", None), TimeOffset):
                time = getattr(inst, "time")
                assert isinstance(time, TimeOffset)
                if pending_lit > 0:
                    if time.value + pending_lit > TIMED_LIT_MAX:
                        pending_lit = _flush(result, pending_lit)
                        result.append(inst)
                    else:
                        result.append(
                            dataclasses.replace(
                                inst, time=TimeOffset(time.value + pending_lit)
                            )
                        )
                        # pending_lit is NOT reset: subsequent timed insts in
                        # the same baseline segment receive the same delta, and
                        # the TIME must still be emitted at end of block so the
                        # hardware reference clock actually advances.
                else:
                    result.append(inst)
            elif (
                isinstance(inst, TimeInst)
                and inst.c_op == "inc_ref"
                and inst.r1 is not None
            ):
                # Register-driven TIME inc_ref: unknown delta, must flush.
                pending_lit = _flush(result, pending_lit)
                result.append(inst)
            elif isinstance(inst, TimeInst) and inst.c_op in ("set_ref", "updt", "rst"):
                # TIME set_ref/updt/rst sets s14 to an absolute value —
                # accumulated pending delta is invalidated.
                pending_lit = _flush(result, pending_lit)
                result.append(inst)
            elif isinstance(
                inst,
                (
                    PortWriteInst,
                    DportWriteInst,
                    DmemReadInst,
                    DmemWriteInst,
                    WmemWriteInst,
                ),
            ):
                if TIMED_BASE_REG in inst.reg_read or TIMED_BASE_REG in inst.reg_write:
                    pending_lit = _flush(result, pending_lit)
                result.append(inst)
            elif isinstance(inst, RegWriteInst) and not inst.dst.is_volatile_reg():
                result.append(inst)
            else:
                pending_lit = _flush(result, pending_lit)
                result.append(inst)

        _flush(result, pending_lit)

        block.insts = result
