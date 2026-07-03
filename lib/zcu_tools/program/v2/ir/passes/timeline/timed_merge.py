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
- ``WAIT time @T`` is NOT a fold target even though it carries a ``TimeOffset``:
  its ``@T`` is an absolute user-time comparison value (the assembler expands it
  to ``TEST s11 - #(T-10)``), not an offset from the s14 reference.  Folding a
  ``TIME inc_ref`` delta into it would change the wait target time.  ``WaitInst``
  is therefore treated as a flush barrier.
- Mostly-transparent instructions: ``PortWriteInst``, ``DportWriteInst``,
  ``DmemReadInst``, ``DmemWriteInst``, ``WmemWriteInst``, and ``RegWriteInst``
  whose ``dst`` is not a system register (``sN``).  These pass through
  *without* flushing *unless* they carry an explicit ``@T`` (``TimeOffset``)
  field or read/write the time-base register (``TIMED_BASE_REG = s14``).
  Specifically:

  - With ``@T``: the ``TimeOffset`` absorbs ``pending_lit`` (``@T += delta``);
    pending is *not* cleared so the hardware clock still advances at block end.
    This compensation applies only when the instruction does not also write the
    time-base register through ``-wr``.
  - Without ``@T``, reading or writing ``TIMED_BASE_REG``: pending is flushed
    first because the instruction would observe or mutate the reference value.
  - Without ``@T``, not touching ``TIMED_BASE_REG``: truly transparent;
    pending is unchanged.

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
    WaitInst,
    WmemWriteInst,
)
from ...node import BasicBlockNode
from ...operands import Immediate, Operand, TimeOffset
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


def _has_uncompensated_timed_base_access(inst: BaseInst) -> bool:
    """Return true when an anchored timed instruction touches s14 beyond scheduling."""
    if TIMED_BASE_REG in inst.reg_write:
        return True
    for field in dataclasses.fields(inst):
        if field.name == "time":
            continue
        value = getattr(inst, field.name)
        if isinstance(value, Operand) and TIMED_BASE_REG in value.regs():
            return True
    return False


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
            elif not isinstance(inst, WaitInst) and isinstance(
                getattr(inst, "time", None), TimeOffset
            ):
                # WAIT time @T is excluded: its @T is an absolute user-time
                # comparison value (TEST s11 - #(T-10)), not an offset from the
                # s14 reference, so absorbing a TIME inc_ref delta would change
                # the wait target. WaitInst falls through to the else-flush.
                if _has_uncompensated_timed_base_access(inst):
                    pending_lit = _flush(result, pending_lit)
                    result.append(inst)
                    continue

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
            elif not isinstance(inst, TimeInst) and (
                TIMED_BASE_REG in inst.reg_read or TIMED_BASE_REG in inst.reg_write
            ):
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
                result.append(inst)
            elif isinstance(inst, RegWriteInst) and not inst.dst.is_volatile_reg():
                result.append(inst)
            else:
                pending_lit = _flush(result, pending_lit)
                result.append(inst)

        _flush(result, pending_lit)

        block.insts = result
