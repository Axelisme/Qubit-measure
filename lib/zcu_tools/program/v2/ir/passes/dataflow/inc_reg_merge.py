"""IncRegMergePass: merge adjacent constant register increments.

Purpose
-------
Loop bodies often contain multiple increments to the same counter register
(e.g., one per unrolled copy).  This pass sinks constant increments forward
past instructions that do not read or write the register, accumulates them,
and emits a single merged increment.  Zero-valued accumulated increments are
dropped entirely.

Example
-------
Before::

    REG_WR r1 op r1 + #1
    TIME inc_ref #50
    REG_WR r1 op r1 + #1    ; both increments are to r1, no r1 read in between
    PORT_WR ...

After::

    TIME inc_ref #50
    REG_WR r1 op r1 + #2    ; merged and sunk to latest safe position
    PORT_WR ...

QICK Hardware Notes
-------------------
- Only general-purpose registers (``r0``, ``r1``, …) are eligible for this
  optimisation.  System registers (``sN``) and wave registers (``wN`` /
  ``r_wave``) are excluded because they typically represent hardware state or
  pulse parameters whose write ordering matters to the firmware.
- ``TimeInst``, ``WaitInst``, ``PortWriteInst``, ``WmemWriteInst``,
  ``DmemReadInst``, ``DmemWriteInst``, ``NopInst``, and ``RegWriteInst`` are
  transparent to increment motion (the pass can sink past them).  Any other
  instruction type (JumpInst, TestInst, LabelInst, …) is treated as a barrier
  and flushes all pending increments before it.
- Tracking uses canonical register names so that ``w_freq → w0`` aliased
  reads flush the correct pending entry.

Decision Notes
--------------
Increments are *sunk* (moved later), not hoisted.  Sinking is safe because
it delays the register update, whereas hoisting could expose the updated
value to instructions that are supposed to see the old value.  The
accumulated value is flushed immediately when any instruction reads or writes
the register, or at end of block.
"""

from __future__ import annotations

from ...instructions import BaseInst, RegWriteInst
from ...node import BasicBlockNode
from ...operands import AluExpr, AluOp, Immediate, Register, SrcKeyword
from ..base import DATAFLOW_TRANSPARENT_INSTS, BlockChunkPass

# REG_WR rd op (rs +/- #N) encodes the immediate in a 24-bit signed field.
# Use a conservative safe limit well within that range (same policy as
# TIMED_LIT_MAX in TimedMergePass).
INC_REG_IMM_MAX: int = (1 << 20) - 1


def _is_const_increment(inst: BaseInst) -> tuple[str, int] | None:
    if not isinstance(inst, RegWriteInst):
        return None
    if inst.src != SrcKeyword.OP or not isinstance(inst.op, AluExpr):
        return None
    if (
        inst.uf
        or inst.if_cond is not None
        or inst.wr is not None
        or inst.label is not None
        or inst.addr is not None
    ):
        return None

    op = inst.op
    if op.op not in (AluOp.ADD, AluOp.SUB):
        return None
    if not isinstance(op.rhs, Immediate):
        return None

    lhs = op.lhs
    if lhs != inst.dst:
        return None

    # Only optimize general-purpose user registers (r0, r1, ...).
    # System registers (sN) and wave registers (wN/r_wave) are excluded for safety,
    # as they often represent hardware state or pulse parameters that should
    # remain at their original program positions.
    if not lhs.is_general_reg():
        return None

    canon = lhs.canonical_name
    val = op.rhs.value
    if op.op == AluOp.SUB:
        val = -val
    # Pending entries are keyed by canonical name so that aliased writes
    # (e.g. w_freq -> w0) are flushed when *any* alias appears in another
    # instruction's reads or writes.
    return canon, val


def _make_increment_inst(reg: str, val: int) -> RegWriteInst:
    if val >= 0:
        op = AluExpr(Register(reg), AluOp.ADD, Immediate(val))
    else:
        op = AluExpr(Register(reg), AluOp.SUB, Immediate(-val))
    return RegWriteInst(dst=Register(reg), src=SrcKeyword.OP, op=op)


class IncRegMergePass(BlockChunkPass):
    """Merge adjacent constant register increments.

    For free blocks (disable_opt=False):
      - Sinks constant increments (REG_WR rX op (rX + #C)) forward past instructions
        that do not read or write rX.
      - Accumulates multiple increments into a single increment.
      - Drops the increment if the accumulated value is 0.
    """

    def _process_block(self, block: BasicBlockNode) -> bool:
        if block.disable_opt:
            return False
        before = list(block.insts)
        self._merge_free(block)
        return before != block.insts

    def _merge_free(self, block: BasicBlockNode) -> None:
        # Pending entries are keyed by canonical register name (see
        # _is_const_increment), so any alias appearing in another
        # instruction's reads/writes can be flushed correctly.
        pending: dict[str, int] = {}
        result: list[BaseInst] = []

        for inst in block.insts:
            if self._is_increment_motion_barrier(inst):
                # Flush ALL pending before unsafe instruction
                for reg, val in pending.items():
                    if val != 0:
                        result.append(_make_increment_inst(reg, val))
                pending.clear()
                result.append(inst)
                continue

            inc_info = _is_const_increment(inst)
            if inc_info is not None:
                reg, val = inc_info
                new_total = pending.get(reg, 0) + val
                if abs(new_total) > INC_REG_IMM_MAX:
                    # Flush the old accumulation before it overflows the
                    # 24-bit signed immediate field, then start fresh.
                    old = pending.pop(reg, 0)
                    if old != 0:
                        result.append(_make_increment_inst(reg, old))
                    if abs(val) > INC_REG_IMM_MAX:
                        # Single step already exceeds the limit; emit as-is.
                        result.append(inst)
                    else:
                        pending[reg] = val
                else:
                    pending[reg] = new_total
            else:
                reads = inst.reg_read
                writes = inst.reg_write

                # Flush pending increments if their register (or any alias of
                # it) is read or written by this instruction.
                for reg in list(pending.keys()):
                    if reg in reads or reg in writes:
                        val = pending.pop(reg)
                        if val != 0:
                            result.append(_make_increment_inst(reg, val))

                result.append(inst)

        # Flush remaining at the end of the block
        for reg, val in pending.items():
            if val != 0:
                result.append(_make_increment_inst(reg, val))

        block.insts = result

    def _is_increment_motion_barrier(self, inst: BaseInst) -> bool:
        return not isinstance(inst, DATAFLOW_TRANSPARENT_INSTS)
