"""LoopConditionMergePass: merge register increments into conditional jump side-writes.

Purpose
-------
The tProc v2 ``JUMP`` instruction supports a ``-wr`` side-write field that
atomically writes a register as part of the jump.  This pass exploits that
feature to collapse a counter-decrement + conditional-jump pair (2 words)
into a single jump with embedded side-write (1 word), saving pmem space and
a pipeline slot.

Example
-------
Before::

    REG_WR r1 op r1 - #1         ; counter decrement
    JUMP loop -if(NZ) -op(r1 - #0)   ; zero-check

After::

    JUMP loop -if(NZ) -wr(r1 op) -op(r1 - #1)   ; merged: decrement + check in one word

QICK Hardware Notes
-------------------
- The ``-wr`` side-write field executes the ALU expression and writes the
  result to the destination register *unconditionally*, even if the branch
  is not taken.  The counter is always decremented, and the branch checks
  whether the decremented value is zero.
- This pass only targets the ``-op(reg - #0)`` zero-comparison form because
  subtracting zero is the conventional way to set the NZ flag without a
  separate TEST, and the merged form uses the *same* decrement expression
  as both the side-write and the implicit flag update.
- A separate TEST instruction (e.g. ``TEST op(r2 - #5)``) followed by
  ``REG_WR + JUMP`` CANNOT be safely absorbed into the JUMP's ``-op``
  field because the tProc v2 JUMP only has one ``-op`` expression, which
  must serve as both the condition test AND the ``-wr`` data source.  When
  the TEST checks a different register or a different expression, the
  combined form would alter the branch condition.  Such patterns are left
  untouched.

Decision Notes
--------------
Only the ``REG_WR dst op (dst +/- #C)`` + ``JUMP -if(COND) -op(dst - #0)``
pattern is merged.  TEST-based patterns are not handled because they require
two independent ALU expressions (one for the condition, one for the
side-write) which cannot coexist in a single JUMP instruction.
"""

from __future__ import annotations

from ...instructions import JumpInst, RegWriteInst
from ...node import BasicBlockNode
from ...operands import AluExpr, AluOp, Immediate, SideWrite, SrcKeyword
from ..base import BlockChunkPass


def _make_merged_branch(branch: JumpInst, inst: RegWriteInst) -> JumpInst:
    return JumpInst(
        label=branch.label,
        if_cond=branch.if_cond,
        addr=branch.addr,
        wr=SideWrite(inst.dst, "op"),
        op=inst.op,
        uf=branch.uf,
    )


class LoopConditionMergePass(BlockChunkPass):
    """Chunk pass to merge register increments and conditional jumps.

    Before:
        REG_WR r1 op r1 - #1
        JUMP label -if(NZ) -op(r1 - #0)
    After:
        JUMP label -if(NZ) -wr(r1 op) -op(r1 - #1)

    Only the REG_WR + JUMP zero-comparison form is merged.  TEST-based
    patterns are not handled (see module docstring for rationale).
    """

    def _process_block(self, block: BasicBlockNode) -> bool:
        if block.disable_opt:
            return False
        if block.branch is None or not block.insts:
            return False

        # Merge counter decrement/increment into conditional jump.
        return self._merge_zero_comparison(block)

    def _merge_zero_comparison(self, block: BasicBlockNode) -> bool:
        """Pattern 1: REG_WR r1 op r1-1 + JUMP op(r1-0) -> JUMP wr(r1) op(r1-1)."""
        if not block.insts:
            return False

        last_idx = len(block.insts) - 1
        last_inst = block.insts[last_idx]
        branch = block.branch

        if (
            isinstance(last_inst, RegWriteInst)
            and last_inst.src == SrcKeyword.OP
            and last_inst.op is not None
            and last_inst.lit is None
            and last_inst.if_cond is None
            and not last_inst.uf
            and last_inst.wr is None
            and last_inst.label is None
            and last_inst.addr is None
            and branch is not None
            and branch.if_cond is not None
            and branch.op is not None
            and branch.wr is None
        ):
            # Target pattern: branch.op is "reg - #0" and last_inst.dst is "reg"
            op = branch.op
            if (
                isinstance(op, AluExpr)
                and op.lhs == last_inst.dst
                and op.op == AluOp.SUB
                and op.rhs == Immediate(0)
            ):
                block.branch = _make_merged_branch(branch, last_inst)
                block.insts.pop(last_idx)
                return True
        return False

