"""ZeroDelayDCEPass: remove TIME inc_ref #0 instructions.

Purpose
-------
A ``TIME inc_ref #0`` advances the time reference by zero ticks — it is a
pure no-op that takes a pmem word without doing anything useful.  These
appear after constant-folding passes reduce a delay to zero.  Removing them
shrinks the instruction stream.

Example
-------
Before::

    TIME inc_ref #50
    TIME inc_ref #0     ; no-op
    PORT_WR 2 ...

After::

    TIME inc_ref #50
    PORT_WR 2 ...

QICK Hardware Notes
-------------------
- ``TIME inc_ref #0`` is semantically a no-op: ``s14 += 0`` leaves the time
  reference unchanged.  No downstream instruction observes any difference.
- Blocks with ``fix_addr_size=True`` (dispatch-table stubs) are skipped
  because their pmem word count is fixed by the jump-table encoding.

Decision Notes
--------------
This pass is intentionally minimal — it only removes the literal zero case.
Non-zero increments are handled by TimedMergePass which can also fold them
into anchored ``@N`` timestamps.
"""

from __future__ import annotations

from ...instructions import BaseInst, TimeInst
from ...node import BasicBlockNode
from ...operands import Immediate
from ...pipeline import AbsChunkPass, ChunkList, PipeLineContext


def _is_zero_ref_increment(inst: BaseInst) -> bool:
    if not isinstance(inst, TimeInst):
        return False
    if inst.c_op != "inc_ref":
        return False
    if inst.r1 is not None:
        return False
    return inst.lit == Immediate(0)


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
