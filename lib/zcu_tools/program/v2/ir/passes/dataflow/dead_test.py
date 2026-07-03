"""DeadTestEliminationPass: remove TestInst whose flag is never consumed.

Purpose
-------
A ``TEST`` instruction evaluates an ALU expression and sets the condition
flags, but has no other side effects.  If no conditional instruction reads
those flags before the next flag overwrite or before the block exits, the
instruction is dead and can be removed.

Example
-------
Before::

    TEST op(r0 - #5)      ; dead: flag never consumed
    TEST op(r1 - #3)      ; flag consumed by conditional write below
    REG_WR r2 imm #1 -if(NZ)

After::

    TEST op(r1 - #3)
    REG_WR r2 imm #1 -if(NZ)

QICK Hardware Notes
-------------------
- ``TestInst`` has no hardware side effects beyond setting the condition flags.
  It is therefore safe to remove any TEST whose flags are never read.
- Any instruction with ``if_cond is not None`` consumes the current flags.
  If the same instruction also carries ``-uf``, the condition reads the old
  flag first and the ALU update overwrites flags for later instructions.
- A ``-uf`` instruction without ``if_cond`` overwrites the ALU flags without
  consuming them, so a preceding pending TEST becomes dead.
- Opaque control boundaries such as CALL/RET clear local pending state without
  marking the TEST dead because the callee/return boundary may observe flags.
- The scan is conservative at fall-through boundaries: if a block has no
  terminal branch, the next block may consume the flags, so a pending TEST is
  kept. A pending TEST is only marked dead at block exit when an explicit
  unconditional branch proves there is no fall-through consumer.

Decision Notes
--------------
The pass is local to a single BasicBlockNode (no cross-block flag liveness).
Cross-block analysis would be more precise but adds complexity; in practice,
dead TEST instructions arise from dead-code patterns within a single block
after other passes run.
"""

from __future__ import annotations

from ...instructions import BaseInst, JumpInst, TestInst
from ...node import BasicBlockNode
from ..base import DATAFLOW_TRANSPARENT_INSTS, BlockChunkPass


class DeadTestEliminationPass(BlockChunkPass):
    """Remove dead TestInst from free BasicBlockNode chunks."""

    def _process_block(self, block: BasicBlockNode) -> bool:
        if block.disable_opt:
            return False
        dead = self._find_dead_indices(block.insts, block.branch)
        if not dead:
            return False
        block.insts = [inst for i, inst in enumerate(block.insts) if i not in dead]
        return True

    def _find_dead_indices(
        self, insts: list[BaseInst], branch: JumpInst | None
    ) -> set[int]:
        pending: int | None = None  # index of the last TestInst not yet consumed
        dead: set[int] = set()

        for idx, inst in enumerate(insts):
            if isinstance(inst, TestInst):
                if pending is not None:
                    dead.add(pending)  # previous TEST was never consumed
                pending = idx
            elif getattr(inst, "if_cond", None) is not None:
                pending = None  # flag consumed by any conditional instruction
            elif getattr(inst, "uf", False):
                # A -uf instruction (e.g. REG_WR -uf) overwrites the ALU flags
                # as a side effect. Any pending TEST is now dead — its flags can
                # never be observed. The -uf instruction itself is kept: it has
                # its own register-write side effect and is not a TestInst.
                if pending is not None:
                    dead.add(pending)
                pending = None
            elif not isinstance(inst, DATAFLOW_TRANSPARENT_INSTS):
                pending = None  # opaque boundary may observe flags outside this block

        # After all insts, check the block's branch.
        if pending is not None:
            if branch is not None and branch.if_cond is None:
                dead.add(pending)  # flag never consumed before block exit

        return dead
