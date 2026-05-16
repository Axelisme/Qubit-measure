"""DeadTestEliminationPass: remove TestInst whose flag is never consumed.

Purpose
-------
A ``TEST`` instruction evaluates an ALU expression and sets the condition
flags, but has no other side effects.  If no conditional jump reads those
flags before the next ``TEST`` or before the block exits, the instruction is
dead and can be removed.

Example
-------
Before::

    TEST op(r0 - #5)      ; dead: flag never consumed
    TEST op(r1 - #3)      ; flag consumed by JUMP below
    JUMP loop -if(NZ)

After::

    TEST op(r1 - #3)
    JUMP loop -if(NZ)

QICK Hardware Notes
-------------------
- ``TestInst`` has no hardware side effects beyond setting the condition flags.
  It is therefore safe to remove any TEST whose flags are never read.
- Only a ``JumpInst`` with ``if_cond is not None`` consumes the flags.  A
  ``RegWriteInst`` with ``-uf`` *also* updates flags but it does so as a
  *write*, not a read — so it does NOT consume a preceding TEST's flag.
- The scan is conservative in one direction: if a block exits (via its branch
  field or by falling off the end) without a conditional jump, the pending
  TEST is marked dead.

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
from ..base import BlockChunkPass


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
            elif getattr(inst, "uf", False):
                # A -uf instruction (e.g. REG_WR -uf) overwrites the ALU flags
                # as a side effect. Any pending TEST is now dead — its flags can
                # never be observed. The -uf instruction itself is kept: it has
                # its own register-write side effect and is not a TestInst.
                if pending is not None:
                    dead.add(pending)
                pending = None
            elif isinstance(inst, JumpInst) and inst.if_cond is not None:
                pending = None  # flag consumed by conditional jump

        # After all insts, check the block's branch.
        if pending is not None:
            if branch is None or branch.if_cond is None:
                dead.add(pending)  # flag never consumed before block exit

        return dead
