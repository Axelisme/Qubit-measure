"""UnreachableEliminationPass: remove dead BasicBlockNodes after unconditional jumps.

Purpose
-------
An unconditional jump (no ``if_cond``) transfers control away permanently.
Any BasicBlockNode that immediately follows such a jump and has no labels
(i.e., no other instruction jumps to it) is dead code and can be safely
removed.

Example
-------
Before::

    block_a:
      JUMP exit             ; unconditional

    block_dead:             ; no labels → unreachable
      REG_WR r0 imm #99

    exit:
      ...

After::

    block_a:
      JUMP exit

    exit:
      ...

QICK Hardware Notes
-------------------
- MetaInst nodes are structural pipeline markers (e.g., loop / branch
  boundaries at the chunk layer).  They must be preserved even inside a dead
  region because the pipeline infrastructure relies on them for structural
  bookkeeping.
- A block with labels is always potentially reachable (some dynamic branch may
  target it), so dead-mode ends as soon as a labelled block is encountered.

Decision Notes
--------------
This pass operates on flat ChunkList (chunk layer), not on the tree IR.
It uses a single linear scan with a ``dead_mode`` flag rather than graph
reachability, which is sufficient because QICK IR has no computed gotos
other than dispatch-table islands (which are labelled).
"""

from __future__ import annotations

from ...instructions import MetaInst
from ...node import BasicBlockNode
from ...pipeline import AbsChunkListPass, ChunkList, PipeLineContext


class UnreachableEliminationPass(AbsChunkListPass):
    """Remove unreachable BasicBlockNodes after unconditional jumps.

    Keep MetaInst even in dead regions (structural markers must not be dropped).
    A block is unreachable when the preceding block ends with an unconditional
    branch and the block itself has no labels (i.e., not a jump target).
    """

    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]:
        _ = ctx
        result: ChunkList = []
        dead_mode = False
        changed = False

        for chunk in chunks:
            if dead_mode:
                if isinstance(chunk, MetaInst):
                    result.append(chunk)
                elif isinstance(chunk, BasicBlockNode) and chunk.labels:
                    dead_mode = False
                    result.append(chunk)
                else:
                    changed = True
                    continue
            else:
                result.append(chunk)
                if (
                    isinstance(chunk, BasicBlockNode)
                    and chunk.branch is not None
                    and chunk.branch.if_cond is None
                ):
                    dead_mode = True

        return result, changed
