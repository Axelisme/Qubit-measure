"""BranchEliminationPass: remove redundant unconditional branches to the next block.

Purpose
-------
When Block A's branch jumps unconditionally to Block B, and Block B physically
follows Block A in the flat block list, the branch is redundant — the CPU
would fall through to Block B anyway.  Removing it shrinks pmem and removes a
pipeline flush that an unconditional jump would otherwise cause.

Example
-------
Before::

    block_a:
      REG_WR r0 imm #1
      JUMP block_b          ; redundant: block_b is the very next block

    block_b:
      TIME inc_ref #50

After::

    block_a:
      REG_WR r0 imm #1
                            ; branch removed

    block_b:
      TIME inc_ref #50

QICK Hardware Notes
-------------------
- Only plain unconditional label jumps (``if_cond=None``, ``op=None``,
  ``wr=None``) can be eliminated.  Conditional jumps must be kept because
  they test flags.  Jumps with ``-wr`` perform a register side-write as a
  hardware side effect and cannot be removed even if they appear unconditional.
- Register-address jumps (``addr`` is a Register rather than a Label) cannot
  be statically resolved to the next block, so they are also left untouched.
- Blocks with ``disable_opt=True`` (dispatch-table stubs) must not be
  modified — their branch is part of the fixed-width encoding.
"""

from __future__ import annotations

from ...labels import Label
from ...node import BasicBlockNode
from ...pipeline import AbsChunkListPass, ChunkList, PipeLineContext


class BranchEliminationPass(AbsChunkListPass):
    """Remove redundant unconditional branches to the next block.

    A branch from Block A to Block B is redundant when Block B immediately
    follows Block A in the flat chunk list.

    Only unconditional jumps (if_cond is None, op is None) that target a
    plain Label (not a register address) are considered for elimination.
    """

    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]:  # noqa: ARG002
        changed = False
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, BasicBlockNode):
                changed |= self._try_eliminate_branch(chunk, chunks, i)
        return chunks, changed

    def _try_eliminate_branch(
        self, block: BasicBlockNode, chunks: ChunkList, idx: int
    ) -> bool:
        if block.disable_opt:
            return False
        branch = block.branch
        if branch is None:
            return False
        # Only eliminate plain unconditional label jumps with no side effects.
        if branch.if_cond is not None or branch.op is not None or branch.wr is not None:
            return False
        if not isinstance(branch.label, Label):
            return False

        # Find the next BasicBlockNode in the flat chunk list.
        next_block = next(
            (item for item in chunks[idx + 1 :] if isinstance(item, BasicBlockNode)),
            None,
        )
        if next_block is None:
            return False

        # Check if the branch targets the immediately following block.
        target = branch.label
        if not any(lbl.name == target for lbl in next_block.labels):
            return False

        block.branch = None
        return True
