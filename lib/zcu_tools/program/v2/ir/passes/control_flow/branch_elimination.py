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

Decision Notes
--------------
The pass uses a recursive structural walk (not flat chunk iteration) so it
can correctly identify the "next" block even across nested BlockNode
containers (e.g., the first block of an IRLoop body).
"""

from __future__ import annotations

from typing import Optional

from ...labels import Label
from ...node import BasicBlockNode, BlockNode, IRBranch, IRLoop, IRNode, RootNode
from ...pipeline import AbsIRPass, PipeLineContext


class BranchEliminationPass(AbsIRPass):
    """Remove redundant unconditional branches to the next block.

    A branch from Block A to Block B is redundant when Block B immediately
    follows Block A in the flat block list.

    Only unconditional jumps (if_cond is None, op is None) that target a
    plain Label (not a register address) are considered for elimination.
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> tuple[RootNode, bool]:
        self._changed = False
        self._process_block(ir)
        return ir, self._changed

    def _process_block(self, node: IRNode) -> None:
        if isinstance(node, BlockNode):
            blocks = node.insts
            for i, item in enumerate(blocks):
                if isinstance(item, BasicBlockNode):
                    self._try_eliminate_branch(item, blocks, i)
                else:
                    self._process_block(item)
        elif isinstance(node, IRLoop):
            self._process_block(node.body)
        elif isinstance(node, IRBranch):
            for case in node.cases:
                self._process_block(case)

    def _try_eliminate_branch(
        self, block: BasicBlockNode, siblings: list[IRNode], idx: int
    ) -> None:
        if block.disable_opt:
            return
        branch = block.branch
        if branch is None:
            return
        # Only eliminate plain unconditional label jumps with no side effects.
        if branch.if_cond is not None or branch.op is not None or branch.wr is not None:
            return
        if not isinstance(branch.label, Label):
            return

        # Find the next BasicBlockNode sibling.
        next_block = _next_basic_block(siblings, idx)
        if next_block is None:
            return

        # Check if the branch targets the immediately following block.
        target = branch.label
        if not any(lbl.name == target for lbl in next_block.labels):
            return

        block.branch = None
        self._changed = True


def _next_basic_block(
    siblings: list[IRNode], from_idx: int
) -> Optional[BasicBlockNode]:
    """Return the first BasicBlockNode after from_idx in the sibling list."""
    for item in siblings[from_idx + 1 :]:
        if isinstance(item, BasicBlockNode):
            return item
        if isinstance(item, BlockNode):
            first = _first_basic_block(item)
            if first is not None:
                return first
    return None


def _first_basic_block(node: IRNode) -> Optional[BasicBlockNode]:
    if isinstance(node, BasicBlockNode):
        return node
    if isinstance(node, BlockNode):
        for child in node.insts:
            result = _first_basic_block(child)
            if result is not None:
                return result
    return None
