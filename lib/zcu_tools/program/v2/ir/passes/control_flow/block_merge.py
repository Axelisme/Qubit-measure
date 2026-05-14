"""BlockMergePass: merge adjacent BasicBlockNodes when safe.

Purpose
-------
Adjacent BasicBlockNodes with no labels on the second block and no branch on
the first can be merged into a single block.  Fewer blocks means fewer
boundary constraints for downstream passes (e.g., IncRegMergePass can sink
increments across what used to be block boundaries).

Example
-------
Before::

    block_a:               ; no branch (falls through)
      REG_WR r0 imm #1

    block_b:               ; no labels → not a jump target
      TIME inc_ref #50

After::

    block_a:
      REG_WR r0 imm #1
      TIME inc_ref #50

QICK Hardware Notes
-------------------
- A block with ``disable_opt=True`` is a dispatch-table stub whose pmem
  word count is fixed by the jump-table encoding.  Such blocks must not be
  merged with neighbours even if the structural conditions are met.
- A block that has any alive label (referenced by a jump, not pseudo) is a
  potential branch target; merging it would shift its start address and break
  any jump pointing there.

Decision Notes
--------------
Merging is done with an inner fixed-point loop inside each container so that
a chain ``A → B → C → D`` (all mergeable) collapses to one block in a single
pass invocation rather than requiring O(n) pipeline iterations.
"""

from __future__ import annotations

from ...instructions import BaseInst
from ...labels import Label
from ...node import BasicBlockNode, BlockNode, IRBranch, IRLoop, IRNode, RootNode
from ...pipeline import AbsIRPass, PipeLineContext


def _collect_referenced_labels_tree(node: IRNode) -> set[Label]:
    """Collect all labels referenced by any instruction in the IR tree."""
    refs: set[Label] = set()
    _collect_from_node(node, refs)
    return refs


def _collect_from_node(node: IRNode, refs: set[Label]) -> None:
    if isinstance(node, BasicBlockNode):
        for inst in (*node.labels, *node.insts, *([node.branch] if node.branch else [])):
            if isinstance(inst, BaseInst) and inst.need_label is not None:
                refs.add(inst.need_label)
    elif isinstance(node, BlockNode):
        for child in node.insts:
            _collect_from_node(child, refs)
    elif isinstance(node, IRLoop):
        _collect_from_node(node.body, refs)
    elif isinstance(node, IRBranch):
        for case in node.cases:
            _collect_from_node(case, refs)


class BlockMergePass(AbsIRPass):
    """Merge adjacent BasicBlockNodes when safe.

    Block A and Block B can be merged when:
      - Block A has no branch (falls through).
      - Block B has no alive labels (not a jump target).
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> tuple[RootNode, bool]:
        referenced = _collect_referenced_labels_tree(ir)
        changed = self._merge_block(ir, referenced)
        return ir, changed

    def _merge_block(self, node: IRNode, referenced: set[Label]) -> bool:
        changed_any = False
        if isinstance(node, BlockNode):
            changed = True
            while changed:
                changed = self._merge_pass(node.insts, referenced)
                changed_any |= changed
            for child in node.insts:
                changed_any |= self._merge_block(child, referenced)
        elif isinstance(node, IRLoop):
            changed_any |= self._merge_block(node.body, referenced)
        elif isinstance(node, IRBranch):
            for case in node.cases:
                changed_any |= self._merge_block(case, referenced)
        return changed_any

    def _merge_pass(self, items: list[IRNode], referenced: set[Label]) -> bool:
        i = 0
        changed = False
        while i < len(items) - 1:
            a = items[i]
            b = items[i + 1]
            if (
                isinstance(a, BasicBlockNode)
                and isinstance(b, BasicBlockNode)
                and a.branch is None
                and not _has_alive_labels(b, referenced)
                and not a.disable_opt
                and not b.disable_opt
            ):
                # Merge b into a.
                a.insts.extend(b.insts)
                a.branch = b.branch
                del items[i + 1]
                changed = True
            else:
                i += 1
        return changed


def _has_alive_labels(block: BasicBlockNode, referenced: set[Label]) -> bool:
    return any(
        lbl.name in referenced and not lbl.name.is_pseudo_name()
        for lbl in block.labels
    )
