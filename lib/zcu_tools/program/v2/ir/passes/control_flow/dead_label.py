"""DeadLabelEliminationPass: remove labels that are never referenced.

Purpose
-------
After optimizations remove jumps or restructure control flow, some labels
become orphaned — no instruction refers to them. This pass collects all
labels that appear in ``need_label`` fields across the entire IR, then drops
any LabelInst (and BasicBlockNode.labels entry) that is not in that set.

Example
-------
Before::

    label_orphan:
    REG_WR r0 imm #1
    label_used:
    TIME inc_ref #100

After::

    REG_WR r0 imm #1
    label_used:
    TIME inc_ref #100

QICK Hardware Notes
-------------------
- Pseudo labels (``HERE`` / ``NEXT``) are hardware-reserved jump targets used
  by the QICK assembler to encode short relative offsets.  They must never be
  removed even if no IR instruction references them explicitly.
- Labels with ``can_remove=False`` are anchoring markers (e.g. loop entry
  points, dispatch-table entries) that must survive independently of whether
  any jump currently targets them.

Decision Notes
--------------
The pass does two independent sweeps: one over LabelInst nodes inside
BasicBlockNode.insts (via IRTransformer), and one over BasicBlockNode.labels
(direct field mutation).  Both must be cleared because labels can appear in
either location depending on how the IR was built.
"""

from __future__ import annotations

from typing import Optional, cast

from ...instructions import BaseInst, Instruction, LabelInst
from ...labels import Label
from ...node import BasicBlockNode, IRNode, RootNode
from ...pipeline import PipeLineContext
from ..base import OptimizationPassBase, walk_basic_blocks, walk_instructions


def _collect_referenced_labels(ir: RootNode) -> set[Label]:
    return {
        label
        for inst in walk_instructions(ir)
        if isinstance(inst, BaseInst) and (label := inst.need_label) is not None
    }


class DeadLabelEliminationPass(OptimizationPassBase):
    """Remove labels that are never referenced by any instruction."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> tuple[RootNode, bool]:
        self.ctx = ctx
        before = sum(len(block.labels) for block in walk_basic_blocks(ir))
        self._referenced_labels = _collect_referenced_labels(ir)
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        out = cast(RootNode, res or ir)
        after = sum(len(block.labels) for block in walk_basic_blocks(out))
        return out, before != after

    def visit_LabelInst(self, inst: LabelInst) -> Optional[Instruction]:
        if inst.name.is_pseudo_name():
            return inst
        if not inst.can_remove:
            return inst
        if inst.name not in self._referenced_labels:
            return None
        return inst

    def visit_BasicBlockNode(self, node: BasicBlockNode) -> IRNode:
        node.labels = [
            lbl
            for lbl in node.labels
            if lbl.name.is_pseudo_name()
            or not lbl.can_remove
            or lbl.name in self._referenced_labels
        ]
        return node
