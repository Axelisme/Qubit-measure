from __future__ import annotations

from typing import Optional, cast

from ..instructions import Instruction, LabelInst, NopInst
from ..labels import PSEUDO_LABELS, Label
from ..node import BasicBlockNode, BlockNode, IRNode, RootNode
from ..pipeline import LinearPipeline, PipeLineContext
from ..traversal import walk_instructions
from .base import OptimizationPassBase


def _collect_referenced_labels(ir: RootNode) -> set[Label]:
    return {
        label
        for inst in walk_instructions(ir)
        if (label := inst.need_label) is not None
    }


class DeadLabelEliminationPass(OptimizationPassBase):
    """Remove labels that are never referenced by any instruction."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        self.ctx = ctx
        if not self.ctx.config.enable_dead_label:
            return ir

        self._referenced_labels = _collect_referenced_labels(ir)
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        return cast(RootNode, res or ir)

    def visit_LabelInst(self, inst: LabelInst) -> Optional[Instruction]:
        if str(inst.name) in PSEUDO_LABELS:
            return inst
        if inst.name not in self._referenced_labels:
            return None
        return inst

    def visit_BasicBlockNode(self, node: BasicBlockNode) -> IRNode:
        node.labels = [
            lbl for lbl in node.labels
            if str(lbl.name) in PSEUDO_LABELS or lbl.name in self._referenced_labels
        ]
        return node


class BranchEliminationPass(OptimizationPassBase):
    """Remove or NOP-pad redundant unconditional branches to the next block.

    A branch from Block A to Block B is redundant when Block B immediately
    follows Block A in the flat block list.

    - fix_inst_num=False: remove the branch entirely (shrinks the block).
    - fix_inst_num=True : replace the branch with a NopInst to preserve stride.

    Only unconditional jumps (if_cond is None, op is None) that target a
    plain Label (not a register address) are considered for elimination.
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        self.ctx = ctx
        self._process_block(ir)
        return ir

    def _process_block(self, node: IRNode) -> None:
        if not isinstance(node, BlockNode):
            return
        blocks = node.insts
        for i, item in enumerate(blocks):
            if isinstance(item, BasicBlockNode):
                self._try_eliminate_branch(item, blocks, i)
            elif isinstance(item, BlockNode):
                self._process_block(item)

    def _try_eliminate_branch(
        self, block: BasicBlockNode, siblings: list[IRNode], idx: int
    ) -> None:
        branch = block.branch
        if branch is None:
            return
        # Only eliminate plain unconditional label jumps.
        if branch.if_cond is not None or branch.op is not None:
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

        if block.fix_inst_num:
            # Preserve instruction count: replace branch with NOP.
            block.insts.append(NopInst())
            block.branch = None
        else:
            block.branch = None


def _next_basic_block(
    siblings: list[IRNode], from_idx: int
) -> Optional[BasicBlockNode]:
    """Return the first BasicBlockNode after from_idx in the sibling list."""
    for item in siblings[from_idx + 1:]:
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


class BlockMergePass(OptimizationPassBase):
    """Merge adjacent BasicBlockNodes when safe.

    Block A and Block B can be merged when:
      - Block A has no branch (falls through).
      - Block B has no alive labels (not a jump target).

    After merging, re-runs ``post_linear`` across merged boundaries to
    eliminate dead writes exposed by the merge.
    """

    def __init__(self, post_linear: Optional[LinearPipeline] = None) -> None:
        self._post_linear = post_linear

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        self.ctx = ctx
        referenced = _collect_referenced_labels(ir)
        self._merge_block(ir, referenced)
        if self._post_linear is not None:
            self._post_linear.process(ir)
        return ir

    def _merge_block(self, node: IRNode, referenced: set[Label]) -> None:
        if not isinstance(node, BlockNode):
            return
        changed = True
        while changed:
            changed = self._merge_pass(node.insts, referenced)
        # Recurse into remaining structural nodes.
        for child in node.insts:
            if isinstance(child, BlockNode):
                self._merge_block(child, referenced)

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
                and not a.fix_inst_num
                and not b.fix_inst_num
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
        lbl.name in referenced and str(lbl.name) not in PSEUDO_LABELS
        for lbl in block.labels
    )
