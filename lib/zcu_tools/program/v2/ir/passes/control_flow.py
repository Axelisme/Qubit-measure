from __future__ import annotations

from typing import Optional, cast

from ..instructions import BaseInst, Instruction, JumpInst, LabelInst, MetaInst
from ..labels import PSEUDO_LABELS, Label
from ..node import BasicBlockNode, BlockNode, IRBranch, IRLoop, IRNode, RootNode
from ..pipeline import AbsFlatPass, PipeLineContext
from .base import OptimizationPassBase, walk_basic_blocks, walk_instructions


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
        if str(inst.name) in PSEUDO_LABELS:
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
            if str(lbl.name) in PSEUDO_LABELS
            or not lbl.can_remove
            or lbl.name in self._referenced_labels
        ]
        return node


class BranchEliminationPass(OptimizationPassBase):
    """Remove redundant unconditional branches to the next block.

    A branch from Block A to Block B is redundant when Block B immediately
    follows Block A in the flat block list.

    Only unconditional jumps (if_cond is None, op is None) that target a
    plain Label (not a register address) are considered for elimination.
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> tuple[RootNode, bool]:
        self.ctx = ctx
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
        if block.fix_addr_size:
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


class BlockMergePass(OptimizationPassBase):
    """Merge adjacent BasicBlockNodes when safe.

    Block A and Block B can be merged when:
      - Block A has no branch (falls through).
      - Block B has no alive labels (not a jump target).
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> tuple[RootNode, bool]:
        self.ctx = ctx
        referenced = _collect_referenced_labels(ir)
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
                and not a.fix_addr_size
                and not b.fix_addr_size
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


class UnreachableEliminationPass(AbsFlatPass):
    """Remove unreachable instructions after unconditional jumps.

    Keep structural metadata (`MetaInst`) even in dead regions.
    """

    def process(
        self, insts: list[Instruction], ctx: PipeLineContext
    ) -> tuple[list[Instruction], bool]:
        _ = ctx
        final_insts: list[Instruction] = []
        dead_mode = False
        changed = False

        for inst in insts:
            if dead_mode:
                if isinstance(inst, LabelInst):
                    dead_mode = False
                elif isinstance(inst, MetaInst):
                    final_insts.append(inst)
                    continue
                else:
                    changed = True
                    continue

            final_insts.append(inst)

            if isinstance(inst, JumpInst) and inst.if_cond is None:
                dead_mode = True

        return final_insts, changed
