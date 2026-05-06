from __future__ import annotations

from abc import abstractmethod
from typing import cast

from ..instructions import (
    DmemReadInst,
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    RegWriteInst,
    TestInst,
    TimeInst,
    WaitInst,
)
from ..node import BasicBlockNode, BlockNode, InstNode, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class OptimizationPassBase(AbsPipeLinePass, IRTransformer):
    """Base class for optimization passes with recursive IR traversal."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        self.ctx = ctx
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        return cast(RootNode, res or ir)


class AbsLinearPass:
    """Interface for straight-line instruction-list optimisations.

    Operates on a flat list[Instruction] (the `insts` of a BasicBlockNode).
    Must NOT inspect or modify labels or the branch field — those belong to
    the structural level.
    """

    @abstractmethod
    def process_linear(self, insts: list[Instruction]) -> list[Instruction]: ...


class LinearPassAdapter(AbsPipeLinePass):
    """Wraps one or more AbsLinearPass instances into a structural pipeline pass.

    Iterates every BasicBlockNode in the IR tree and applies each linear pass
    to its `insts` list in order.  Blocks with `fix_inst_num=True` are skipped
    to preserve jump-table stride accuracy.

    Legacy InstNode / BlockNode content is left untouched (those paths will be
    cleaned up when all callers migrate to BasicBlockNode).
    """

    def __init__(self, *passes: AbsLinearPass) -> None:
        self.linear_passes = list(passes)

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:  # noqa: ARG002
        self._apply_to_block(ir)
        return ir

    def _apply_to_block(self, node: IRNode) -> None:
        if isinstance(node, BasicBlockNode):
            if not node.fix_inst_num:
                for lp in self.linear_passes:
                    node.insts = lp.process_linear(node.insts)
            return

        if isinstance(node, BlockNode):
            for child in node.insts:
                self._apply_to_block(child)


def is_label_or_branching_inst(inst: Instruction) -> bool:
    return isinstance(inst, (LabelInst, JumpInst, TestInst, MetaInst))


def is_safe_linear_inst(inst: Instruction) -> bool:
    return isinstance(inst, (TimeInst, WaitInst, RegWriteInst, DmemReadInst, NopInst))


def block_contains_structural_node(block: BlockNode) -> bool:
    for item in block.insts:
        if not isinstance(item, InstNode):
            return True
        if is_label_or_branching_inst(item.inst):
            return True
    return False
