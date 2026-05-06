from __future__ import annotations

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
from ..node import BlockNode, InstNode, RootNode

# Re-export for callers that import LinearPassAdapter from here.
from ..pipeline import (
    AbsIRPass,
    LinearPassAdapter,  # noqa: F401
    PipeLineContext,
)
from ..traversal import IRTransformer


class OptimizationPassBase(AbsIRPass, IRTransformer):
    """Base class for IR-level optimization passes with recursive IR traversal."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        self.ctx = ctx
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        return cast(RootNode, res or ir)


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
