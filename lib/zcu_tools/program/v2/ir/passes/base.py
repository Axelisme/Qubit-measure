from __future__ import annotations

from typing import cast

from ..instructions import (
    DmemReadInst,
    GenericInst,
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
from ..node import BlockNode, InstNode, IRLoop, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer, walk_instructions


class OptimizationPassBase(AbsPipeLinePass, IRTransformer):
    """Base class for optimization passes with recursive IR traversal."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        self.ctx = ctx
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        return cast(RootNode, res or ir)

    def _bump_stat(self, key: str, delta: int = 1) -> None:
        stats = self.ctx.pass_stats
        stats[key] = stats.get(key, 0) + delta


def is_label_or_branching_inst(inst: Instruction) -> bool:
    return isinstance(inst, (LabelInst, JumpInst, TestInst, MetaInst))


def is_safe_linear_inst(inst: Instruction) -> bool:
    if isinstance(inst, (TimeInst, WaitInst, RegWriteInst, DmemReadInst, NopInst)):
        return True
    if isinstance(inst, GenericInst) and inst.cmd == "NOP":
        return True
    return False


def block_contains_structural_node(block: BlockNode) -> bool:
    for item in block.insts:
        if isinstance(item, IRNode) and not isinstance(item, InstNode):
            return True
        if isinstance(item, InstNode) and is_label_or_branching_inst(item.inst):
            return True
    return False


def loop_is_label_sensitive(loop: IRLoop) -> bool:
    if block_contains_structural_node(loop.body):
        return True

    for inst in walk_instructions(loop.body):
        if isinstance(inst, LabelInst):
            return True
        if isinstance(inst, (JumpInst, TestInst, MetaInst)):
            return True
        if inst.need_label is not None:
            return True
        if loop.counter_reg in inst.reg_read:
            return True
        if loop.counter_reg in inst.reg_write:
            return True

    return False
