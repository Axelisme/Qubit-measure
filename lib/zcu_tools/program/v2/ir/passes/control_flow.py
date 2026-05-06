from __future__ import annotations

from typing import Optional, cast

from ..instructions import Instruction, LabelInst
from ..labels import PSEUDO_LABELS
from ..node import RootNode
from ..pipeline import PipeLineContext
from ..traversal import walk_instructions
from .base import OptimizationPassBase


class DeadLabelEliminationPass(OptimizationPassBase):
    """Remove labels that are never referenced by any instruction."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        self.ctx = ctx
        if not self.ctx.config.enable_dead_label:
            return ir

        self._referenced_labels = {
            label
            for inst in walk_instructions(ir)
            if (label := inst.need_label) is not None
        }
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
