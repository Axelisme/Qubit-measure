from __future__ import annotations

from typing import cast

from ..node import RootNode
from ..pipeline import AbsIRPass, PipeLineContext
from ..traversal import IRTransformer


class OptimizationPassBase(AbsIRPass, IRTransformer):
    """Base class for IR-level optimization passes with recursive IR traversal."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        self.ctx = ctx
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        return cast(RootNode, res or ir)
