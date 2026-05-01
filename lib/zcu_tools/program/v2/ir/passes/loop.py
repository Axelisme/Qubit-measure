from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Union

from ..node import IRLoop, IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class ConstantLoopUnrollPass(AbsPipeLinePass, IRTransformer):
    """Unroll loops with an explicit small constant trip count."""

    def __init__(self, max_trip_count: int = 16):
        self.max_trip_count = max_trip_count

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return res or ir

    def visit_IRLoop(self, node: IRLoop) -> Union[IRNode, List[IRNode], None]:
        # First recurse into inner structures
        self.generic_visit(node)

        # Try unrolling this loop
        unrolled = self._try_unroll(node)
        if unrolled is not None:
            return unrolled
        return node

    def _try_unroll(self, loop: IRLoop) -> Optional[List[IRNode]]:
        trip_count = loop.trip_count
        if trip_count is None:
            return None
        if trip_count < 0:
            raise ValueError(f"IRLoop '{loop.name}' trip_count must be non-negative")
        if trip_count > self.max_trip_count:
            return None

        out: list[IRNode] = []
        out.extend(deepcopy(loop.initial.insts))
        for _ in range(trip_count):
            out.extend(deepcopy(loop.body.insts))
            out.extend(deepcopy(loop.update.insts))
        return out
