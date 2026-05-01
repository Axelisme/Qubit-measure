from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Union, cast

from ..node import IRLoop, IRNode, RootNode, InstNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class ConstantLoopUnrollPass(AbsPipeLinePass, IRTransformer):
    """Unroll loops with an explicit small constant trip count."""

    def __init__(self, max_trip_count: int = 16):
        self.max_trip_count = max_trip_count

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return cast(RootNode, res or ir)

    def visit_IRLoop(self, node: IRLoop) -> Union[IRNode, List[IRNode], None]:
        # First recurse into inner structures
        self.generic_visit(node)

        # Try unrolling this loop
        unrolled = self._try_unroll(node)
        if unrolled is not None:
            return unrolled
        return node

    def _try_unroll(self, loop: IRLoop) -> Optional[List[IRNode]]:
        if not isinstance(loop.n, int):
            return None
        trip_count = loop.n
        if trip_count < 0:
            raise ValueError(f"IRLoop '{loop.name}' trip_count must be non-negative")
        if trip_count > self.max_trip_count:
            return None

        from ..instructions import RegWriteInst

        out: list[IRNode] = []
        # Initialize counter
        out.append(InstNode(RegWriteInst(dst=loop.counter_reg, src="imm", extra_args={"LIT": "#0"})))
        for _ in range(trip_count):
            out.extend(deepcopy(loop.body.insts))
            # body.insts naturally contains the IncReg for the counter, 
            # so we don't need to manually emit an update block.
        return out
