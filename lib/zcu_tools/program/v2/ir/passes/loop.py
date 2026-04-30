from __future__ import annotations

from copy import deepcopy
from typing import Union

from ..instructions import Instruction
from ..node import BlockNode, IRLoop, IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext


class ConstantLoopUnrollPass(AbsPipeLinePass):
    """Unroll loops with an explicit small constant trip count."""

    def __init__(self, max_trip_count: int = 16):
        self.max_trip_count = max_trip_count

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        self._rewrite_node(ir)
        return ir

    def _rewrite_node(self, node: IRNode) -> None:
        if isinstance(node, IRLoop):
            self._rewrite_node(node.initial)
            self._rewrite_node(node.stop_check)
            self._rewrite_node(node.body)
            self._rewrite_node(node.update)
            return

        if not isinstance(node, BlockNode):
            return

        rewritten = []
        for item in node.insts:
            if isinstance(item, IRLoop):
                unrolled = self._try_unroll(item)
                if unrolled is None:
                    self._rewrite_node(item)
                    rewritten.append(item)
                else:
                    rewritten.extend(unrolled)
            elif isinstance(item, IRNode):
                self._rewrite_node(item)
                rewritten.append(item)
            else:
                rewritten.append(item)
        node.insts = rewritten

    def _try_unroll(self, loop: IRLoop) -> list[Union[Instruction, IRNode]] | None:
        trip_count = loop.trip_count
        if trip_count is None:
            return None
        if trip_count < 0:
            raise ValueError(f"IRLoop '{loop.name}' trip_count must be non-negative")
        if trip_count > self.max_trip_count:
            return None

        out: list[Union[Instruction, IRNode]] = []
        out.extend(deepcopy(loop.initial.insts))
        for _ in range(trip_count):
            out.extend(deepcopy(loop.body.insts))
            out.extend(deepcopy(loop.update.insts))
        return out
