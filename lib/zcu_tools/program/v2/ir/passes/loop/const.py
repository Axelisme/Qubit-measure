from __future__ import annotations

from copy import deepcopy
from typing import Optional

from ...node import IRLoop, IRNode
from ..base import OptimizationPassBase, loop_is_label_sensitive


def _clone_nodes(nodes: list[IRNode], repeat: int) -> list[IRNode]:
    cloned: list[IRNode] = []
    for _ in range(repeat):
        cloned.extend(deepcopy(nodes))
    return cloned


class UnrollSmallLoopPass(OptimizationPassBase):
    """Expand small, structurally simple IRLoop nodes into repeated bodies."""

    def visit_IRLoop(self, node: IRLoop) -> Optional[IRNode | list[IRNode]]:
        if not self.ctx.config.enable_unroll_loop:
            return self.generic_visit(node)

        if (
            type(node.n) is int
            and 1 <= node.n <= self.ctx.config.max_loop_unroll_count
            and not loop_is_label_sensitive(node)
        ):
            self._bump_stat("unroll_loop.removed")
            return _clone_nodes(node.body.insts, node.n)

        return self.generic_visit(node)
