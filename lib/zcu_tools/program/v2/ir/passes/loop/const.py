from __future__ import annotations

from copy import deepcopy
from typing import Optional

from ...node import IRLoop, IRNode, BlockNode
from ..base import OptimizationPassBase, loop_is_counter_sensitive


def _clone_nodes(nodes: list[IRNode], repeat: int) -> list[IRNode]:
    cloned: list[IRNode] = []
    for _ in range(repeat):
        cloned.extend(deepcopy(nodes))
    return cloned


class UnrollSmallLoopPass(OptimizationPassBase):
    """Adaptive budget-driven loop unrolling."""

    def visit_IRLoop(self, node: IRLoop) -> Optional[IRNode | list[IRNode]]:
        if not self.ctx.config.enable_unroll_loop:
            return self.generic_visit(node)

        n = None
        is_runtime = False
        if isinstance(node.n, int):
            n = node.n
        elif node.range_hint is not None and node.range_hint[0] == node.range_hint[1]:
            n = node.range_hint[0]
            is_runtime = True

        if n is None or n <= 0:
            return self.generic_visit(node)

        if loop_is_counter_sensitive(node):
            return self.generic_visit(node)

        # Budget-driven logic
        from ...analysis import estimate_body_cost
        body_cost = estimate_body_cost(node.body.insts, self.ctx.config)

        # If it's short, we want to expand it to reduce control overhead.
        # We'll expand it fully if n <= max_loop_unroll_count.
        if n <= self.ctx.config.max_loop_unroll_count:
            if is_runtime:
                # If it's a register, we can't just drop the loop if we don't have
                # arbitrary control-flow rewrite (we would need to ignore the register).
                # But since range_hint is EXACT, the register is always `n`.
                # We can just unroll and ignore the register value if we assume the hint is a hard contract.
                pass
            self._bump_stat("unroll_loop.removed")
            return _clone_nodes(node.body.insts, n)

        # Partial unroll strategy: Unroll body by k times inside the loop
        # and leave a remainder outside.
        # We want body_cost * k to be roughly >= some threshold (e.g. 20 cycles)
        # bounded by max_loop_unroll_count.
        target_chunk_cost = 20
        if body_cost == 0:
            k = self.ctx.config.max_loop_unroll_count
        else:
            k = min(self.ctx.config.max_loop_unroll_count, max(1, target_chunk_cost // body_cost))

        # If k == 1, no unrolling needed.
        if k == 1:
            return self.generic_visit(node)

        if is_runtime:
            # For v1, if it's a register-driven loop, we don't partial unroll yet
            # because that requires emitting arithmetic to divide the register by k.
            # We only fully unroll if n <= max_loop_unroll_count.
            return self.generic_visit(node)

        iters = n // k
        remainder = n % k

        self._bump_stat("unroll_loop.partial")

        nodes: list[IRNode] = []
        if iters > 0:
            unrolled_body = _clone_nodes(node.body.insts, k)
            from ...labels import Label
            new_loop = IRLoop(
                name=f"{node.name}_unrolled",
                counter_reg=node.counter_reg,
                n=iters,
                range_hint=(iters, iters),
                start_label=Label.make_new(f"{node.name}_unrolled_start"),
                end_label=Label.make_new(f"{node.name}_unrolled_end"),
                body=BlockNode(insts=unrolled_body)
            )
            nodes.append(new_loop)

        if remainder > 0:
            nodes.extend(_clone_nodes(node.body.insts, remainder))

        return nodes
