from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

from ..node import BlockNode, IRLoop, IRNode
from .base import OptimizationPassBase, loop_is_counter_sensitive
from ..labels import Label
from ..analysis import estimate_body_scheduled_ticks, estimate_flat_size


def _clone_nodes(nodes: list[IRNode], repeat: int) -> list[IRNode]:
    cloned: list[IRNode] = []
    for _ in range(repeat):
        cloned.extend(deepcopy(nodes))
    return cloned


class UnrollSmallLoopPass(OptimizationPassBase):
    """Scheduled-window-driven loop unrolling.

    Unrolls a loop by factor k when the loop control overhead (TEST + JUMP_back)
    consumes an unacceptably large fraction of the scheduled IO window available
    per iteration (sum of literal inc_ref delay ticks in the body).

    k is chosen as the minimum factor that brings the overhead fraction below
    `unroll_overhead_threshold`, capped by the pmem budget.
    """

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

        # scheduled_ticks: total literal inc_ref ticks available per iteration.
        # None means a dynamic delay is present — cannot analyze, skip unroll.
        scheduled_ticks = estimate_body_scheduled_ticks(node.body.insts)
        if scheduled_ticks is None:
            return self.generic_visit(node)

        cfg = self.ctx.config
        loop_overhead = 2 * cfg.cost_default + cfg.cost_jump_flush  # TEST + JUMP_back

        # Check if overhead is significant enough to bother unrolling.
        if loop_overhead / scheduled_ticks < cfg.unroll_overhead_threshold:
            return self.generic_visit(node)

        # Minimum k to push overhead fraction below threshold:
        #   loop_overhead / (k * scheduled_ticks) < threshold
        #   → k > loop_overhead / (threshold * scheduled_ticks)
        k_needed = math.ceil(
            loop_overhead / (cfg.unroll_overhead_threshold * scheduled_ticks)
        )

        # Cap k by pmem budget.
        body_size = estimate_flat_size(node.body.insts)
        if body_size > 0 and cfg.pmem_budget is not None:
            k_max_budget = cfg.pmem_budget // body_size
        else:
            k_max_budget = k_needed  # no budget constraint

        k = min(k_needed, k_max_budget)

        if k <= 1:
            return self.generic_visit(node)

        # Full expansion: all n copies fit inline.
        if n <= k:
            if is_runtime:
                # Exact range_hint — safe to expand; register value equals n.
                pass
            self._bump_stat("unroll_loop.removed")
            return _clone_nodes(node.body.insts, n)

        # Partial unroll: register-driven loops cannot be partially unrolled
        # without emitting runtime arithmetic to divide the counter by k.
        if is_runtime:
            return self.generic_visit(node)

        iters = n // k
        remainder = n % k

        self._bump_stat("unroll_loop.partial")

        result: list[IRNode] = []
        if iters > 0:
            unrolled_body = _clone_nodes(node.body.insts, k)
            new_loop = IRLoop(
                name=f"{node.name}_unrolled",
                counter_reg=node.counter_reg,
                n=iters,
                range_hint=(iters, iters),
                start_label=Label.make_new(f"{node.name}_unrolled_start"),
                end_label=Label.make_new(f"{node.name}_unrolled_end"),
                body=BlockNode(insts=unrolled_body),
            )
            result.append(new_loop)

        if remainder > 0:
            result.extend(_clone_nodes(node.body.insts, remainder))

        return result
