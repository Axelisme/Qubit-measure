from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

from ..analysis import (
    estimate_body_cost,
    estimate_body_scheduled_ticks,
    estimate_flat_size,
)
from ..labels import Label
from ..node import BlockNode, IRLoop, IRNode
from .base import OptimizationPassBase, loop_is_counter_sensitive


def _clone_nodes(nodes: list[IRNode], repeat: int) -> list[IRNode]:
    cloned: list[IRNode] = []
    for _ in range(repeat):
        cloned.extend(deepcopy(nodes))
    return cloned


def _select_unroll_k(
    body_insts: list[IRNode],
    loop_overhead: int,
    config,
) -> int:
    """Joint k selection (Phase 8 design).

    slack       = scheduled_ticks - body_cost
    if slack <= 0:
        k_timing = max_unroll_factor       # body already overloaded
    else:
        k_timing = ceil(loop_overhead / slack)
        k_timing = min(k_timing, max_unroll_factor)

    k_budget = pmem_budget // body_size  if body_size > 0 else k_timing
    k        = min(k_timing, k_budget)
    """
    scheduled_ticks = estimate_body_scheduled_ticks(body_insts)
    if scheduled_ticks is None:
        return 1  # no scheduled IO at all → no benefit from unroll

    body_cost = estimate_body_cost(body_insts, config)
    slack = scheduled_ticks - body_cost

    if slack <= 0:
        k_timing = config.max_unroll_factor
    else:
        k_timing = min(
            math.ceil(loop_overhead / slack), config.max_unroll_factor
        )

    body_size = estimate_flat_size(body_insts)
    if body_size > 0 and config.pmem_budget is not None:
        k_budget = config.pmem_budget // body_size
    else:
        k_budget = k_timing

    return max(1, min(k_timing, k_budget))


class UnrollSmallLoopPass(OptimizationPassBase):
    """Scheduled-window-driven loop unrolling (Phase 8).

    k is chosen jointly from per-iteration timing slack and pmem budget.
    Visits the body first (post-order) so inner loops are already unrolled
    before the outer loop's body_words is measured.
    """

    def visit_IRLoop(self, node: IRLoop) -> Optional[IRNode | list[IRNode]]:
        if not self.ctx.config.enable_unroll_loop:
            return self.generic_visit(node)

        # Post-order: recurse into the body first so any inner loops are
        # rewritten before we measure this loop's body size. generic_visit
        # mutates and returns the same IRLoop instance.
        visited = self.generic_visit(node)
        assert isinstance(visited, IRLoop)
        node = visited

        n: Optional[int] = None
        is_runtime = False
        if isinstance(node.n, int):
            n = node.n
        elif node.range_hint is not None and node.range_hint[0] == node.range_hint[1]:
            n = node.range_hint[0]
            is_runtime = True

        if n is None or n <= 0:
            return node  # register-driven without exact hint — Phase 8D

        if loop_is_counter_sensitive(node):
            return node

        cfg = self.ctx.config
        loop_overhead = 2 * cfg.cost_default + cfg.cost_jump_flush  # TEST + JUMP_back
        k = _select_unroll_k(node.body.insts, loop_overhead, cfg)

        if k <= 1:
            return node

        # Full expansion: all n copies fit inline.
        if n <= k:
            self._bump_stat("unroll_loop.removed")
            return BlockNode(insts=_clone_nodes(node.body.insts, n))

        # Partial unroll: register-driven (exact hint) loops cannot be
        # partially unrolled without runtime division logic.
        if is_runtime:
            return node

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

        return BlockNode(insts=result)
