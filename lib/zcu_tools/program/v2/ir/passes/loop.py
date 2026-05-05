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
from .loop_dispatch import IRJumpTableLoop, shift_add_multiply


def _clone_nodes(nodes: list[IRNode], repeat: int) -> list[IRNode]:
    cloned: list[IRNode] = []
    for _ in range(repeat):
        cloned.extend(deepcopy(nodes))
    return cloned


def _floor_pow2(x: int) -> int:
    """Largest power of 2 that is <= x. Returns 0 if x < 1."""
    if x < 1:
        return 0
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


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

        if loop_is_counter_sensitive(node):
            return node

        cfg = self.ctx.config
        loop_overhead = 2 * cfg.cost_default + cfg.cost_jump_flush  # TEST + JUMP_back

        n: Optional[int] = None
        is_runtime_exact = False
        if isinstance(node.n, int):
            n = node.n
        elif node.range_hint is not None and node.range_hint[0] == node.range_hint[1]:
            n = node.range_hint[0]
            is_runtime_exact = True

        if n is not None and n <= 0:
            return node

        # ── Constant / exact-hint path ─────────────────────────────
        if n is not None:
            k = _select_unroll_k(node.body.insts, loop_overhead, cfg)
            if k <= 1:
                return node

            # Full expansion: all n copies fit inline.
            if n <= k:
                self._bump_stat("unroll_loop.removed")
                return BlockNode(insts=_clone_nodes(node.body.insts, n))

            # Partial unroll cannot use a constant remainder when the
            # iteration count is only known at runtime — the register
            # value would have to be divided by k at execution time.
            if is_runtime_exact:
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

        # ── Register-driven (no exact hint) → jump-table dispatch ──
        if not isinstance(node.n, str):
            return node  # unexpected n type
        return self._maybe_build_jump_table(node, loop_overhead, cfg) or node

    def _maybe_build_jump_table(
        self, node: IRLoop, loop_overhead: int, cfg
    ) -> Optional[IRNode]:
        """Try to build an IRJumpTableLoop for a register-driven loop.

        Returns None when any precondition fails (k <= 1 after pow2
        rounding, body_words == 0, dispatch shift-add too long, etc.) so
        the caller falls back to no-unroll.
        """
        body_size = estimate_flat_size(node.body.insts)
        if body_size <= 0:
            return None

        k_raw = _select_unroll_k(node.body.insts, loop_overhead, cfg)
        if k_raw <= 1:
            return None

        k = _floor_pow2(k_raw)
        if k <= 1:
            return None

        # Probe the dispatch shift-add — if body_words can't be encoded
        # within the word budget, abort before constructing the node.
        probe = shift_add_multiply(
            src_reg=node.counter_reg,
            dst_reg="s15",
            constant=body_size,
            max_words=cfg.max_dispatch_words,
        )
        if probe is None:
            return None

        entry_labels = [
            Label.make_new(f"{node.name}_jt_entry_{i}") for i in range(k)
        ]
        exit_label = Label.make_new(f"{node.name}_jt_exit")
        bodies = [
            BlockNode(insts=deepcopy(node.body.insts)) for _ in range(k)
        ]

        self._bump_stat("unroll_loop.register_partial")
        return IRJumpTableLoop(
            n_reg=str(node.n),
            counter_reg=node.counter_reg,
            k=k,
            body_words=body_size,
            entry_labels=entry_labels,
            exit_label=exit_label,
            bodies=bodies,
            name=node.name,
        )
