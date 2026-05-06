from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from ..analysis import (
    estimate_body_cost,
    estimate_body_scheduled_ticks,
    estimate_flat_size,
)
from ..instructions import RegWriteInst
from ..labels import Label
from ..node import BlockNode, InstNode, IRJumpTableLoop, IRLoop, IRNode
from .base import OptimizationPassBase
from .loop_dispatch import shift_add_multiply

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnrollAnalysis:
    scheduled_ticks: Optional[int]
    body_cost: int
    slack: Optional[int]
    body_size: int
    k_timing: int
    k_budget: int
    k_final: int


def _clone_nodes_with_incr(
    nodes: list[IRNode], repeat: int, counter_reg: str, final_incr: bool = True
) -> list[IRNode]:
    """Clone nodes `repeat` times, inserting a counter increment after each copy.

    If final_incr is False, the last copy does NOT get an increment appended
    (useful when the caller's IRLoop.emit() will add the final one).
    """
    incr = InstNode(RegWriteInst(dst=counter_reg, src="op", op=f"{counter_reg} + #1"))
    result: list[IRNode] = []
    for i in range(repeat):
        result.extend(deepcopy(nodes))
        if final_incr or i < repeat - 1:
            result.append(deepcopy(incr))
    return result


def _floor_pow2(x: int) -> int:
    """Largest power of 2 that is <= x. Returns 0 if x < 1."""
    if x < 1:
        return 0
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def _analyze_unroll(
    body_insts: list[IRNode],
    loop_overhead: int,
    config,
) -> UnrollAnalysis:
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
    body_cost = estimate_body_cost(body_insts, config)
    body_size = estimate_flat_size(body_insts)

    if scheduled_ticks is None:
        return UnrollAnalysis(
            scheduled_ticks=None,
            body_cost=body_cost,
            slack=None,
            body_size=body_size,
            k_timing=1,
            k_budget=1,
            k_final=1,
        )

    slack = scheduled_ticks - body_cost

    if slack <= 0:
        k_timing = config.max_unroll_factor
    else:
        k_timing = min(math.ceil(loop_overhead / slack), config.max_unroll_factor)

    if body_size > 0 and config.pmem_budget is not None:
        k_budget = config.pmem_budget // body_size
    else:
        k_budget = k_timing

    return UnrollAnalysis(
        scheduled_ticks=scheduled_ticks,
        body_cost=body_cost,
        slack=slack,
        body_size=body_size,
        k_timing=k_timing,
        k_budget=k_budget,
        k_final=max(1, min(k_timing, k_budget)),
    )


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
            logger.debug(
                "UnrollSmallLoopPass: skip loop name=%s because n=%s <= 0",
                node.name,
                n,
            )
            return node

        # ── Constant / exact-hint path ─────────────────────────────
        if n is not None:
            analysis = _analyze_unroll(node.body.insts, loop_overhead, cfg)
            logger.debug(
                "UnrollSmallLoopPass: analyze constant/exact loop name=%s n=%s exact=%s "
                "scheduled_ticks=%s body_cost=%s slack=%s body_size=%s "
                "k_timing=%s k_budget=%s k_final=%s",
                node.name,
                n,
                is_runtime_exact,
                analysis.scheduled_ticks,
                analysis.body_cost,
                analysis.slack,
                analysis.body_size,
                analysis.k_timing,
                analysis.k_budget,
                analysis.k_final,
            )
            k = analysis.k_final
            if k <= 1:
                logger.debug(
                    "UnrollSmallLoopPass: skip loop name=%s because k_final=%s <= 1",
                    node.name,
                    k,
                )
                return node

            iters = n // k
            remainder = n % k

            # Full expansion: n // k <= 1. A 1-iteration loop + remainder uses
            # more pmem (2 overhead words) than just fully expanding n copies inline.
            if iters <= 1:
                logger.debug(
                    "UnrollSmallLoopPass: fully expand loop name=%s n=%s k=%s (iters <= 1)",
                    node.name,
                    n,
                    k,
                )
                self._bump_stat("unroll_loop.removed")
                # Preserve loop semantics: counter starts at 0, increments after
                # each body copy. No surrounding IRLoop, so we emit all increments.
                return BlockNode(
                    insts=[
                        InstNode(
                            RegWriteInst(dst=node.counter_reg, src="imm", lit="#0")
                        ),
                        *_clone_nodes_with_incr(node.body.insts, n, node.counter_reg),
                    ]
                )

            # Partial unroll cannot use a constant remainder when the
            # iteration count is only known at runtime — the register
            # value would have to be divided by k at execution time.
            if is_runtime_exact:
                logger.debug(
                    "UnrollSmallLoopPass: skip partial unroll for loop name=%s because "
                    "range_hint is exact runtime-only n=%s and n > k=%s",
                    node.name,
                    n,
                    k,
                )
                return node

            logger.debug(
                "UnrollSmallLoopPass: partially expand loop name=%s n=%s k=%s "
                "iters=%s remainder=%s",
                node.name,
                n,
                k,
                iters,
                remainder,
            )
            self._bump_stat("unroll_loop.partial")

            result: list[IRNode] = []
            # Each of the k body copies needs its own counter increment.
            # IRLoop.emit() will append the final increment after all k copies,
            # so we omit it from the last copy (final_incr=False).
            unrolled_body = _clone_nodes_with_incr(
                node.body.insts, k, node.counter_reg, final_incr=False
            )
            full_iters = iters * k
            new_loop = IRLoop(
                name=f"{node.name}_unrolled",
                counter_reg=node.counter_reg,
                # Keep the original counter semantics: this loop body now
                # executes k original iterations per loop-round, so stop at
                # `iters * k` before appending remainder copies.
                n=full_iters,
                range_hint=(full_iters, full_iters),
                start_label=Label.make_new(f"{node.name}_unrolled_start"),
                end_label=Label.make_new(f"{node.name}_unrolled_end"),
                body=BlockNode(insts=unrolled_body),
            )
            result.append(new_loop)
            if remainder > 0:
                result.extend(
                    _clone_nodes_with_incr(node.body.insts, remainder, node.counter_reg)
                )
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
        analysis = _analyze_unroll(node.body.insts, loop_overhead, cfg)
        body_size = analysis.body_size
        logger.debug(
            "UnrollSmallLoopPass: analyze register-driven loop name=%s n_reg=%s "
            "scheduled_ticks=%s body_cost=%s slack=%s body_size=%s "
            "k_timing=%s k_budget=%s k_raw=%s",
            node.name,
            node.n,
            analysis.scheduled_ticks,
            analysis.body_cost,
            analysis.slack,
            analysis.body_size,
            analysis.k_timing,
            analysis.k_budget,
            analysis.k_final,
        )
        if body_size <= 0:
            logger.debug(
                "UnrollSmallLoopPass: skip jump-table loop name=%s because body_size=%s <= 0",
                node.name,
                body_size,
            )
            return None

        k_raw = analysis.k_final
        if k_raw <= 1:
            logger.debug(
                "UnrollSmallLoopPass: skip jump-table loop name=%s because k_raw=%s <= 1",
                node.name,
                k_raw,
            )
            return None

        k = _floor_pow2(k_raw)
        if k <= 1:
            logger.debug(
                "UnrollSmallLoopPass: skip jump-table loop name=%s because floor_pow2(%s)=%s <= 1",
                node.name,
                k_raw,
                k,
            )
            return None

        # Probe the dispatch shift-add — stride is body_size + 1 because
        # emit_jump_table_loop appends a per-iteration counter increment after
        # each body copy. If the stride can't be encoded within the word
        # budget, abort before constructing the node.
        stride = body_size + 1
        probe = shift_add_multiply(
            src_reg=node.counter_reg,
            dst_reg="s15",
            constant=stride,
            max_words=cfg.max_dispatch_words,
        )
        if probe is None:
            logger.debug(
                "UnrollSmallLoopPass: skip jump-table loop name=%s because "
                "shift_add_multiply(stride=%s, max_dispatch_words=%s) failed",
                node.name,
                stride,
                cfg.max_dispatch_words,
            )
            return None

        logger.debug(
            "UnrollSmallLoopPass: build jump-table loop name=%s n_reg=%s "
            "k_raw=%s k_pow2=%s body_words=%s dispatch_words=%s",
            node.name,
            node.n,
            k_raw,
            k,
            body_size,
            len(probe),
        )
        entry_labels = [Label.make_new(f"{node.name}_jt_entry_{i}") for i in range(k)]
        exit_label = Label.make_new(f"{node.name}_jt_exit")
        bodies = [BlockNode(insts=deepcopy(node.body.insts)) for _ in range(k)]

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
