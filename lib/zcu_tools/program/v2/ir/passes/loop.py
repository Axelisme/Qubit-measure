from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Optional, cast

from ..analysis import (
    estimate_body_cost,
    estimate_body_scheduled_ticks,
    estimate_flat_size,
)
from ..dispatch import _needs_big_jump, dispatch_entry_words
from ..factory import IRParser
from ..instructions import JumpInst, LabelInst, RegWriteInst
from ..labels import Label
from ..node import BasicBlockNode, BlockNode, IRLoop, IRNode, RootNode
from ..operands import AluExpr, AluOp, Immediate, Register, SrcKeyword
from ..pipeline import PipeLineContext
from .base import OptimizationPassBase
from .loop_dispatch import build_jump_table_blocks

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


def _clone_body_nodes(
    nodes: list[IRNode],
    repeat: int,
    pmem_size: int | None = None,
) -> list[BasicBlockNode]:
    """Clone `repeat` full loop-body copies as BasicBlockNodes.

    The loop body is trusted to already include the loop-carried counter
    update. Later linear passes may merge or reorder that write, so unroll
    logic must clone the body as a semantic unit instead of trying to append
    a structural increment block.
    """
    result: list[BasicBlockNode] = []
    for _ in range(repeat):
        lowered = IRParser(pmem_size=pmem_size).lower_block(
            BlockNode(insts=deepcopy(nodes))
        )
        result.extend(lowered)
    return result


def _prepend_label_to_body(body: list, label: Label) -> None:
    """Insert a LabelInst for `label` at the front of the first BasicBlockNode.
    If body is empty or the first item is not a BasicBlockNode, prepend a new one."""
    if body and isinstance(body[0], BasicBlockNode):
        body[0].labels.insert(0, LabelInst(name=label))
    else:
        body.insert(0, BasicBlockNode(labels=[LabelInst(name=label)]))


def _floor_pow2(x: int) -> int:
    """Largest power of 2 that is <= x. Returns 0 if x < 1."""
    if x < 1:
        return 0
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def _analyze_unroll(
    body_insts: list[IRNode], loop_overhead: int, ctx: PipeLineContext
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
    body_cost = estimate_body_cost(body_insts, ctx.config)
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
        k_timing = ctx.config.max_unroll_factor
    else:
        k_timing = min(math.ceil(loop_overhead / slack), ctx.config.max_unroll_factor)

    if body_size > 0 and ctx.pmem_budget is not None:
        k_budget = ctx.pmem_budget // body_size
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


class UnrollLoopPass(OptimizationPassBase):
    """Scheduled-window-driven loop unrolling (Phase 8).

    k is chosen jointly from per-iteration timing slack and pmem budget.
    Visits the body first (post-order) so inner loops are already unrolled
    before the outer loop's body_words is measured.
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> tuple[RootNode, bool]:
        self.ctx = ctx
        self._changed = False
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        out = cast(RootNode, res or ir)
        return out, self._changed

    def visit_IRLoop(self, node: IRLoop) -> Optional[IRNode | list[IRNode]]:
        # Post-order: recurse into the body first so any inner loops are
        # rewritten before we measure this loop's body size. generic_visit
        # mutates and returns the same IRLoop instance.
        visited = self.generic_visit(node)
        assert isinstance(visited, IRLoop)
        node = visited

        cfg = self.ctx.config
        # Counter update cost stays inside body_cost because it exists in both
        # the rolled and unrolled forms. This overhead models only the single
        # condensed back-edge JUMP plus its control-flow flush penalty.
        loop_overhead = cfg.cost_default + cfg.cost_jump_flush

        n: Optional[int] = None
        is_runtime_exact = False
        if isinstance(node.n, int):
            n = node.n
        elif node.range_hint is not None and node.range_hint[0] == node.range_hint[1]:
            n = node.range_hint[0]
            is_runtime_exact = True

        if n is not None and n <= 0:
            logger.debug(
                "UnrollLoopPass: skip loop name=%s because n=%s <= 0",
                node.name,
                n,
            )
            return node

        # ── Constant / exact-hint path ─────────────────────────────
        if n is not None:
            analysis = _analyze_unroll(node.body.insts, loop_overhead, self.ctx)
            logger.debug(
                "UnrollLoopPass: analyze constant/exact loop name=%s n=%s exact=%s "
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
                    "UnrollLoopPass: skip loop name=%s because k_final=%s <= 1",
                    node.name,
                    k,
                )
                return node

            iters = n // k
            remainder = n % k

            # Full expansion: n <= k. A loop of n copies is at most k copies,
            # which is our unroll limit anyway. Fully expanding saves loop overhead.
            if n <= k:
                logger.debug(
                    "UnrollLoopPass: fully expand loop name=%s n=%s k=%s (n <= k)",
                    node.name,
                    n,
                    k,
                )
                # Preserve loop semantics: counter starts at 0. The cloned body
                # already contains the loop-carried update as a semantic unit.
                pmem_size = self.ctx.config.pmem_capacity
                init_bb = BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=Register(node.counter_reg),
                            src=SrcKeyword.IMM,
                            lit=Immediate(0),
                        )
                    ]
                )
                self._changed = True
                return [
                    init_bb,
                    *_clone_body_nodes(node.body.insts, n, pmem_size=pmem_size),
                ]

            # Partial unroll cannot use a constant remainder when the
            # iteration count is only known at runtime — the register
            # value would have to be divided by k at execution time.
            if is_runtime_exact:
                logger.debug(
                    "UnrollLoopPass: skip partial unroll for loop name=%s because "
                    "range_hint is exact runtime-only n=%s and n > k=%s",
                    node.name,
                    n,
                    k,
                )
                return node

            logger.debug(
                "UnrollLoopPass: partially expand loop name=%s n=%s k=%s "
                "iters=%s remainder=%s",
                node.name,
                n,
                k,
                iters,
                remainder,
            )

            pmem_size = self.ctx.config.pmem_capacity
            result: list[IRNode] = []

            if remainder > 0:
                entry_label = Label.make_new(f"{node.name}_remainder_entry")

                part1 = _clone_body_nodes(
                    node.body.insts, k - remainder, pmem_size=pmem_size
                )
                part2 = _clone_body_nodes(
                    node.body.insts, remainder, pmem_size=pmem_size
                )

                if not part2:
                    part2 = [BasicBlockNode(labels=[LabelInst(name=entry_label)])]
                else:
                    part2[0].labels.append(LabelInst(name=entry_label))

                unrolled_body = part1 + part2

                if _needs_big_jump(pmem_size):
                    init_bb = BasicBlockNode(
                        insts=[
                            RegWriteInst(
                                dst=Register(node.counter_reg),
                                src=SrcKeyword.IMM,
                                lit=Immediate(0),
                            ),
                            RegWriteInst(
                                dst=Register("s15"),
                                src=SrcKeyword.LABEL,
                                label=entry_label,
                            ),
                        ],
                        branch=JumpInst(addr=Register("s15")),
                    )
                else:
                    init_bb = BasicBlockNode(
                        insts=[
                            RegWriteInst(
                                dst=Register(node.counter_reg),
                                src=SrcKeyword.IMM,
                                lit=Immediate(0),
                            )
                        ],
                        branch=JumpInst(label=entry_label),
                    )
                result.append(init_bb)
            else:
                unrolled_body = _clone_body_nodes(
                    node.body.insts, k, pmem_size=pmem_size
                )

            # Build flat loop structure: start label → unrolled body → back-edge → end label.
            # Do not wrap in IRLoop because the body is no longer a single canonical
            # iteration; wrapping would cause parse/unparse to reconstruct an IRLoop
            # that gets unrolled again on the next pipeline iteration.
            start = Label.make_new(f"{node.name}_unrolled_start")
            end = Label.make_new(f"{node.name}_unrolled_end")

            if remainder == 0:
                # No init_bb yet: insert counter init before start label.
                result.append(
                    BasicBlockNode(
                        insts=[
                            RegWriteInst(
                                dst=Register(node.counter_reg),
                                src=SrcKeyword.IMM,
                                lit=Immediate(0),
                            )
                        ]
                    )
                )

            _prepend_label_to_body(unrolled_body, start)
            result.extend(unrolled_body)

            # Back-edge: jump back to start while counter < n.
            counter = Register(node.counter_reg)
            n_val = Immediate(n)
            op_str = AluExpr(counter, AluOp.SUB, n_val)
            if _needs_big_jump(pmem_size):
                back_bb = BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=Register("s15"), src=SrcKeyword.LABEL, label=start
                        )
                    ],
                    branch=JumpInst(addr=Register("s15"), if_cond="S", op=op_str),
                )
            else:
                back_bb = BasicBlockNode(
                    branch=JumpInst(label=start, if_cond="S", op=op_str)
                )
            result.append(back_bb)
            result.append(BasicBlockNode(labels=[LabelInst(name=end)]))

            self._changed = True
            return BlockNode(insts=result)

        # ── Register-driven (no exact hint) → jump-table dispatch ──
        if not isinstance(node.n, str):
            return node  # unexpected n type
        jt_blocks = self._maybe_build_jump_table(node, loop_overhead, self.ctx)
        if jt_blocks is None:
            return node
        self._changed = True
        return list(jt_blocks)  # list[BasicBlockNode] → list[IRNode] (coercion)

    def _maybe_build_jump_table(
        self, node: IRLoop, loop_overhead: int, ctx: PipeLineContext
    ) -> Optional[list[BasicBlockNode]]:
        """Try to build jump-table BasicBlockNodes for a register-driven loop.

        Returns None when any precondition fails (k <= 1 after pow2 rounding,
        body_words == 0, etc.) so the caller falls back to no-unroll.
        """
        analysis = _analyze_unroll(node.body.insts, loop_overhead, ctx)
        body_size = analysis.body_size
        logger.debug(
            "UnrollLoopPass: analyze register-driven loop name=%s n_reg=%s "
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
                "UnrollLoopPass: skip jump-table loop name=%s because body_size=%s <= 0",
                node.name,
                body_size,
            )
            return None

        k_raw = analysis.k_final
        if k_raw <= 1:
            logger.debug(
                "UnrollLoopPass: skip jump-table loop name=%s because k_raw=%s <= 1",
                node.name,
                k_raw,
            )
            return None

        k = _floor_pow2(k_raw)
        if k <= 1:
            logger.debug(
                "UnrollLoopPass: skip jump-table loop name=%s because floor_pow2(%s)=%s <= 1",
                node.name,
                k_raw,
                k,
            )
            return None

        logger.debug(
            "UnrollLoopPass: build jump-table loop name=%s n_reg=%s "
            "k_raw=%s k_pow2=%s body_words=%s entry_words=%s",
            node.name,
            node.n,
            k_raw,
            k,
            body_size,
            dispatch_entry_words(self.ctx.config.pmem_capacity),
        )

        entry_labels = [Label.make_new(f"{node.name}_jt_entry_{i}") for i in range(k)]
        exit_label = Label.make_new(f"{node.name}_jt_exit")
        bodies = [BlockNode(insts=deepcopy(node.body.insts)) for _ in range(k)]

        return build_jump_table_blocks(
            n_reg=str(node.n),
            counter_reg=node.counter_reg,
            k=k,
            entry_labels=entry_labels,
            exit_label=exit_label,
            bodies=bodies,
            pmem_size=self.ctx.config.pmem_capacity,
        )
