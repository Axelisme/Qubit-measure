"""UnrollLoopPass: scheduled-window-driven loop unrolling (Phase 8).

Purpose
-------
Tight loops in QICK programs often have a large IO scheduling window (TIME
inc_ref delay) relative to the actual instruction execution cost.  Unrolling
k copies of the body amortises the loop overhead (one back-edge JUMP + its
pipeline flush penalty) across k iterations, freeing more time for the
scheduler to do useful work.

Example
-------
Before (k=2, n=4)::

    counter = 0
    loop (4 times):
      PORT_WR ...
      REG_WR r1 op r1 + #1
      TIME inc_ref #200

After (partial unroll, k=2)::

    counter = 0
    loop_unrolled_start:
      PORT_WR ...          ; copy 0
      REG_WR r1 op r1 + #1
      TIME inc_ref #200
      PORT_WR ...          ; copy 1
      REG_WR r1 op r1 + #1
      TIME inc_ref #200
      JUMP loop_unrolled_start -if(S) -op(counter - 4)

QICK Hardware Notes
-------------------
- Loop overhead is modelled as ``cost_default + cost_jump_flush``: one JUMP
  instruction plus the pipeline flush penalty when the branch is taken.
- ``k`` is bounded by both timing slack and pmem budget simultaneously:
  - ``k_timing`` = ceil(loop_overhead / slack); body already overloaded → max_unroll_factor
  - ``k_budget``  = pmem_budget // body_size (words)
  - ``k_final``   = min(k_timing, k_budget)
- For register-driven loops (iteration count unknown at compile time), ``k``
  is rounded down to the nearest power of 2 so that ``n AND (k-1)`` computes
  the remainder with a single AND instruction.
- In big-PMEM mode (``_needs_big_jump``), all cross-section jumps use an
  indirect ``REG_WR s15 label / JUMP s15`` pair (2 words) instead of a
  direct 1-word ``JUMP label``.

Decision Notes
--------------
``transform`` reasons about the IR tree, not flat chunks: it uses the
``estimate_*`` cost/size functions on ``IRLoop.body``. Post-order convergence
means the body's nested structure is already settled when ``transform`` runs,
but the body has not been flattened to a ``BasicBlockNode`` list nor been
through ChunkPass optimization — so the estimates may slightly over-count
``body_cost`` / ``body_size``. This is an accepted approximation: a tighter
estimate would need inner-loop ``scheduled_ticks`` reconstructed from flat
blocks, which the flat form does not preserve.

Three unrolling strategies:
1. Full expansion (``n ≤ k``): emit n body copies, drop the loop entirely.
   Counter init is prepended; the cloned body already contains the
   loop-carried counter update.
2. Partial unroll (``n > k``, compile-time constant): emit a loop of
   ``n // k`` iterations over a k-copy body, plus a remainder prefix.
   Not applicable when n is only known at runtime (``is_runtime_exact``).
3. Register-driven (n unknown): returns a BlockNode containing prologue
   BasicBlockNodes + dispatch-table stubs + k body BlockNodes + back-edge.
   Pipeline recursively lowers the body BlockNodes so they benefit from
   ChunkPass optimizations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from typing_extensions import Optional, cast

from ...analysis import (
    estimate_body_cost,
    estimate_body_scheduled_ticks,
    estimate_flat_size,
)
from ...hw_semantics import needs_big_jump
from ...instructions import BaseInst, JumpInst, LabelInst, RegWriteInst
from ...labels import Label, LabelRef, make_label
from ...node import (
    BasicBlockNode,
    BlockNode,
    IRLoop,
    IRNode,
    _collect_subtree_names,
    clone_renamed,
)
from ...operands import AluExpr, AluOp, Immediate, Register, SrcKeyword
from ...pipeline import AbsIRTreePass, PipeLineContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnrollAnalysis:
    scheduled_ticks: int
    body_cost: int
    slack: int
    body_size: int
    k_timing: int
    k_budget: int
    k_final: int


def _clone_body_copies(nodes: list[IRNode], repeat: int) -> list[IRNode]:
    """Clone `repeat` full loop-body copies, preserving structure.

    Each copy is a deep clone with all internal labels and nested
    IRLoop / IRBranch / IRDispatch names uniquified, so the copies do not
    collide.  Nested structural nodes survive as IRNodes — the pipeline
    re-recurses the unroll result, so a nested loop that was not itself
    unrolled is still lowered correctly by unparse().

    A local `allocated` set seeded from the body's existing labels keeps
    successive copies disjoint; it is discarded when the function returns.
    """
    body_block = BlockNode(insts=list(nodes))
    label_names, struct_names = _collect_subtree_names(body_block)
    allocated: set[str] = set(label_names) | set(struct_names)
    result: list[IRNode] = []
    for _ in range(repeat):
        cloned = clone_renamed(body_block, allocated)
        assert isinstance(cloned, BlockNode)
        result.extend(cloned.insts)
    return result


def _prepend_label_to_body(body: list[IRNode], label: Label) -> None:
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

    slack = scheduled_ticks - body_cost

    if slack <= 0:
        k_timing = ctx.config.max_unroll_factor
    else:
        k_timing = min(math.ceil(loop_overhead / slack), ctx.config.max_unroll_factor)

    if body_size > 0:
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


class UnrollLoopPass(AbsIRTreePass):
    """Scheduled-window-driven loop unrolling.

    k is chosen jointly from per-iteration timing slack and pmem budget.
    Being an AbsIRTreePass, ``transform`` sees the IR tree (post-order
    converged children, not yet flattened) and bases the unroll decision on
    IR-tree cost/size estimates.  See the module docstring for the estimation
    tradeoff.
    """

    def transform(
        self,
        node: IRNode,
        ctx: PipeLineContext,
    ) -> Optional[IRNode]:
        if not isinstance(node, IRLoop):
            return None

        # Validation: check register safety for unrolling.
        from ...hw_semantics import ADDR_REG

        counter = node.counter_reg
        if not counter.is_general_reg():
            raise ValueError(
                f"UnrollLoopPass: loop {node.name!r} counter_reg {str(counter)!r} "
                f"is not a general-purpose register (r0-r14)."
            )

        if isinstance(node.n, Register):
            n_reg = node.n
            if n_reg.canonical_name == counter.canonical_name:
                raise ValueError(
                    f"UnrollLoopPass: loop {node.name!r} n_reg and counter_reg "
                    f"are the same register: {str(n_reg)!r}."
                )
            if n_reg.canonical_name == ADDR_REG:
                raise ValueError(
                    f"UnrollLoopPass: loop {node.name!r} n_reg {str(n_reg)!r} "
                    f"conflicts with reserved address register {ADDR_REG}."
                )

        cfg = ctx.config
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
                "UnrollLoopPass: skip loop name=%s because n=%s <= 0", node.name, n
            )
            return None

        if n is not None:
            return self._unroll_constant(node, n, is_runtime_exact, loop_overhead, ctx)

        # Register-driven (no exact hint) → jump-table dispatch.
        return self._maybe_build_jump_table(node, loop_overhead, ctx)

    def _unroll_constant(
        self,
        node: IRLoop,
        n: int,
        is_runtime_exact: bool,
        loop_overhead: int,
        ctx: PipeLineContext,
    ) -> Optional[IRNode]:
        """Handle loops with a known (compile-time or exact-hint) iteration count."""
        analysis = _analyze_unroll(cast(BlockNode, node.body).insts, loop_overhead, ctx)
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
            return None

        if n <= k:
            return self._unroll_full(node, n)

        if is_runtime_exact:
            logger.debug(
                "UnrollLoopPass: skip partial unroll for loop name=%s because "
                "range_hint is exact runtime-only n=%s and n > k=%s",
                node.name,
                n,
                k,
            )
            return None

        return self._unroll_partial(node, n, k, ctx)

    def _unroll_full(self, node: IRLoop, n: int) -> BlockNode:
        """Full expansion: emit n body copies, drop the loop entirely (n <= k)."""
        logger.debug("UnrollLoopPass: fully expand loop name=%s n=%s", node.name, n)
        body_insts = cast(BlockNode, node.body).insts
        init_bb = BasicBlockNode(
            insts=[
                RegWriteInst(dst=node.counter_reg, src=SrcKeyword.IMM, lit=Immediate(0))
            ]
        )
        return BlockNode(
            insts=[
                init_bb,
                *_clone_body_copies(body_insts, n),
            ]
        )

    def _unroll_partial(
        self, node: IRLoop, n: int, k: int, ctx: PipeLineContext
    ) -> BlockNode:
        """Partial unroll: loop of n//k iterations over a k-copy body + remainder prefix."""
        remainder = n % k
        logger.debug(
            "UnrollLoopPass: partially expand loop name=%s n=%s k=%s iters=%s remainder=%s",
            node.name,
            n,
            k,
            n // k,
            remainder,
        )
        pmem_size = ctx.config.pmem_capacity
        body_insts = cast(BlockNode, node.body).insts
        result: list[IRNode] = []

        # Local allocated set for labels generated within this unroll session.
        body_label_names, body_struct_names = _collect_subtree_names(
            BlockNode(insts=list(body_insts))
        )
        local_allocated: set[str] = set(body_label_names) | set(body_struct_names)

        if remainder > 0:
            entry_label = make_label(f"{node.name}_remainder_entry", local_allocated)
            part1 = _clone_body_copies(body_insts, k - remainder)
            part2 = _clone_body_copies(body_insts, remainder)

            if not part2:
                part2 = [BasicBlockNode(labels=[LabelInst(name=entry_label)])]
            else:
                _prepend_label_to_body(part2, entry_label)
            unrolled_body = part1 + part2

            if needs_big_jump(pmem_size):
                init_bb = BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=node.counter_reg, src=SrcKeyword.IMM, lit=Immediate(0)
                        ),
                        RegWriteInst(
                            dst=Register("s15"),
                            src=SrcKeyword.LABEL,
                            label=LabelRef(entry_label),
                        ),
                    ],
                    branch=JumpInst(addr=Register("s15")),
                )
            else:
                init_bb = BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=node.counter_reg, src=SrcKeyword.IMM, lit=Immediate(0)
                        )
                    ],
                    branch=JumpInst(label=LabelRef(entry_label)),
                )
            result.append(init_bb)
        else:
            unrolled_body = _clone_body_copies(body_insts, k)

        # Build flat loop: start label → unrolled body → back-edge → end label.
        # Not wrapped in IRLoop to prevent re-unrolling on the next pipeline iteration.
        start = make_label(f"{node.name}_unrolled_start", local_allocated)
        end = make_label(f"{node.name}_unrolled_end", local_allocated)

        if remainder == 0:
            result.append(
                BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=node.counter_reg, src=SrcKeyword.IMM, lit=Immediate(0)
                        )
                    ]
                )
            )

        _prepend_label_to_body(unrolled_body, start)
        result.extend(unrolled_body)

        op_str = AluExpr(node.counter_reg, AluOp.SUB, Immediate(n))
        if needs_big_jump(pmem_size):
            back_bb = BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("s15"), src=SrcKeyword.LABEL, label=LabelRef(start)
                    )
                ],
                branch=JumpInst(addr=Register("s15"), if_cond="S", op=op_str),
            )
        else:
            back_bb = BasicBlockNode(
                branch=JumpInst(label=LabelRef(start), if_cond="S", op=op_str)
            )
        result.append(back_bb)
        result.append(BasicBlockNode(labels=[LabelInst(name=end)]))

        return BlockNode(insts=result)

    def _maybe_build_jump_table(
        self, node: IRLoop, loop_overhead: int, ctx: PipeLineContext
    ) -> Optional[BlockNode]:
        """Try to build a jump-table BlockNode for a register-driven loop.

        Returns a BlockNode containing:
          [prologue BasicBlockNodes] + IRDispatch + [k body BlockNodes] +
          [back-edge BasicBlockNodes] + [exit BasicBlockNode]

        Returns None when any precondition fails (k <= 1, body_words == 0, etc.)
        so the caller falls back to no-unroll.
        """
        body_insts = cast(BlockNode, node.body).insts
        analysis = _analyze_unroll(body_insts, loop_overhead, ctx)
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

        if not ctx.available_regs:
            logger.debug(
                "UnrollLoopPass: skip jump-table loop name=%s because no available_regs",
                node.name,
            )
            return None

        pmem_size = ctx.config.pmem_capacity
        logger.debug(
            "UnrollLoopPass: build jump-table loop name=%s n_reg=%s "
            "k_raw=%s k_pow2=%s body_words=%s",
            node.name,
            node.n,
            k_raw,
            k,
            body_size,
        )

        assert isinstance(node.n, Register)
        i = node.counter_reg
        n = node.n
        # Pop one available register (deterministic: pick min)
        scratch = Register(min(ctx.available_regs))

        # Local allocated set for labels generated within this jump-table session.
        _jt_label_names, _jt_struct_names = _collect_subtree_names(
            BlockNode(insts=list(body_insts))
        )
        local_allocated: set[str] = set(_jt_label_names) | set(_jt_struct_names)
        entry_labels = [
            make_label(f"{node.name}_jt_entry_{idx}", local_allocated)
            for idx in range(k)
        ]
        exit_label = make_label(f"{node.name}_jt_exit", local_allocated)

        result: list[IRNode] = []

        # ── prologue ──────────────────────────────────────────────────────────
        # Reset counter i to 0 at the very start of the loop setup
        result.append(
            BasicBlockNode(
                insts=[RegWriteInst(dst=i, src=SrcKeyword.IMM, lit=Immediate(0))]
            )
        )

        # Guard: skip entirely when n == 0.
        if needs_big_jump(pmem_size):
            result.append(
                BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=Register("s15"),
                            src=SrcKeyword.LABEL,
                            label=LabelRef(exit_label),
                        )
                    ],
                    branch=JumpInst(
                        addr=Register("s15"),
                        if_cond="Z",
                        op=AluExpr(n, AluOp.SUB, Immediate(0)),
                    ),
                )
            )
        else:
            result.append(
                BasicBlockNode(
                    branch=JumpInst(
                        label=LabelRef(exit_label),
                        if_cond="Z",
                        op=AluExpr(n, AluOp.SUB, Immediate(0)),
                    ),
                )
            )

        # Compute remainder r = n % k (stored in scratch temporarily).
        # If r == 0, jump straight to entry_0 (full rounds only).
        if needs_big_jump(pmem_size):
            result.append(
                BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=scratch,
                            src=SrcKeyword.OP,
                            op=AluExpr(n, AluOp.AND, Immediate(k - 1)),
                        )
                    ]
                )
            )
            result.append(
                BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=Register("s15"),
                            src=SrcKeyword.LABEL,
                            label=LabelRef(entry_labels[0]),
                        )
                    ],
                    branch=JumpInst(
                        addr=Register("s15"),
                        if_cond="Z",
                        op=AluExpr(scratch, AluOp.SUB, Immediate(0)),
                    ),
                )
            )
        else:
            result.append(
                BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=scratch,
                            src=SrcKeyword.OP,
                            op=AluExpr(n, AluOp.AND, Immediate(k - 1)),
                        )
                    ],
                    branch=JumpInst(
                        label=LabelRef(entry_labels[0]),
                        if_cond="Z",
                        op=AluExpr(scratch, AluOp.SUB, Immediate(0)),
                    ),
                )
            )

        # Compute dispatch offset: scratch = k - r, then dispatch.
        from ...node import IRDispatch

        offset_insts: list[BaseInst] = [
            RegWriteInst(
                dst=scratch,
                src=SrcKeyword.OP,
                op=AluExpr(scratch, AluOp.SUB, Immediate(k)),
            ),  # scratch = r - k
            RegWriteInst(
                dst=scratch, src=SrcKeyword.OP, op=AluExpr(scratch, AluOp.ABS)
            ),  # scratch = k - r
        ]
        result.append(BasicBlockNode(insts=offset_insts))

        # ── dispatch node (lowered by pipeline → SimplifyDispatchPass or fallback) ──
        result.append(
            IRDispatch(
                name=f"{node.name}_jt_entry_0",
                value_reg=scratch,
                target_labels=entry_labels,
            )
        )

        # ── k body copies (free-form IRNodes, lowered by pipeline) ────────────
        for idx in range(k):
            entry_bb = BasicBlockNode(labels=[LabelInst(name=entry_labels[idx])])
            cloned_body = clone_renamed(
                BlockNode(insts=list(body_insts)), local_allocated
            )
            assert isinstance(cloned_body, BlockNode)
            body_block = BlockNode(insts=[entry_bb, *cloned_body.insts])
            result.append(body_block)

        # ── back edge ─────────────────────────────────────────────────────────
        # Use S (i < n) to jump back to entry_0; when condition fails (i >= n)
        # fall through to exit_label — mirrors normal loop back-edge structure.
        op_cmp = AluExpr(i, AluOp.SUB, n)
        if needs_big_jump(pmem_size):
            result.append(
                BasicBlockNode(
                    insts=[
                        RegWriteInst(
                            dst=Register("s15"),
                            src=SrcKeyword.LABEL,
                            label=LabelRef(entry_labels[0]),
                        )
                    ],
                    branch=JumpInst(addr=Register("s15"), if_cond="S", op=op_cmp),
                )
            )
        else:
            result.append(
                BasicBlockNode(
                    branch=JumpInst(
                        label=LabelRef(entry_labels[0]), if_cond="S", op=op_cmp
                    )
                )
            )

        # ── exit ──────────────────────────────────────────────────────────────
        result.append(
            BasicBlockNode(labels=[LabelInst(name=exit_label, can_remove=True)])
        )

        return BlockNode(insts=result)
