"""Core IR passes for optimization and validation."""

from __future__ import annotations

from typing import List, Union, cast

from .nodes import (
    IRBranch,
    IRCondJump,
    IRDelay,
    IRDelayAuto,
    IRJump,
    IRLabel,
    IRLoop,
    IRNode,
    IRPulse,
    IRPulseWmemReg,
    IRReadDmem,
    IRReadout,
    IRRegLoop,
    IRRegOp,
    IRSendReadoutConfig,
    IRSeq,
)
from .pass_base import Pass, PassConfig, PassCtx, PassPipeline


def estimate_insts(node: IRNode) -> int:
    """Rough instruction-count estimate for budget-aware unroll decisions.

    Constants are coarse approximations of QICK macro expansion, not exact:
      - Estimating high → fewer loops unrolled (safe).
      - Estimating low → may approach pmem cap; absorbed by the 20% safety
        margin applied at budget-injection time.

    Per-node estimates:
      - Pulse-like / delay / reg-op / read-dmem / jump / nop: 1
      - IRPulseWmemReg: 2 (address read + pulse trigger)
      - IRCondJump: 2 (TEST + JUMP)
      - IRLabel: 0 (label is an address, not an instruction)
      - IRSeq: sum of children
      - IRLoop / IRRegLoop: open(2) + body + close(2)
      - IRBranch: sum(arms) + 2 * (num_arms - 1) for binary dispatch
    """
    if isinstance(node, IRSeq):
        return sum(estimate_insts(child) for child in node.body)
    if isinstance(node, IRLoop):
        return 2 + estimate_insts(node.body) + 2
    if isinstance(node, IRRegLoop):
        return 2 + estimate_insts(node.body) + 2
    if isinstance(node, IRBranch):
        arms_total = sum(estimate_insts(arm) for arm in node.arms)
        dispatch_overhead = 2 * max(0, len(node.arms) - 1)
        return arms_total + dispatch_overhead
    if isinstance(node, IRPulseWmemReg):
        return 2
    if isinstance(node, IRCondJump):
        return 2
    if isinstance(node, IRLabel):
        return 0
    # Default leaf: pulse / readout / send_readoutconfig / delay / delay_auto /
    # reg_op / read_dmem / jump / nop -> 1
    return 1


class FreshLabels(Pass):
    """Rename all labels and jump targets to avoid collisions.

    Stateless: the rename map lives on ``PassCtx``. Runs FIRST so subsequent
    passes (e.g. UnrollShortLoops) that duplicate subtrees see already-canonical
    names — although duplicating labels still requires post-unroll uniqueness
    handling at the duplication site.
    """

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if isinstance(node, IRLabel):
            return IRLabel(name=ctx.fresh_label(node.name), meta=node.meta)

        if isinstance(node, IRJump):
            return IRJump(target=ctx.fresh_label(node.target), meta=node.meta)

        if isinstance(node, IRCondJump):
            return IRCondJump(
                target=ctx.fresh_label(node.target),
                arg1=node.arg1,
                test=node.test,
                op=node.op,
                arg2=node.arg2,
                meta=node.meta,
            )

        return node


class FlattenSeq(Pass):
    """Normalize IRSeq structure for downstream passes.

    Guarantees after this pass:
      - No IRSeq directly contains another IRSeq (children are spliced in).
      - No single-element IRSeq remains as a child of IRSeq / IRLoop body /
        IRRegLoop body / IRBranch arm — it is unwrapped to its sole child.
      - Empty IRSeq is preserved (emitter treats it as no-op).

    The builder always wraps loop bodies / branch arms / top-level emissions
    into IRSeq, so this pass is what restores compact, canonical structure.
    """

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if isinstance(node, IRSeq):
            flat: List[IRNode] = []
            for child in node.body:
                if isinstance(child, IRSeq):
                    flat.extend(child.body)
                else:
                    flat.append(child)
            return IRSeq(body=tuple(flat), meta=node.meta)

        if isinstance(node, IRLoop):
            return IRLoop(
                name=node.name,
                n=node.n,
                body=self._unwrap_singleton(node.body),
                meta=node.meta,
            )

        if isinstance(node, IRRegLoop):
            return IRRegLoop(
                name=node.name,
                n_reg=node.n_reg,
                body=self._unwrap_singleton(node.body),
                meta=node.meta,
            )

        if isinstance(node, IRBranch):
            return IRBranch(
                compare_reg=node.compare_reg,
                arms=tuple(self._unwrap_singleton(arm) for arm in node.arms),
                meta=node.meta,
            )

        return node

    @staticmethod
    def _unwrap_singleton(node: IRNode) -> IRNode:
        """If node is a 1-element IRSeq, return its sole child; else passthrough."""
        if isinstance(node, IRSeq) and len(node.body) == 1:
            return node.body[0]
        return node


class UnrollShortLoops(Pass):
    """Unroll small loops with few body instructions.

    Triggers when ``leaf_count(body) <= max_unroll_leaves`` AND
    ``n <= max_unroll_iters``. Leaf counting walks composite nodes recursively
    so that a loop wrapping a single pulse counts as 1 leaf, not 0.
    """

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if not isinstance(node, IRLoop):
            return node

        # If the body reads/writes the loop counter register, the register is
        # declared by the lowering layer's open_loop(). Unrolling removes that
        # declaration, so the body's RegOp/CondJump on the counter would
        # reference a non-existent register. Skip unroll in that case.
        if self._references_reg(node.body, node.name):
            ctx.warn(
                f"skip unroll loop '{node.name}': body references counter register"
            )
            return node

        leaf_count = self._count_leaves(node.body)
        if leaf_count > ctx.config.max_unroll_leaves:
            return node

        if node.n > ctx.config.max_unroll_iters:
            ctx.warn(
                f"skip unroll loop '{node.name}': n={node.n} exceeds "
                f"max_unroll_iters={ctx.config.max_unroll_iters}"
            )
            return node

        if node.n <= 0:
            return IRSeq(body=(), meta=node.meta)

        if node.n == 1:
            return node.body

        return IRSeq(body=tuple(node.body for _ in range(node.n)), meta=node.meta)

    @classmethod
    def _count_leaves(cls, node: IRNode) -> int:
        if isinstance(node, IRSeq):
            return sum(cls._count_leaves(child) for child in node.body)
        if isinstance(node, (IRLoop, IRRegLoop)):
            return cls._count_leaves(node.body)
        if isinstance(node, IRBranch):
            return sum(cls._count_leaves(arm) for arm in node.arms)
        return 1

    def _references_reg(self, node: IRNode, reg_name: str) -> bool:
        if isinstance(node, IRSeq):
            return any(self._references_reg(child, reg_name) for child in node.body)
        if isinstance(node, IRLoop):
            return self._references_reg(node.body, reg_name)
        if isinstance(node, IRRegLoop):
            return node.n_reg == reg_name or self._references_reg(node.body, reg_name)
        if isinstance(node, IRBranch):
            return any(self._references_reg(arm, reg_name) for arm in node.arms)
        if isinstance(node, IRRegOp):
            if node.dst == reg_name or node.lhs == reg_name:
                return True
            return isinstance(node.rhs, str) and node.rhs == reg_name
        if isinstance(node, IRReadDmem):
            return node.dst == reg_name or node.addr == reg_name
        if isinstance(node, IRCondJump):
            if node.arg1 == reg_name:
                return True
            return isinstance(node.arg2, str) and node.arg2 == reg_name
        return False


class FuseAdjacentDelays(Pass):
    """Fuse adjacent numeric IRDelay nodes (same tag) to reduce macro count.

    Assumes FlattenSeq has already removed nested IRSeq, so we don't need to
    splice children here.
    """

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if not isinstance(node, IRSeq):
            return node
        if not ctx.config.enable_fusion:
            return node

        fused: List[IRNode] = []
        for child in node.body:
            if (
                fused
                and isinstance(fused[-1], IRDelay)
                and isinstance(child, IRDelay)
                and isinstance(fused[-1].t, (int, float))
                and isinstance(child.t, (int, float))
                and fused[-1].tag == child.tag
            ):
                prev = cast(IRDelay, fused.pop())
                fused.append(
                    IRDelay(
                        t=float(prev.t) + float(child.t),
                        tag=prev.tag,
                        meta=prev.meta,
                    )
                )
            else:
                fused.append(child)

        return IRSeq(body=tuple(fused), meta=node.meta)


class RemoveZeroDelays(Pass):
    """Remove IRDelay nodes with exact numeric zero duration."""

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if not isinstance(node, IRSeq):
            return node

        compact: List[IRNode] = []
        for child in node.body:
            if isinstance(child, IRDelay) and isinstance(child.t, (int, float)):
                if float(child.t) == 0.0:
                    continue
            compact.append(child)
        return IRSeq(body=tuple(compact), meta=node.meta)


class ReorderPulseLikeByTime(Pass):
    """Reorder pulse-like nodes by ascending t within safe local segments."""

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if not isinstance(node, IRSeq):
            return node

        out: List[IRNode] = []
        segment: List[IRNode] = []

        def flush_segment() -> None:
            if not segment:
                return
            out.extend(self._reorder_segment(segment))
            segment.clear()

        for child in node.body:
            if self._is_barrier(child):
                flush_segment()
                out.append(child)
            else:
                segment.append(child)
        flush_segment()

        return IRSeq(body=tuple(out), meta=node.meta)

    def _reorder_segment(self, segment: List[IRNode]) -> List[IRNode]:
        slots = [i for i, n in enumerate(segment) if self._is_reorderable(n)]
        if len(slots) < 2:
            return list(segment)

        candidates = [
            cast(Union[IRPulse, IRSendReadoutConfig], segment[i]) for i in slots
        ]
        candidates_sorted = sorted(candidates, key=self._t_key)

        rebuilt = list(segment)
        for i, node in zip(slots, candidates_sorted):
            rebuilt[i] = node
        return rebuilt

    @staticmethod
    def _is_reorderable(node: IRNode) -> bool:
        return isinstance(node, (IRPulse, IRSendReadoutConfig))

    @staticmethod
    def _is_barrier(node: IRNode) -> bool:
        return isinstance(
            node,
            (
                IRDelay,
                IRDelayAuto,
                IRReadout,
                IRLabel,
                IRJump,
                IRCondJump,
                IRRegOp,
                IRReadDmem,
                IRLoop,
                IRRegLoop,
                IRBranch,
            ),
        )

    @staticmethod
    def _t_key(node: Union[IRPulse, IRSendReadoutConfig]) -> tuple[int, float]:
        t = node.t
        if isinstance(t, (int, float)):
            return (0, float(t))
        if hasattr(t, "minval"):
            min_v = t.minval()
            if isinstance(min_v, (int, float)):
                return (0, float(min_v))
        return (1, 0.0)


class ValidateInvariants(Pass):
    """Structural validations for emitter assumptions."""

    def __init__(self) -> None:
        self._defined_labels: set[str] = set()
        self._jump_targets: set[str] = set()
        self._branch_semantics_warned = False

    def __call__(self, node: IRNode, ctx: PassCtx | None = None) -> IRNode:
        if ctx is None:
            ctx = PassCtx()
        self._defined_labels.clear()
        self._jump_targets.clear()
        self._branch_semantics_warned = False

        out = super().__call__(node, ctx)

        undefined = sorted(self._jump_targets - self._defined_labels)
        for target in undefined:
            ctx.error(f"undefined jump target label: {target}")
        return out

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if isinstance(node, IRLabel):
            self._defined_labels.add(node.name)
        if isinstance(node, (IRJump, IRCondJump)):
            self._jump_targets.add(node.target)

        if isinstance(node, IRBranch):
            if len(node.arms) < 2:
                ctx.error("IRBranch requires at least 2 arms")
            elif not self._branch_semantics_warned:
                ctx.warn(
                    "IRBranch compare semantics: compare_reg < 0 selects first arm; "
                    "compare_reg >= num_arms selects last arm."
                )
                self._branch_semantics_warned = True
        if (
            isinstance(node, IRDelayAuto)
            and isinstance(node.t, str)
            and node.tag is not None
        ):
            ctx.error("IRDelayAuto tag is invalid when t is register name")
        return node


def make_default_pipeline(config: PassConfig | None = None) -> PassPipeline:
    """Build the default optimization + validation pipeline."""
    return PassPipeline(
        passes=[
            FreshLabels(),
            FlattenSeq(),
            UnrollShortLoops(),
            FlattenSeq(),
            FuseAdjacentDelays(),
            RemoveZeroDelays(),
            ReorderPulseLikeByTime(),
            ValidateInvariants(),
        ],
        config=config,
    )
