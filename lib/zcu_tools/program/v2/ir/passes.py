"""Core IR passes for optimization and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Union, cast

from .nodes import (
    IRBranch,
    IRCondJump,
    IRDelay,
    IRDelayAuto,
    IRJump,
    IRLabel,
    IRLoop,
    IRNode,
    IRNop,
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

    Unknown node types raise TypeError so newly-added IR nodes must be
    classified explicitly here rather than silently defaulting.
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
    if isinstance(
        node,
        (
            IRPulse,
            IRSendReadoutConfig,
            IRReadout,
            IRDelay,
            IRDelayAuto,
            IRRegOp,
            IRReadDmem,
            IRJump,
            IRNop,
        ),
    ):
        return 1
    raise TypeError(f"estimate_insts: unhandled IR node type {type(node).__name__}")


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


@dataclass
class UnrollInfo:
    """Per-IRLoop metadata collected by MarkUnrollInfo for budget-aware unroll.

    Identification uses ``id(node)`` and is valid as long as the IRLoop instance
    is not rebuilt between MarkUnrollInfo and UnrollShortLoops — i.e. no other
    pass runs in between.
    """

    node_id: int
    name: str
    n: int
    depth: int  # 1 for outermost IRLoop, increases with nesting
    body_nonloop_insts: int  # inst count of body excluding direct child IRLoops
    direct_loop_children: List[int] = field(default_factory=list)  # ids of IRLoops directly nested in body
    descendant_loops: List[int] = field(default_factory=list)  # transitively nested IRLoop ids
    counter_referenced: bool = False
    over_iter_limit: bool = False


class MarkUnrollInfo(Pass):
    """Collect IRLoop metadata into ``ctx.unroll_info`` for budget-aware unroll.

    Does not modify the tree — overrides ``__call__`` to walk read-only and
    return the same root, preserving ``id(node)`` for use as dict keys in the
    subsequent ``UnrollShortLoops`` pass.
    """

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:  # pragma: no cover - unused
        return node

    def __call__(self, node: IRNode, ctx: PassCtx | None = None) -> IRNode:
        if ctx is None:
            ctx = PassCtx()
        ctx.unroll_info.clear()
        self._walk(node, ctx, depth=0)
        return node

    def _walk(self, node: IRNode, ctx: PassCtx, depth: int) -> None:
        if isinstance(node, IRLoop):
            self._record_loop(node, ctx, depth + 1)
            self._walk(node.body, ctx, depth + 1)
            return
        if isinstance(node, IRRegLoop):
            self._walk(node.body, ctx, depth + 1)
            return
        if isinstance(node, IRSeq):
            for child in node.body:
                self._walk(child, ctx, depth)
            return
        if isinstance(node, IRBranch):
            for arm in node.arms:
                self._walk(arm, ctx, depth)
            return
        # leaves: nothing to record

    def _record_loop(self, node: IRLoop, ctx: PassCtx, depth: int) -> None:
        direct_children: List[int] = []
        descendants: List[int] = []
        body_nonloop = self._body_nonloop_insts(node.body, direct_children, descendants)
        max_iters = ctx.config.max_unroll_iters
        info = UnrollInfo(
            node_id=id(node),
            name=node.name,
            n=node.n,
            depth=depth,
            body_nonloop_insts=body_nonloop,
            direct_loop_children=direct_children,
            descendant_loops=descendants,
            counter_referenced=_references_reg(node.body, node.name),
            over_iter_limit=node.n > max_iters,
        )
        ctx.unroll_info[info.node_id] = info

    @classmethod
    def _body_nonloop_insts(
        cls,
        node: IRNode,
        direct_children: List[int],
        descendants: List[int],
    ) -> int:
        """Sum inst estimate of body, treating direct IRLoop children as 0.

        IRLoop children's contribution to parent body is computed dynamically
        in UnrollShortLoops based on whether they are chosen for unrolling.
        Descendants are still collected for transitive lookup.
        """
        if isinstance(node, IRLoop):
            direct_children.append(id(node))
            descendants.append(id(node))
            descendants.extend(cls._collect_descendant_loops(node.body))
            return 0
        if isinstance(node, IRSeq):
            return sum(
                cls._body_nonloop_insts(child, direct_children, descendants)
                for child in node.body
            )
        if isinstance(node, IRBranch):
            arms_total = sum(
                cls._body_nonloop_insts(arm, direct_children, descendants)
                for arm in node.arms
            )
            return arms_total + 2 * max(0, len(node.arms) - 1)
        # IRRegLoop and other composites/leaves: use estimate_insts directly,
        # and still collect any nested IRLoops as descendants.
        descendants.extend(cls._collect_descendant_loops(node))
        return estimate_insts(node)

    @classmethod
    def _collect_descendant_loops(cls, node: IRNode) -> List[int]:
        out: List[int] = []
        if isinstance(node, IRLoop):
            out.append(id(node))
            out.extend(cls._collect_descendant_loops(node.body))
            return out
        if isinstance(node, IRRegLoop):
            out.extend(cls._collect_descendant_loops(node.body))
            return out
        if isinstance(node, IRSeq):
            for child in node.body:
                out.extend(cls._collect_descendant_loops(child))
            return out
        if isinstance(node, IRBranch):
            for arm in node.arms:
                out.extend(cls._collect_descendant_loops(arm))
            return out
        return out


class UnrollShortLoops(Pass):
    """Budget-aware greedy IRLoop unrolling.

    Reads ``ctx.unroll_info`` populated by ``MarkUnrollInfo``. Candidates are
    sorted by ``(-depth, body_insts_baseline, n)`` and unrolled greedily as
    long as ``ctx.pmem_used`` stays within ``config.pmem_budget``. ``pmem_used``
    starts at the baseline (no-unroll) inst estimate of the entire tree.

    Counter-referenced loops and loops with ``n > max_unroll_iters`` are
    skipped (with diagnostic warnings). Loops with ``n < 2`` cannot save any
    instructions and are not unrolled either (preserved as-is).
    """

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:  # pragma: no cover - unused
        return node

    def __call__(self, node: IRNode, ctx: PassCtx | None = None) -> IRNode:
        if ctx is None:
            ctx = PassCtx()
        budget = ctx.config.pmem_budget
        if budget <= 0:
            raise ValueError(
                f"UnrollShortLoops requires positive pmem_budget, got {budget}"
            )
        if not ctx.unroll_info:
            # No loops to consider — nothing to do.
            ctx.pmem_used = estimate_insts(node)
            return node

        chosen = self._select_candidates(ctx, node, budget)
        return self._rewrite(node, chosen)

    def _select_candidates(self, ctx: PassCtx, root: IRNode, budget: int) -> Set[int]:
        infos: Dict[int, UnrollInfo] = ctx.unroll_info
        empty_chosen: Set[int] = set()
        candidates: List[UnrollInfo] = []
        for info in infos.values():
            if info.counter_referenced:
                ctx.warn(
                    f"skip unroll loop '{info.name}': body references counter register"
                )
                continue
            if info.over_iter_limit:
                ctx.warn(
                    f"skip unroll loop '{info.name}': n={info.n} exceeds "
                    f"max_unroll_iters={ctx.config.max_unroll_iters}"
                )
                continue
            if info.n < 2:
                continue
            candidates.append(info)

        # Sort: inner first (depth desc), small baseline body first, small n first.
        candidates.sort(
            key=lambda info: (
                -info.depth,
                self._effective_body_insts(info, infos, empty_chosen),
                info.n,
            )
        )

        chosen: Set[int] = set()
        pmem_used = estimate_insts(root)  # baseline: nothing unrolled
        for info in candidates:
            eff_body = self._effective_body_insts(info, infos, chosen)
            # Keep cost: 4 (open+close) + eff_body. Unroll cost: n * eff_body.
            delta = (info.n - 1) * eff_body - 4
            if delta <= 0:
                chosen.add(info.node_id)
                pmem_used += delta
                continue
            if pmem_used + delta <= budget:
                chosen.add(info.node_id)
                pmem_used += delta

        ctx.pmem_used = pmem_used
        return chosen

    @classmethod
    def _effective_body_insts(
        cls,
        info: UnrollInfo,
        infos: Dict[int, UnrollInfo],
        chosen: Set[int],
    ) -> int:
        """Compute body inst count given current chosen-set decisions.

        Direct IRLoop children contribute either ``n * eff(child)`` if chosen,
        or ``4 + eff(child)`` if preserved.
        """
        total = info.body_nonloop_insts
        for child_id in info.direct_loop_children:
            child = infos[child_id]
            child_eff = cls._effective_body_insts(child, infos, chosen)
            if child_id in chosen:
                total += child.n * child_eff
            else:
                total += 4 + child_eff
        return total

    def _rewrite(self, node: IRNode, chosen: Set[int]) -> IRNode:
        """Bottom-up rebuild, expanding only IRLoops whose original id is chosen."""
        if isinstance(node, IRLoop):
            should_unroll = id(node) in chosen
            new_body = self._rewrite(node.body, chosen)
            if should_unroll:
                if node.n <= 0:
                    return IRSeq(body=(), meta=node.meta)
                if node.n == 1:
                    return new_body
                return IRSeq(body=tuple(new_body for _ in range(node.n)), meta=node.meta)
            return IRLoop(name=node.name, n=node.n, body=new_body, meta=node.meta)
        if isinstance(node, IRRegLoop):
            new_body = self._rewrite(node.body, chosen)
            return IRRegLoop(name=node.name, n_reg=node.n_reg, body=new_body, meta=node.meta)
        if isinstance(node, IRSeq):
            new_children = tuple(self._rewrite(child, chosen) for child in node.body)
            return IRSeq(body=new_children, meta=node.meta)
        if isinstance(node, IRBranch):
            new_arms = tuple(self._rewrite(arm, chosen) for arm in node.arms)
            return IRBranch(compare_reg=node.compare_reg, arms=new_arms, meta=node.meta)
        return node


def _references_reg(node: IRNode, reg_name: str) -> bool:
    if isinstance(node, IRSeq):
        return any(_references_reg(child, reg_name) for child in node.body)
    if isinstance(node, IRLoop):
        return _references_reg(node.body, reg_name)
    if isinstance(node, IRRegLoop):
        return node.n_reg == reg_name or _references_reg(node.body, reg_name)
    if isinstance(node, IRBranch):
        return any(_references_reg(arm, reg_name) for arm in node.arms)
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
            MarkUnrollInfo(),
            UnrollShortLoops(),
            FlattenSeq(),
            FuseAdjacentDelays(),
            RemoveZeroDelays(),
            ReorderPulseLikeByTime(),
            ValidateInvariants(),
        ],
        config=config,
    )
