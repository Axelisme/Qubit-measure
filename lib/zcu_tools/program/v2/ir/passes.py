"""Core IR passes for optimization and validation."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Union, cast

from .nodes import (
    IRBranch,
    IRCondJump,
    IRDelay,
    IRDelayAuto,
    IRJump,
    IRLabel,
    IRLoop,
    IRMeta,
    IRNode,
    IRNop,
    IRPulse,
    IRPulseWmemReg,
    IRReadDmem,
    IRReadout,
    IRSendReadoutConfig,
    IRRegLoop,
    IRRegOp,
    IRSeq,
)
from .pass_base import Pass, PassConfig, PassCtx, PassPipeline


class FreshLabels(Pass):
    """Rename all labels to avoid collisions from structural duplication.

    Must run FIRST in the pipeline.
    """

    def __init__(self) -> None:
        self._label_map: Dict[str, str] = {}
        self._counter: int = 0
        self._needs_reset: bool = True

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if self._needs_reset:
            self._label_map.clear()
            self._counter = 0
            self._needs_reset = False

        if isinstance(node, IRLabel):
            if node.name not in self._label_map:
                self._label_map[node.name] = f"_label_{self._counter}"
                self._counter += 1
            return IRLabel(name=self._label_map[node.name], meta=node.meta)

        elif isinstance(node, IRJump):
            if node.target not in self._label_map:
                self._label_map[node.target] = f"_label_{self._counter}"
                self._counter += 1
            return IRJump(target=self._label_map[node.target], meta=node.meta)

        elif isinstance(node, IRCondJump):
            if node.target not in self._label_map:
                self._label_map[node.target] = f"_label_{self._counter}"
                self._counter += 1
            return IRCondJump(
                target=self._label_map[node.target],
                arg1=node.arg1,
                test=node.test,
                op=node.op,
                arg2=node.arg2,
                meta=node.meta,
            )

        else:
            return node

    def __call__(self, node: IRNode, ctx: PassCtx | None = None) -> IRNode:
        self._needs_reset = True
        return super().__call__(node, ctx)


class EstimateDurations(Pass):
    """Estimate IR subtree duration via bottom-up walk.

    Duration semantics:
    - IRPulse: 0 (no timeline advance — only IRDelay advances ref_t)
    - IRDelay: t (numeric) or None (QickParam)
    - IRDelayAuto: None (hardware alignment — unknown statically)
    - IRSeq: sum of body durations; None if any child is None
    - IRLoop: n * body.duration; None if body has None
    - IRBranch: max(arm durations); None if any arm is None
    """

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if isinstance(node, IRPulse):
            # Pulses don't advance ref_t; they contribute 0 to timeline
            return IRPulse(
                ch=node.ch,
                pulse_name=node.pulse_name,
                t=node.t,
                tag=node.tag,
                meta=self._with_dur(node.meta, 0.0),
            )

        elif isinstance(node, IRPulseWmemReg):
            return IRPulseWmemReg(
                ch=node.ch,
                addr_reg=node.addr_reg,
                t=node.t,
                flat_top_pulse=node.flat_top_pulse,
                meta=self._with_dur(node.meta, 0.0),
            )

        elif isinstance(node, IRSendReadoutConfig):
            return IRSendReadoutConfig(
                ch=node.ch,
                pulse_name=node.pulse_name,
                t=node.t,
                meta=self._with_dur(node.meta, 0.0),
            )

        elif isinstance(node, IRReadout):
            return IRReadout(
                ch=node.ch,
                ro_chs=node.ro_chs,
                pulse_name=node.pulse_name,
                t=node.t,
                meta=self._with_dur(node.meta, 0.0),
            )

        elif isinstance(node, IRDelay):
            dur: Optional[float] = (
                float(node.t) if isinstance(node.t, (int, float)) else None
            )
            return IRDelay(t=node.t, tag=node.tag, meta=self._with_dur(node.meta, dur))

        elif isinstance(node, IRDelayAuto):
            # Hardware alignment: static duration unknown
            return IRDelayAuto(
                t=node.t,
                gens=node.gens,
                ros=node.ros,
                tag=node.tag,
                meta=self._with_dur(node.meta, None),
            )

        elif isinstance(node, IRSeq):
            total: Optional[float] = 0.0
            for child in node.body:
                if total is None:
                    break
                child_dur = child.meta.duration
                if child_dur is None:
                    total = None
                else:
                    total += child_dur
            return IRSeq(body=node.body, meta=self._with_dur(node.meta, total))

        elif isinstance(node, IRLoop):
            body_dur = node.body.meta.duration
            dur2 = None if body_dur is None else node.n * body_dur
            return IRLoop(
                name=node.name,
                n=node.n,
                body=node.body,
                meta=self._with_dur(node.meta, dur2),
            )

        elif isinstance(node, IRRegLoop):
            return IRRegLoop(
                name=node.name,
                n_reg=node.n_reg,
                body=node.body,
                meta=self._with_dur(node.meta, None),
            )

        elif isinstance(node, IRBranch):
            if not node.arms:
                dur3: Optional[float] = 0.0
            else:
                arm_durs = [arm.meta.duration for arm in node.arms]
                if any(d is None for d in arm_durs):
                    dur3 = None
                else:
                    dur3 = max(cast(list[float], arm_durs))
            return IRBranch(
                compare_reg=node.compare_reg,
                arms=node.arms,
                meta=self._with_dur(node.meta, dur3),
            )

        else:
            # Label, Jump, CondJump, RegOp, ReadDmem, Nop — contribute 0
            return self._rebuild_zero_dur(node)

    @staticmethod
    def _with_dur(meta: IRMeta, dur: Optional[float]) -> IRMeta:
        return replace(meta, duration=dur)

    def _rebuild_zero_dur(self, node: IRNode) -> IRNode:
        if isinstance(node, IRLabel):
            return IRLabel(name=node.name, meta=self._with_dur(node.meta, 0.0))
        elif isinstance(node, IRJump):
            return IRJump(target=node.target, meta=self._with_dur(node.meta, 0.0))
        elif isinstance(node, IRCondJump):
            return IRCondJump(
                target=node.target,
                arg1=node.arg1,
                test=node.test,
                op=node.op,
                arg2=node.arg2,
                meta=self._with_dur(node.meta, 0.0),
            )
        elif isinstance(node, IRRegOp):
            return IRRegOp(
                dst=node.dst,
                lhs=node.lhs,
                op=node.op,
                rhs=node.rhs,
                meta=self._with_dur(node.meta, 0.0),
            )
        elif isinstance(node, IRReadDmem):
            return IRReadDmem(
                dst=node.dst, addr=node.addr, meta=self._with_dur(node.meta, 0.0)
            )
        elif isinstance(node, IRNop):
            return IRNop(meta=self._with_dur(node.meta, 0.0))
        else:
            return node


class UnrollShortLoops(Pass):
    """Unroll small-count loops with short known body duration."""

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if not isinstance(node, IRLoop):
            return node

        if self._references_reg(node.body, node.name):
            ctx.warn(
                f"skip unroll loop '{node.name}': body references loop counter register"
            )
            return node

        body_dur = node.body.meta.duration
        if body_dur is None:
            return node
        if body_dur > ctx.config.min_body_us:
            return node

        max_unroll_iters = int(ctx.config.extra.get("max_unroll_iters", 16))
        if node.n > max_unroll_iters:
            ctx.warn(
                f"skip unroll loop '{node.name}': n={node.n} exceeds max_unroll_iters={max_unroll_iters}"
            )
            return node

        if node.n <= 0:
            return IRSeq(body=(), meta=node.meta)
        if node.n == 1:
            return node.body

        unrolled: List[IRNode] = []
        for _ in range(node.n):
            unrolled.append(node.body)
        return IRSeq(body=tuple(unrolled), meta=node.meta)

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
    """Fuse adjacent numeric IRDelay nodes to reduce macro count."""

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if not isinstance(node, IRSeq):
            return node
        if not ctx.config.enable_fusion:
            return node

        fused: List[IRNode] = []
        linear_body: List[IRNode] = []
        for child in node.body:
            if isinstance(child, IRSeq):
                linear_body.extend(child.body)
            else:
                linear_body.append(child)

        for child in linear_body:
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


class AlignBranchDispatch(Pass):
    """Pad branch arms with IRNop so arm dispatch paths are balanced."""

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        if not isinstance(node, IRBranch):
            return node
        if not ctx.config.enable_align_branches:
            return node
        if len(node.arms) < 2:
            return node

        arm_costs = [self._inst_cost(arm) for arm in node.arms]
        if any(cost is None for cost in arm_costs):
            ctx.warn("skip branch alignment: non-static arm cost detected")
            return node

        known_costs = cast(List[int], arm_costs)
        max_cost = max(known_costs)
        new_arms: List[IRNode] = []
        for arm, cost in zip(node.arms, known_costs):
            pad = max_cost - cost
            if pad <= 0:
                new_arms.append(arm)
                continue
            arm_seq = arm if isinstance(arm, IRSeq) else IRSeq(body=(arm,))
            padded_body = arm_seq.body + tuple(IRNop() for _ in range(pad))
            new_arms.append(IRSeq(body=padded_body, meta=arm_seq.meta))
        return IRBranch(compare_reg=node.compare_reg, arms=tuple(new_arms), meta=node.meta)

    def _inst_cost(self, node: IRNode) -> Optional[int]:
        if isinstance(node, IRSeq):
            total = 0
            for child in node.body:
                c = self._inst_cost(child)
                if c is None:
                    return None
                total += c
            return total
        if isinstance(node, IRLoop):
            body = self._inst_cost(node.body)
            return None if body is None else 2 + body * node.n
        if isinstance(node, IRRegLoop):
            return None
        if isinstance(node, IRBranch):
            arm_costs = [self._inst_cost(arm) for arm in node.arms]
            if any(c is None for c in arm_costs):
                return None
            return 1 + max(cast(List[int], arm_costs))
        return 1


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
        if isinstance(node, IRDelayAuto) and isinstance(node.t, str) and node.tag is not None:
            ctx.error("IRDelayAuto tag is invalid when t is register name")
        return node


def make_default_pipeline(config: PassConfig | None = None) -> PassPipeline:
    """Build the default optimization + validation pipeline."""
    return PassPipeline(
        passes=[
            FreshLabels(),
            EstimateDurations(),
            UnrollShortLoops(),
            FuseAdjacentDelays(),
            ReorderPulseLikeByTime(),
            AlignBranchDispatch(),
            ValidateInvariants(),
        ],
        config=config,
    )
