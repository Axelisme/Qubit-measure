"""Core IR passes for optimization and validation."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, cast

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
    IRReadDmem,
    IRReadout,
    IRRegLoop,
    IRRegOp,
    IRSeq,
)
from .pass_base import Pass, PassCtx


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
                name=node.name, n=node.n, body=node.body, meta=self._with_dur(node.meta, dur2)
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
            return IRReadDmem(dst=node.dst, addr=node.addr, meta=self._with_dur(node.meta, 0.0))
        elif isinstance(node, IRNop):
            return IRNop(meta=self._with_dur(node.meta, 0.0))
        else:
            return node
