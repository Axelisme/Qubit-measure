"""Core IR passes for optimization and validation."""

from __future__ import annotations

from typing import Dict, Optional, cast
from dataclasses import replace
from .nodes import (
    IRNode, IRMeta, IRLabel, IRJump, IRCondJump, IRLoop, IRRegLoop, IRSeq, IRDelay,
    IRPulse, IRReadout, IRBranch, IRNop, IRRegOp, IRReadDmem, IRSoftDelay, IRParallel
)
from .pass_base import Pass, PassCtx


class FreshLabels(Pass):
    """Rename all labels to avoid collisions from structural duplication.

    When a subtree is cloned (e.g., in UnrollShortLoops or SoftRepeat
    lowering), label names must be fresh to avoid jump target ambiguity.

    This pass must run FIRST in the pipeline.
    """

    def __init__(self):
        self._label_map: Dict[str, str] = {}
        self._counter: int = 0
        self._needs_reset: bool = True

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        """Rename label, jump, and condjump nodes using a fresh name counter."""
        # Reset mapping at root (top-level call)
        if self._needs_reset:
            self._label_map.clear()
            self._counter = 0
            self._needs_reset = False

        if isinstance(node, IRLabel):
            if node.name not in self._label_map:
                self._label_map[node.name] = f"_label_{self._counter}"
                self._counter += 1
            fresh_name = self._label_map[node.name]
            return IRLabel(name=fresh_name, meta=node.meta)

        elif isinstance(node, IRJump):
            old_target = node.target
            if old_target not in self._label_map:
                self._label_map[old_target] = f"_label_{self._counter}"
                self._counter += 1
            return IRJump(target=self._label_map[old_target], meta=node.meta)

        elif isinstance(node, IRCondJump):
            old_target = node.target
            if old_target not in self._label_map:
                self._label_map[old_target] = f"_label_{self._counter}"
                self._counter += 1
            return IRCondJump(
                target=self._label_map[old_target],
                arg1=node.arg1,
                test=node.test,
                op=node.op,
                arg2=node.arg2,
                meta=node.meta
            )

        else:
            return node

    def __call__(self, node: IRNode, ctx: PassCtx | None = None) -> IRNode:
        """Override to reset state before each top-level call."""
        self._needs_reset = True
        return super().__call__(node, ctx)


class EstimateDurations(Pass):
    """Estimate IR subtree duration via bottom-up walk.

    Sets `node.meta.duration` for all nodes. Nodes with QickParam
    expressions leave duration as None.

    Duration semantics:
    - IRPulse: pre_delay + pulse_length + post_delay
    - IRReadout: similar to pulse
    - IRDelay: duration value (or expression)
    - IRSeq: sum of body durations
    - IRLoop: n * body.duration (or None if body has None)
    - Composite nodes propagate None upward if any child has None
    """

    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        """Compute duration for this node based on children."""
        if isinstance(node, IRPulse):
            # Pulse duration = pre_delay + pulse + post_delay
            # Note: actual pulse length would come from pulse registry (not in IR)
            # For now, we estimate as pre + post (actual length added by emitter)
            dur = node.pre_delay + node.post_delay
            meta_with_dur = self._update_meta_duration(node.meta, dur)
            return IRPulse(
                ch=node.ch,
                pulse_name=node.pulse_name,
                pre_delay=node.pre_delay,
                post_delay=node.post_delay,
                advance=node.advance,
                tag=node.tag,
                meta=meta_with_dur
            )

        elif isinstance(node, IRReadout):
            if isinstance(node.trig_offset, (int, float)):
                dur = float(node.trig_offset)
            else:
                dur = None
            meta_with_dur = self._update_meta_duration(node.meta, dur)
            return IRReadout(
                ch=node.ch,
                ro_chs=node.ro_chs,
                pulse_name=node.pulse_name,
                trig_offset=node.trig_offset,
                meta=meta_with_dur
            )

        elif isinstance(node, IRDelay):
            # Delay duration is the duration value if numeric
            if isinstance(node.duration, (int, float)):
                dur: Optional[float] = float(node.duration)
            else:
                # QickParam expression or string — duration unknown
                dur = None
            meta_with_dur = self._update_meta_duration(node.meta, dur)
            return IRDelay(
                duration=node.duration,
                auto=node.auto,
                gens=node.gens,
                ros=node.ros,
                tag=node.tag,
                meta=meta_with_dur
            )

        elif isinstance(node, IRSoftDelay):
            if isinstance(node.duration, (int, float)):
                dur = float(node.duration)
            else:
                dur = None
            meta_with_dur = self._update_meta_duration(node.meta, dur)
            return IRSoftDelay(duration=node.duration, meta=meta_with_dur)

        elif isinstance(node, IRSeq):
            # Sum durations; None if any child is None
            dur = 0.0
            for child in node.body:
                child_dur = child.meta.duration
                if child_dur is None:
                    dur = None
                    break
                dur += child_dur
            meta_with_dur = self._update_meta_duration(node.meta, dur)
            return IRSeq(body=node.body, meta=meta_with_dur)

        elif isinstance(node, IRLoop):
            # Loop duration = n * body.duration
            body_dur = node.body.meta.duration
            if body_dur is None:
                dur = None
            else:
                dur = node.n * body_dur
            meta_with_dur = self._update_meta_duration(node.meta, dur)
            return IRLoop(name=node.name, n=node.n, body=node.body, meta=meta_with_dur)

        elif isinstance(node, IRRegLoop):
            # Register-driven loop: duration unknown (depends on runtime register)
            meta_with_dur = self._update_meta_duration(node.meta, None)
            return IRRegLoop(
                name=node.name,
                n_reg=node.n_reg,
                body=node.body,
                meta=meta_with_dur
            )

        elif isinstance(node, IRBranch):
            # Branch duration = max(arm durations), None if any arm is None
            if not node.arms:
                dur = 0.0
            else:
                arm_durs = [arm.meta.duration for arm in node.arms]
                if any(d is None for d in arm_durs):
                    dur = None
                else:
                    dur = max(cast(list[float], arm_durs))
            meta_with_dur = self._update_meta_duration(node.meta, dur)
            return IRBranch(compare_reg=node.compare_reg, arms=node.arms, meta=meta_with_dur)

        elif isinstance(node, IRParallel):
            if not node.body:
                dur = 0.0
            else:
                child_durs = [child.meta.duration for child in node.body]
                if any(d is None for d in child_durs):
                    dur = None
                elif node.end_policy == "index":
                    dur = cast(list[float], child_durs)[node.end_index]
                else:
                    numeric_durs = cast(list[float], child_durs)
                    dur = max(numeric_durs) if numeric_durs else 0.0
            meta_with_dur = self._update_meta_duration(node.meta, dur)
            return IRParallel(
                body=node.body,
                end_policy=node.end_policy,
                end_index=node.end_index,
                meta=meta_with_dur,
            )

        else:
            # Leaf nodes without explicit duration (Label, Jump, CondJump, etc.)
            # These contribute 0 to durations
            meta_with_dur = self._update_meta_duration(node.meta, 0.0)
            # Need to reconstruct node with new meta based on type
            if isinstance(node, IRLabel):
                return IRLabel(name=node.name, meta=meta_with_dur)
            elif isinstance(node, IRJump):
                return IRJump(target=node.target, meta=meta_with_dur)
            elif isinstance(node, IRCondJump):
                return IRCondJump(
                    target=node.target,
                    arg1=node.arg1,
                    test=node.test,
                    op=node.op,
                    arg2=node.arg2,
                    meta=meta_with_dur
                )
            else:
                # Other leaf nodes (should have handled all types above)
                return node

    @staticmethod
    def _update_meta_duration(meta: IRMeta, dur: Optional[float]) -> IRMeta:
        """Create new metadata with updated duration."""
        return replace(meta, duration=dur)
