"""LowerCtx and Emitter for IR-based module lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from qick.asm_v2 import QickParam

from .ir import (
    IRBranch,
    IRCondJump,
    IRDelay,
    IRJump,
    IRLabel,
    IRLoop,
    IRNode,
    IRNop,
    IRParallel,
    IRPulse,
    IRReadDmem,
    IRReadout,
    IRRegLoop,
    IRRegOp,
    IRSeq,
    IRSoftDelay,
)
from .modules.util import merge_max_length

if TYPE_CHECKING:
    from .modular import ModularProgramV2

class NameAllocator:
    """Allocates fresh label and register names to avoid collisions."""

    def __init__(self) -> None:
        self._label_counter = 0
        self._reg_counter = 0

    def fresh_label(self, prefix: str = "label") -> str:
        """Generate a fresh label name."""
        name = f"{prefix}_{self._label_counter}"
        self._label_counter += 1
        return name

    def fresh_reg(self, prefix: str = "tmp_reg") -> str:
        """Generate a fresh register name."""
        name = f"{prefix}_{self._reg_counter}"
        self._reg_counter += 1
        return name


class LowerCtx:
    """Context passed to Module.lower() to provide IR generation support."""

    def __init__(
        self,
        prog: ModularProgramV2,
        name_alloc: NameAllocator | None = None,
        parent_path: Tuple[str, ...] | None = None,
    ) -> None:
        self.prog = prog
        self.name_alloc = name_alloc or NameAllocator()
        self.parent_path = parent_path or ()

    def with_child(self, child_name: str) -> LowerCtx:
        """Create a child context for a nested module."""
        return LowerCtx(
            prog=self.prog,
            name_alloc=self.name_alloc,
            parent_path=self.parent_path + (child_name,),
        )


class Emitter:
    """Converts IR nodes back to QICK macros."""

    def __init__(
        self, prog: ModularProgramV2, name_alloc: NameAllocator | None = None
    ) -> None:
        self.prog = prog
        self.name_alloc = name_alloc or NameAllocator()

    def emit(self, node: IRNode, t: float | QickParam = 0.0) -> float | QickParam:
        """Emit an IR node as QICK macros and return next module reference time."""
        if isinstance(node, IRPulse):
            return self._emit_pulse(node, t)
        elif isinstance(node, IRReadout):
            return self._emit_readout(node, t)
        elif isinstance(node, IRDelay):
            return self._emit_delay(node)
        elif isinstance(node, IRSoftDelay):
            return self._emit_soft_delay(node, t)
        elif isinstance(node, IRLabel):
            self._emit_label(node)
            return t
        elif isinstance(node, IRNop):
            self._emit_nop(node)
            return t
        elif isinstance(node, IRRegOp):
            self._emit_reg_op(node)
            return t
        elif isinstance(node, IRReadDmem):
            self._emit_read_dmem(node)
            return t
        elif isinstance(node, IRCondJump):
            self._emit_cond_jump(node)
            return t
        elif isinstance(node, IRJump):
            self._emit_jump(node)
            return t
        elif isinstance(node, IRSeq):
            return self._emit_seq(node, t)
        elif isinstance(node, IRLoop):
            return self._emit_loop(node, t)
        elif isinstance(node, IRRegLoop):
            return self._emit_reg_loop(node, t)
        elif isinstance(node, IRBranch):
            return self._emit_branch(node, t)
        elif isinstance(node, IRParallel):
            return self._emit_parallel(node, t)
        else:
            raise TypeError(f"Unknown IR node type: {type(node)}")

    def _emit_pulse(self, node: IRPulse, t: float | QickParam) -> float | QickParam:
        """Emit a pulse node.

        ``advance`` already encodes pre_delay + waveform length + post_delay,
        so we never reference post_delay here.
        """
        self.prog.pulse(int(node.ch), node.pulse_name, t=t + node.pre_delay, tag=node.tag)  # type: ignore
        return t + node.advance

    def _emit_readout(self, node: IRReadout, t: float | QickParam) -> float | QickParam:
        """Emit a readout node."""
        ros_chs = [int(ch) for ch in node.ro_chs]
        self.prog.send_readoutconfig(int(node.ch), node.pulse_name, t=t)  # type: ignore
        self.prog.trigger(ros=ros_chs, t=t + node.trig_offset)  # type: ignore
        return t

    def _emit_delay(self, node: IRDelay) -> float:
        """Emit a delay node.

        Always returns 0.0 to reset the module-level reference time, mirroring
        legacy ``Delay.run()`` / ``DelayAuto.run()``. Subsequent IR nodes in an
        IRSeq accumulate from 0.
        """
        if node.auto:
            if isinstance(node.duration, str):
                # Register-based delay
                self.prog.delay_reg_auto(
                    time_reg=node.duration, gens=node.gens, ros=node.ros
                )
            else:
                # Regular auto delay
                self.prog.delay_auto(
                    t=node.duration, gens=node.gens, ros=node.ros, tag=node.tag
                )
        else:
            if isinstance(node.duration, str):
                raise ValueError("IRDelay(auto=False) cannot use register duration")
            self.prog.delay(t=node.duration, tag=node.tag)
        return 0.0

    def _emit_soft_delay(
        self, node: IRSoftDelay, t: float | QickParam
    ) -> float | QickParam:
        """Emit a timeline-only delay node."""
        return t + node.duration

    def _emit_label(self, node: IRLabel) -> None:
        """Emit a label node."""
        self.prog.label(node.name)

    def _emit_nop(self, _node: IRNop) -> None:
        """Emit a NOP node."""
        self.prog.nop()

    def _emit_read_dmem(self, node: IRReadDmem) -> None:
        """Emit a DMEM read node."""
        self.prog.read_dmem(dst=node.dst, addr=node.addr)

    def _emit_reg_op(self, node: IRRegOp) -> None:
        """Emit a register operation node."""
        self.prog.write_reg_op(node.dst, node.lhs, node.op, node.rhs)

    def _emit_cond_jump(self, node: IRCondJump) -> None:
        """Emit a conditional jump node."""
        self.prog.cond_jump(node.target, node.arg1, node.test, op=node.op, arg2=node.arg2)

    def _emit_jump(self, node: IRJump) -> None:
        """Emit an unconditional jump node."""
        self.prog.jump(node.target)

    def _emit_seq(self, node: IRSeq, t: float | QickParam) -> float | QickParam:
        """Emit a sequence node (sequential composition)."""
        cur_t = t
        for child in node.body:
            cur_t = self.emit(child, cur_t)
        return cur_t

    def _emit_loop(self, node: IRLoop, t: float | QickParam) -> float | QickParam:
        """Emit a loop node."""
        self.prog.open_loop(n=node.n, name=node.name)
        self.emit(node.body, t=0.0)
        self.prog.close_loop()
        return t

    def _emit_reg_loop(
        self, node: IRRegLoop, t: float | QickParam
    ) -> float | QickParam:
        """Emit a register-driven loop node."""
        self.prog.open_loop_reg(n_reg=node.n_reg, name=node.name)
        self.emit(node.body, t=0.0)
        self.prog.close_loop_reg(name=node.name)
        return t

    def _emit_branch(
        self, _node: IRBranch, _t: float | QickParam
    ) -> float | QickParam:
        """Emit a branch node (binary dispatch tree).

        Not implemented in Phase 1 — control-flow modules land in Phase 2.
        """
        raise NotImplementedError(
            "IRBranch emission is deferred to Phase 2 (control-flow modules)"
        )

    def _emit_parallel(
        self, node: IRParallel, t: float | QickParam
    ) -> float | QickParam:
        """Emit children from same start-t and merge end-t by policy.

        Mirrors legacy parallel-block scheduling: each child starts at ``t``,
        ``end_policy="max"`` returns the longest end-time via merge_max_length
        (matching Join/PulseReadout/TwoPulseReset semantics), and
        ``end_policy="index"`` picks one branch's end-time (used by BathReset
        to chain on the cavity tone).
        """
        if node.disable_delay:
            with self.prog.disable_delay():
                child_ends = [self.emit(child, t=t) for child in node.body]
        else:
            child_ends = [self.emit(child, t=t) for child in node.body]
        if not child_ends:
            return t

        if node.end_policy == "index":
            return child_ends[node.end_index]

        return merge_max_length(*child_ends)
