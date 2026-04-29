"""LowerCtx and Emitter for IR-based module lowering."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

from .ir import IRNode, IRPulse, IRReadout, IRDelay, IRLabel, IRNop, IRRegOp, IRCondJump, IRJump, IRSeq, IRLoop, IRRegLoop, IRBranch

if TYPE_CHECKING:
    from .modular import ModularProgramV2

logger = logging.getLogger(__name__)


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

    def __init__(self, prog: ModularProgramV2) -> None:
        self.prog = prog

    def emit(self, node: IRNode) -> None:
        """Emit an IR node as QICK macros."""
        if isinstance(node, IRPulse):
            self._emit_pulse(node)
        elif isinstance(node, IRReadout):
            self._emit_readout(node)
        elif isinstance(node, IRDelay):
            self._emit_delay(node)
        elif isinstance(node, IRLabel):
            self._emit_label(node)
        elif isinstance(node, IRNop):
            self._emit_nop(node)
        elif isinstance(node, IRRegOp):
            self._emit_reg_op(node)
        elif isinstance(node, IRCondJump):
            self._emit_cond_jump(node)
        elif isinstance(node, IRJump):
            self._emit_jump(node)
        elif isinstance(node, IRSeq):
            self._emit_seq(node)
        elif isinstance(node, IRLoop):
            self._emit_loop(node)
        elif isinstance(node, IRRegLoop):
            self._emit_reg_loop(node)
        elif isinstance(node, IRBranch):
            self._emit_branch(node)
        else:
            raise TypeError(f"Unknown IR node type: {type(node)}")

    def _emit_pulse(self, node: IRPulse) -> None:
        """Emit a pulse node."""
        t = int(node.pre_delay) if isinstance(node.pre_delay, (int, float)) else 0
        self.prog.pulse(int(node.ch), node.pulse_name, t=t, tag=None)

    def _emit_readout(self, node: IRReadout) -> None:
        """Emit a readout node."""
        t = int(node.trig_offset) if isinstance(node.trig_offset, (int, float)) else 0
        ros_chs = [int(ch) for ch in node.ro_chs]
        self.prog.trigger(ros=ros_chs, t=t)

    def _emit_delay(self, node: IRDelay) -> None:
        """Emit a delay node."""
        if node.auto:
            if isinstance(node.duration, str):
                # Register-based delay
                self.prog.delay_reg_auto(time_reg=node.duration)
            else:
                # Regular auto delay
                self.prog.delay_auto(t=node.duration, tag=node.tag)
        else:
            # Regular delay
            delay_t = node.duration if isinstance(node.duration, (int, float)) else 0
            self.prog.delay(t=delay_t, tag=node.tag)

    def _emit_label(self, node: IRLabel) -> None:
        """Emit a label node."""
        self.prog.label(node.name)

    def _emit_nop(self, node: IRNop) -> None:
        """Emit a NOP node."""
        # NOP is typically just a label with no other effect; emit it as a label
        self.prog.label("nop")

    def _emit_reg_op(self, node: IRRegOp) -> None:
        """Emit a register operation node."""
        self.prog.write_reg_op(node.dst, node.lhs, node.op, node.rhs)

    def _emit_cond_jump(self, node: IRCondJump) -> None:
        """Emit a conditional jump node."""
        self.prog.cond_jump(node.target, node.arg1, node.test, op=node.op, arg2=node.arg2)

    def _emit_jump(self, node: IRJump) -> None:
        """Emit an unconditional jump node."""
        self.prog.jump(node.target)

    def _emit_seq(self, node: IRSeq) -> None:
        """Emit a sequence node (sequential composition)."""
        for child in node.body:
            self.emit(child)

    def _emit_loop(self, node: IRLoop) -> None:
        """Emit a loop node."""
        self.prog.open_loop(n=node.n, name=node.name)
        self.emit(node.body)
        self.prog.close_loop()

    def _emit_reg_loop(self, node: IRRegLoop) -> None:
        """Emit a register-driven loop node."""
        self.prog.open_loop_reg(n_reg=node.n_reg, name=node.name)
        self.emit(node.body)
        self.prog.close_loop_reg(name=node.name)

    def _emit_branch(self, node: IRBranch) -> None:
        """Emit a branch node (binary dispatch tree)."""
        # Build a binary tree of conditional branches
        # For N arms, we need N-1 comparisons
        # This is a simplified implementation; full version builds the tree structure
        for i, arm in enumerate(node.arms):
            if i < len(node.arms) - 1:
                # Conditional branch to next arm
                self.prog.cond_jump(
                    label=f"arm_{i}",
                    arg1=node.compare_reg,
                    test="==",
                    op=i,
                    arg2=i,
                )
            self.prog.label(f"arm_{i}")
            self.emit(arm)
