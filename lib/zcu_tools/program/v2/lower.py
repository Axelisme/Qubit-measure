"""Emitter: converts IR tree back to QICK macros."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ir import (
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
    IRReadDmem,
    IRReadout,
    IRRegLoop,
    IRRegOp,
    IRSeq,
)

if TYPE_CHECKING:
    from .modular import ModularProgramV2


class Emitter:
    """Converts IR nodes back to QICK macros.

    Each leaf node maps 1:1 to a single prog.* call.
    IRDelay / IRDelayAuto are the only nodes that reset/advance ref_t.
    """

    def __init__(self, prog: ModularProgramV2) -> None:
        self.prog = prog

    def emit(self, node: IRNode) -> None:
        """Emit an IR node as QICK macros (no t threading — nodes carry t)."""
        if isinstance(node, IRPulse):
            self._emit_pulse(node)
        elif isinstance(node, IRReadout):
            self._emit_readout(node)
        elif isinstance(node, IRDelay):
            self._emit_delay(node)
        elif isinstance(node, IRDelayAuto):
            self._emit_delay_auto(node)
        elif isinstance(node, IRLabel):
            self.prog.label(node.name)
        elif isinstance(node, IRNop):
            self.prog.nop()
        elif isinstance(node, IRRegOp):
            self.prog.write_reg_op(node.dst, node.lhs, node.op, node.rhs)
        elif isinstance(node, IRReadDmem):
            self.prog.read_dmem(dst=node.dst, addr=node.addr)
        elif isinstance(node, IRCondJump):
            self.prog.cond_jump(node.target, node.arg1, node.test, op=node.op, arg2=node.arg2)
        elif isinstance(node, IRJump):
            self.prog.jump(node.target)
        elif isinstance(node, IRSeq):
            for child in node.body:
                self.emit(child)
        elif isinstance(node, IRLoop):
            self._emit_loop(node)
        elif isinstance(node, IRRegLoop):
            self._emit_reg_loop(node)
        elif isinstance(node, IRBranch):
            self._emit_branch(node)
        else:
            raise TypeError(f"Unknown IR node type: {type(node)}")

    def _emit_pulse(self, node: IRPulse) -> None:
        self.prog.pulse(int(node.ch), node.pulse_name, t=node.t, tag=node.tag)  # type: ignore[arg-type]

    def _emit_readout(self, node: IRReadout) -> None:
        ro_ch = int(node.ch)
        # send_readoutconfig uses the pulse ch; trigger uses the ro_chs
        self.prog.send_readoutconfig(ro_ch, node.pulse_name, t=node.t)  # type: ignore[arg-type]
        ros = [int(ch) for ch in node.ro_chs]
        self.prog.trigger(ros=ros, t=node.t)  # type: ignore[arg-type]

    def _emit_delay(self, node: IRDelay) -> None:
        self.prog.delay(t=node.t, tag=node.tag)  # type: ignore[arg-type]

    def _emit_delay_auto(self, node: IRDelayAuto) -> None:
        if isinstance(node.t, str):
            self.prog.delay_reg_auto(time_reg=node.t, gens=node.gens, ros=node.ros)
        else:
            self.prog.delay_auto(t=node.t, gens=node.gens, ros=node.ros, tag=node.tag)  # type: ignore[arg-type]

    def _emit_loop(self, node: IRLoop) -> None:
        self.prog.open_loop(n=node.n, name=node.name)
        self.emit(node.body)
        self.prog.close_loop()

    def _emit_reg_loop(self, node: IRRegLoop) -> None:
        self.prog.open_loop_reg(n_reg=node.n_reg, name=node.name)
        self.emit(node.body)
        self.prog.close_loop_reg(name=node.name)

    def _emit_branch(self, _node: IRBranch) -> None:
        raise NotImplementedError(
            "IRBranch emission is deferred to Phase 2R (control-flow modules)"
        )
