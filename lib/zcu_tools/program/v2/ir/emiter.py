"""Emitter: converts IR tree back to QICK macros."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator
from contextlib import contextmanager

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
    IRPulseByReg,
    IRReadDmem,
    IRReadout,
    IRRegLoop,
    IRRegOp,
    IRSendReadoutConfig,
    IRSeq,
)

if TYPE_CHECKING:
    from ..modular import ModularProgramV2


class Emitter:
    """Converts IR nodes back to QICK macros."""

    def __init__(self, prog: ModularProgramV2) -> None:
        self.prog = prog
        self.branch_counter = 0
        self.reg_stack = [0]
        self.temp_reg_num = 0

    def emit(self, node: IRNode) -> None:
        """Emit an IR node as QICK macros (no t threading — nodes carry t)."""
        if isinstance(node, IRPulse):
            self._emit_pulse(node)
        elif isinstance(node, IRReadout):
            self._emit_readout(node)
        elif isinstance(node, IRPulseByReg):
            self._emit_pulse_by_reg(node)
        elif isinstance(node, IRSendReadoutConfig):
            self._emit_send_readoutconfig(node)
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
            self.prog.cond_jump(
                node.target, node.arg1, node.test, op=node.op, arg2=node.arg2
            )
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
        self.prog.pulse(node.ch, node.pulse_id, t=node.t)  # type: ignore[arg-type]

    def _emit_readout(self, node: IRReadout) -> None:
        self.prog.trigger(ros=node.ro_chs, t=node.t)  # type: ignore[arg-type]

    def _emit_pulse_by_reg(self, node: IRPulseByReg) -> None:
        addr_reg1 = node.addr_reg
        if node.flat_top_pulse:
            with self.acquire_temp_reg(2) as (addr_reg2, addr_reg3):
                self.prog.write_reg_op(addr_reg2, node.addr_reg, "+", 1)
                self.prog.write_reg_op(addr_reg3, node.addr_reg, "+", 2)
                self.prog.pulse_by_reg(
                    node.ch, [addr_reg1, addr_reg2, addr_reg3], t=node.t
                )
        else:
            self.prog.pulse_by_reg(node.ch, [addr_reg1], t=node.t)

    def _emit_send_readoutconfig(self, node: IRSendReadoutConfig) -> None:
        self.prog.send_readoutconfig(node.ch, node.readout_id, t=node.t)  # type: ignore[arg-type]

    def _emit_delay(self, node: IRDelay) -> None:
        self.prog.delay(t=node.t)

    def _emit_delay_auto(self, node: IRDelayAuto) -> None:
        if isinstance(node.t, str):
            self.prog.delay_auto_by_reg(time_reg=node.t, gens=node.gens, ros=node.ros)
        else:
            self.prog.delay_auto(t=node.t, gens=node.gens, ros=node.ros)

    def _emit_loop(self, node: IRLoop) -> None:
        self.prog.open_loop(n=node.n, name=node.name)
        self.emit(node.body)
        self.prog.close_loop()

    def _emit_reg_loop(self, node: IRRegLoop) -> None:
        self.prog.open_loop_reg(n_reg=node.n_reg, name=node.name)
        self.emit(node.body)
        self.prog.close_loop_reg(name=node.name)

    def _emit_branch(self, node: IRBranch) -> None:
        if len(node.arms) < 2:
            raise ValueError("IRBranch requires at least 2 arms")

        branch_id = self.branch_counter
        self.branch_counter += 1

        def emit_dispatch(lo: int, hi: int) -> None:
            if hi - lo == 1:
                self.emit(node.arms[lo])
                return

            mid = (lo + hi) // 2
            left_label = f"irb{branch_id}_l_{lo}_{mid}"
            end_label = f"irb{branch_id}_e_{lo}_{hi}"

            self.prog.cond_jump(
                left_label,
                node.compare_reg,
                "S",
                op="-",
                arg2=mid,
            )
            emit_dispatch(mid, hi)
            self.prog.jump(end_label)
            self.prog.label(left_label)
            emit_dispatch(lo, mid)
            self.prog.label(end_label)

        emit_dispatch(0, len(node.arms))

    @contextmanager
    def acquire_temp_reg(self, num: int = 1) -> Generator[list[str], None, None]:
        if num < 0:
            raise ValueError(f"num must be >= 0, got {num}")
        if num == 0:
            yield []
            return

        used = self.reg_stack[-1]
        total = used + num
        if total > self.temp_reg_num:
            self.temp_reg_num = total

        self.reg_stack.append(total)
        try:
            yield [f"temp_reg_{i}" for i in range(used, total)]
        finally:
            if len(self.reg_stack) <= 1:
                raise RuntimeError("IRBuilder: temp_reg scope stack underflow")
            self.reg_stack.pop()
