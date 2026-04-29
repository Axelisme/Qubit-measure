"""IRBuilder: scope-stack based IR tree construction.

Module authors call builder.ir_*(…) methods to emit IR nodes. Context managers
ir_loop / ir_branch push a scope on entry and pop+wrap into IRLoop / IRBranch on
exit. The builder tracks no timing state — callers compute and pass t themselves.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple, Union

from qick.asm_v2 import QickParam

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
    IRRegLoop,
    IRRegOp,
    IRSendReadoutConfig,
    IRSeq,
    RegOp,
)


class IRBuilder:
    """Builds an IR tree via a scope stack.

    Usage::

        builder = IRBuilder()
        t = builder.ir_pulse("ch0", "my_pulse", t=0.0)
        with builder.ir_loop("loop0", n=100):
            builder.ir_delay(0.0)
            builder.ir_delay_auto(0.0)
            # ... body ...
            builder.ir_delay(body_t)
            builder.ir_delay_auto(0.0)
        root = builder.build()
    """

    def __init__(self) -> None:
        # Stack of pending node lists: each entry is the body list for the
        # current scope. Bottom of the stack is the top-level sequence.
        self._stack: List[List[IRNode]] = [[]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, node: IRNode) -> None:
        self._stack[-1].append(node)

    def _push(self) -> None:
        self._stack.append([])

    def _pop(self) -> Tuple[IRNode, ...]:
        nodes = self._stack.pop()
        return tuple(nodes)

    # ------------------------------------------------------------------
    # Leaf emitters
    # ------------------------------------------------------------------

    def ir_pulse(
        self,
        ch: str,
        pulse_name: str,
        t: Union[float, QickParam],
        tag: Optional[str] = None,
    ) -> None:
        self._emit(IRPulse(ch=ch, pulse_name=pulse_name, t=t, tag=tag))

    def ir_readout(
        self,
        ch: str,
        ro_chs: Tuple[str, ...],
        pulse_name: str,
        t: Union[float, QickParam],
    ) -> None:
        self._emit(IRReadout(ch=ch, ro_chs=ro_chs, pulse_name=pulse_name, t=t))

    def ir_send_readoutconfig(
        self,
        ch: str,
        pulse_name: str,
        t: Union[float, QickParam],
    ) -> None:
        self._emit(IRSendReadoutConfig(ch=ch, pulse_name=pulse_name, t=t))

    def ir_delay(
        self,
        t: Union[float, QickParam],
        tag: Optional[str] = None,
    ) -> None:
        self._emit(IRDelay(t=t, tag=tag))

    def ir_delay_auto(
        self,
        t: Union[float, QickParam, str] = 0.0,
        gens: bool = True,
        ros: bool = True,
        tag: Optional[str] = None,
    ) -> None:
        self._emit(IRDelayAuto(t=t, gens=gens, ros=ros, tag=tag))

    def ir_pulse_wmem_reg(
        self,
        ch: int,
        addr_reg: str,
        t: Union[float, QickParam] = 0.0,
        flat_top_pulse: bool = False,
    ) -> None:
        self._emit(
            IRPulseWmemReg(
                ch=ch,
                addr_reg=addr_reg,
                t=t,
                flat_top_pulse=flat_top_pulse,
            )
        )

    def ir_reg_op(
        self,
        dst: str,
        lhs: str,
        op: Union[RegOp, str],
        rhs: Union[int, str, None],
    ) -> None:
        reg_op = op if isinstance(op, RegOp) else RegOp(op)
        self._emit(IRRegOp(dst=dst, lhs=lhs, op=reg_op, rhs=rhs))

    def ir_read_dmem(self, dst: str, addr: str) -> None:
        self._emit(IRReadDmem(dst=dst, addr=addr))

    def ir_cond_jump(
        self,
        target: str,
        arg1: str,
        test: str,
        op: Optional[str] = None,
        arg2: Union[int, str, None] = None,
    ) -> None:
        self._emit(IRCondJump(target=target, arg1=arg1, test=test, op=op, arg2=arg2))

    def ir_jump(self, target: str) -> None:
        self._emit(IRJump(target=target))

    def ir_label(self, name: str) -> None:
        self._emit(IRLabel(name=name))

    def ir_nop(self) -> None:
        self._emit(IRNop())

    # ------------------------------------------------------------------
    # Control-flow context managers
    # ------------------------------------------------------------------

    @contextmanager
    def ir_loop(self, name: str, n: int) -> Generator[None, None, None]:
        """Context manager: body nodes become an IRLoop."""
        self._push()
        try:
            yield
        finally:
            body_nodes = self._pop()
            loop_body = (
                IRSeq(body=body_nodes) if len(body_nodes) != 1 else body_nodes[0]
            )
            self._emit(IRLoop(name=name, n=n, body=loop_body))

    @contextmanager
    def ir_reg_loop(self, name: str, n_reg: str) -> Generator[None, None, None]:
        """Context manager: body nodes become an IRRegLoop."""
        self._push()
        try:
            yield
        finally:
            body_nodes = self._pop()
            loop_body = (
                IRSeq(body=body_nodes) if len(body_nodes) != 1 else body_nodes[0]
            )
            self._emit(IRRegLoop(name=name, n_reg=n_reg, body=loop_body))

    @contextmanager
    def ir_branch(self, compare_reg: str) -> Generator[_BranchCtx, None, None]:
        """Context manager: arms become an IRBranch.

        Usage::

            with builder.ir_branch("my_reg") as branch:
                with branch.arm():
                    builder.ir_pulse(...)
                with branch.arm():
                    builder.ir_delay(...)
        """
        branch_ctx = _BranchCtx(self)
        try:
            yield branch_ctx
        finally:
            arms = branch_ctx.finish()
            self._emit(IRBranch(compare_reg=compare_reg, arms=arms))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> IRNode:
        """Return the top-level IR node (an IRSeq over all top-level emissions)."""
        if len(self._stack) != 1:
            raise RuntimeError(
                f"IRBuilder: {len(self._stack) - 1} unclosed scope(s) at build()"
            )
        nodes = self._stack[0]
        if len(nodes) == 0:
            return IRSeq()
        if len(nodes) == 1:
            return nodes[0]
        return IRSeq(body=tuple(nodes))

    def build_with_meta(self, source_module: str = "") -> IRNode:
        """Like build() but attaches source_module to the root IRSeq meta."""
        root = self.build()
        if isinstance(root, IRSeq):
            return IRSeq(body=root.body, meta=IRMeta(source_module=source_module))
        return root


class _BranchCtx:
    """Helper returned by ir_branch context manager to collect arms."""

    def __init__(self, builder: IRBuilder) -> None:
        self._builder = builder
        self._arms: List[IRNode] = []

    @contextmanager
    def arm(self) -> Generator[None, None, None]:
        """Context manager for a single branch arm."""
        self._builder._push()
        try:
            yield
        finally:
            arm_nodes = self._builder._pop()
            arm_ir = IRSeq(body=arm_nodes) if len(arm_nodes) != 1 else arm_nodes[0]
            self._arms.append(arm_ir)

    def finish(self) -> Tuple[IRNode, ...]:
        return tuple(self._arms)
