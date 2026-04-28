from __future__ import annotations

from contextlib import contextmanager

from qick.asm_v2 import AsmV2, QickParam
from typing_extensions import Generator, Optional, Union

from .debug import PrintTimeStamp
from .delay import DelayRegAuto
from .loop import CloseLoopReg, OpenLoopReg
from .pluse_reg import PulseFromWmemReg
from .write_reg import WriteRegOp


class ImproveAsmV2(AsmV2):
    def __init__(self, *args, **kwargs):
        self._delay_disabled = False
        self._temp_regs: list[str] = []
        self._temp_reg_scope_stack: list[int] = []

        super().__init__(*args, **kwargs)

    def write_reg_op(
        self, dst: str, lhs: str, op: str, rhs: Union[int, str, None] = None
    ) -> None:
        """REG_WR dst = lhs op rhs, resolving register names at expand time."""
        self.append_macro(WriteRegOp(dst=dst, lhs=lhs, op=op, rhs=rhs))

    def open_loop_reg(self, n_reg: str, name: str) -> None:
        """Start a register-driven loop.

        Counter register ``{name}_count`` is allocated by the macro's
        preprocess. Must be closed by ``close_loop_reg`` with the same name.
        """
        self.append_macro(OpenLoopReg(name=name, n_reg=n_reg))

    def close_loop_reg(self, name: str) -> None:
        """Close the register-driven loop opened with ``open_loop_reg(name)``."""
        self.append_macro(CloseLoopReg(name=name))

    # ---- delay macro ----

    @contextmanager
    def disable_delay(self):
        self._delay_disabled = True
        try:
            yield
        finally:
            self._delay_disabled = False

    def delay_auto(
        self,
        t: Union[float, QickParam] = 0.0,
        gens=True,
        ros=True,
        tag: Optional[str] = None,
    ):
        if self._delay_disabled:
            raise RuntimeError("Delay macros are currently disabled.")
        return super().delay_auto(t, gens, ros, tag)  # type: ignore

    def delay(self, t: Union[float, QickParam], tag: Optional[str] = None):
        if self._delay_disabled:
            raise RuntimeError("Delay macros are currently disabled.")
        return super().delay(t, tag)  # type: ignore

    def delay_reg_auto(self, time_reg: str, gens=True, ros=True) -> None:
        """Auto-align to timeline, then increment by runtime cycles from a register."""
        if self._delay_disabled:
            raise RuntimeError("Delay macros are currently disabled.")
        self.append_macro(DelayRegAuto(time_reg=time_reg, gens=gens, ros=ros))

    def debug_macro(
        self, name: str, t: Union[float, QickParam], prefix: str = ""
    ) -> None:
        """Insert a debug macro that prints the current time (cycle count) with a name."""
        self.append_macro(PrintTimeStamp(name, t, prefix=prefix))

    def pulse_wmem_reg(
        self,
        ch: int,
        addr_reg: str,
        t: Union[float, QickParam] = 0.0,
        flat_top_pulse: bool = False,
    ) -> None:
        """Play a pulse from wmem using a runtime-computed base address register.

        With ``flat_top_pulse=True``, fires 3 contiguous wmem entries (ramp_up
        at ``addr_reg``, flat at ``+1``, ramp_down at ``+2``) sharing the same
        TIME. The +1/+2 addresses are pre-computed into nested temp regs
        *before* the macro is appended, so the macro emits 3 ``WPORT_WR``
        back-to-back with no other instructions interleaved (preserving
        hardware-level pulse continuity).
        """
        if not flat_top_pulse:
            self.append_macro(PulseFromWmemReg(ch=ch, t=t, addr_regs=[addr_reg]))
            return

        with self.acquire_temp_reg(2) as (addr_reg2, addr_reg3):
            self.write_reg_op(addr_reg2, addr_reg, "+", 1)
            self.write_reg_op(addr_reg3, addr_reg, "+", 2)
            self.append_macro(
                PulseFromWmemReg(ch=ch, t=t, addr_regs=[addr_reg, addr_reg2, addr_reg3])
            )

    @contextmanager
    def acquire_temp_reg(self, num: int = 1) -> Generator[list[str]]:
        """Acquire scratch registers for temporary calculations.

        Nested calls return disjoint registers: an outer ``acquire_temp_reg(N)``
        followed by a nested ``acquire_temp_reg(M)`` yields ``[temp_reg_0..N-1]``
        and ``[temp_reg_N..N+M-1]`` respectively, so inner work cannot clobber
        outer state.
        """
        if num < 0:
            raise ValueError(f"num must be greater than or equal to 0, got {num}")
        elif num == 0:
            yield []
            return

        used = self._temp_reg_scope_stack[-1] if self._temp_reg_scope_stack else 0
        total = used + num

        while len(self._temp_regs) < total:
            reg_name = f"temp_reg_{len(self._temp_regs)}"
            self.add_reg(reg_name)  # type: ignore
            self._temp_regs.append(reg_name)

        self._temp_reg_scope_stack.append(total)
        try:
            yield self._temp_regs[used:total]
        finally:
            if len(self._temp_reg_scope_stack) == 0:
                raise RuntimeError("temp register scope stack is already empty")
            popped = self._temp_reg_scope_stack.pop()
            if popped != total:
                raise RuntimeError(
                    "temp register scope mismatch: "
                    f"expected total {total}, got {popped}"
                )


__all__ = ["ImproveAsmV2"]
