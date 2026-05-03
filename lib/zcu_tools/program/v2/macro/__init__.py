from __future__ import annotations

from contextlib import contextmanager

from qick.asm_v2 import AsmV2, QickParam
from typing_extensions import Generator, Optional, Union

from .debug import PrintTimeStamp
from .delay import DelayRegAuto
from .loop import CloseInnerLoop, OpenInnerLoop
from .meta import MetaMacro
from .pluse_reg import PulseByReg
from .write_reg import WriteRegOp


class AdditionalMacroMixin(AsmV2):
    def __init__(self, *args, **kwargs):
        self._delay_disabled = False
        self._temp_regs: list[str] = []
        self._reg_num_stack: list[int] = []

        super().__init__(*args, **kwargs)

    def write_reg_op(
        self, dst: str, lhs: str, op: str, rhs: Union[int, str, None] = None
    ) -> None:
        """REG_WR dst = lhs op rhs, resolving register names at expand time."""
        self.append_macro(WriteRegOp(dst=dst, lhs=lhs, op=op, rhs=rhs))

    def open_inner_loop(
        self,
        name: str,
        counter_reg: str,
        n: Union[str, int],
        *,
        range_hint: Optional[tuple[int, int]] = None,
    ) -> None:
        self.append_macro(
            OpenInnerLoop(
                name=name, counter_reg=counter_reg, n=n, range_hint=range_hint
            )
        )

    def close_inner_loop(self, name: str, counter_reg: str) -> None:
        self.append_macro(CloseInnerLoop(name=name, counter_reg=counter_reg))

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

    def pulse_by_reg(
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
        """
        if flat_top_pulse:
            with self.acquire_temp_reg(2) as (addr_reg2, addr_reg3):
                self.write_reg_op(addr_reg2, addr_reg, "+", 1)
                self.write_reg_op(addr_reg3, addr_reg, "+", 2)
                self.append_macro(
                    PulseByReg(ch=ch, t=t, addr_regs=[addr_reg, addr_reg2, addr_reg3])
                )
        else:
            self.append_macro(PulseByReg(ch=ch, t=t, addr_regs=[addr_reg]))

    def debug_macro(
        self, name: str, t: Union[float, QickParam], prefix: str = ""
    ) -> None:
        """Insert a debug macro that prints the current time (cycle count) with a name."""
        self.append_macro(PrintTimeStamp(name, t, prefix=prefix))

    def meta_macro(
        self, type: str, name: str, info: Optional[dict] = None
    ) -> None:
        """Insert a meta macro that emits a meta instruction for the IR builder."""
        self.append_macro(MetaMacro(type=type, name=name, info=info))

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

        used = self._reg_num_stack[-1] if self._reg_num_stack else 0
        total = used + num

        while len(self._temp_regs) < total:
            reg_name = f"temp_reg_{len(self._temp_regs)}"
            self.add_reg(reg_name)  # type: ignore
            self._temp_regs.append(reg_name)

        self._reg_num_stack.append(total)
        try:
            yield self._temp_regs[used:total]
        finally:
            if len(self._reg_num_stack) == 0:
                raise RuntimeError("temp register scope stack is already empty")
            self._reg_num_stack.pop()


class ImproveAsmV2(AdditionalMacroMixin, AsmV2): ...


__all__ = ["ImproveAsmV2"]
