from __future__ import annotations


from qick.asm_v2 import AsmV2, QickParam
from typing_extensions import Union, Optional
from contextlib import contextmanager

from .debug import PrintTimeStamp
from .delay import DelayRegAuto
from .loop import CloseLoopReg, OpenLoopReg
from .pluse_reg import PulseFromWmemReg
from .write_reg import WriteRegOp


class ImproveAsmV2(AsmV2):
    def __init__(self, *args, **kwargs):
        self._delay_disabled = False
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
        self, ch: int, addr_reg: str, t: Union[float, QickParam] = 0.0, tag=None
    ) -> None:
        """Play one waveform from wmem using a runtime-computed address register."""
        self.append_macro(PulseFromWmemReg(ch=ch, addr_reg=addr_reg, t=t, tag=tag))


__all__ = ["ImproveAsmV2"]
