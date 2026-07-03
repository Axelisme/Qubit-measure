from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional, Union

from qick.asm_v2 import AsmV2, QickParam

from .debug import PrintTimeStamp
from .delay import DelayRegAuto
from .loop import CloseInnerLoop, OpenInnerLoop
from .meta import MetaMacro
from .pluse_reg import PulseByReg
from .wmem import PatchWmemFromRegs
from .write_reg import WriteRegOp


class AdditionalMacroMixin(AsmV2):
    def __init__(self, *args, **kwargs):
        self._delay_disabled = False
        self._temp_regs: list[str] = []
        self._reg_num_stack: list[int] = []

        super().__init__(*args, **kwargs)

    def _make_asm(self) -> None:
        # QICK resets its low-level program (registers, prog_list) at the
        # start of _make_asm via _init_instructions; mirror that here so our
        # scratch register names are not retained across compiles.  Without
        # this, a second compile would skip add_reg() while QICK has already
        # forgotten the name, causing _get_reg() lookups to fail.
        self._temp_regs = []
        self._reg_num_stack = []
        super()._make_asm()  # type: ignore[misc]

    def write_reg_op(
        self, dst: str, lhs: str, op: str, rhs: int | str | None = None
    ) -> None:
        """REG_WR dst = lhs op rhs, resolving register names at expand time."""
        self.append_macro(WriteRegOp(dst=dst, lhs=lhs, op=op, rhs=rhs))

    def open_inner_loop(
        self,
        name: str,
        counter_reg: str,
        n: str | int,
        *,
        range_hint: tuple[int, int] | None = None,
    ) -> None:
        self.append_macro(
            OpenInnerLoop(
                name=name, counter_reg=counter_reg, n=n, range_hint=range_hint
            )
        )

    def close_inner_loop(self, name: str, counter_reg: str, n: str | int) -> None:
        self.append_macro(CloseInnerLoop(name=name, counter_reg=counter_reg, n=n))

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
        t: float | QickParam = 0.0,
        gens=True,
        ros=True,
        tag: str | None = None,
    ):
        if self._delay_disabled:
            raise RuntimeError("Delay macros are currently disabled.")
        return super().delay_auto(t, gens, ros, tag)  # type: ignore

    def delay(self, t: float | QickParam, tag: str | None = None):
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
        t: float | QickParam = 0.0,
        flat_top_pulse: bool = False,
    ) -> None:
        """Play a pulse from wmem using a runtime-computed base address register.

        With ``flat_top_pulse=True``, fires 3 contiguous wmem entries (ramp_up
        at ``addr_reg``, flat at ``+1``, ramp_down at ``+2``) sharing the same
        TIME. The +1/+2 addresses are pre-computed into nested temp regs
        """
        if flat_top_pulse:
            # axis_sg_int4_v1/v2 generators emit a 4-entry stride for flat_top
            # (extra dummy entry); the +1/+2 stride assumed below would land
            # on the wrong entry, so reject explicitly rather than corrupt
            # the pulse silently.
            gen_type = self.soccfg["gens"][ch].get("type")  # type: ignore[attr-defined]
            if gen_type in ("axis_sg_int4_v1", "axis_sg_int4_v2"):
                raise NotImplementedError(
                    f"pulse_by_reg(flat_top_pulse=True) is not supported on "
                    f"generator type {gen_type!r}: int4 flat_top uses a "
                    f"4-entry wmem stride, not 3. ch={ch}"
                )
            with self.acquire_temp_reg(2) as (addr_reg2, addr_reg3):
                self.write_reg_op(addr_reg2, addr_reg, "+", 1)
                self.write_reg_op(addr_reg3, addr_reg, "+", 2)
                self.append_macro(
                    PulseByReg(ch=ch, t=t, addr_regs=[addr_reg, addr_reg2, addr_reg3])
                )
        else:
            self.append_macro(PulseByReg(ch=ch, t=t, addr_regs=[addr_reg]))

    def patch_wmem_from_regs(
        self,
        name: str,
        *,
        freq_reg: str | None = None,
        gain_reg: str | None = None,
    ) -> None:
        self.append_macro(
            PatchWmemFromRegs(name=name, freq_reg=freq_reg, gain_reg=gain_reg)
        )

    def debug_macro(self, name: str, t: float | QickParam, prefix: str = "") -> None:
        """Insert a debug macro that prints the current time (cycle count) with a name."""
        self.append_macro(PrintTimeStamp(name, t, prefix=prefix))

    def meta_macro(
        self,
        type: str,
        name: str,
        info: dict | None = None,
        regs: dict[str, str] | None = None,
    ) -> None:
        """Insert a meta macro that emits a meta instruction for the IR builder.

        regs maps info keys to register names resolved via prog._get_reg() at
        translate time.  Use this when the register is known only by a logical
        name (e.g. a loop name) rather than a bare ASM address.
        """
        self.append_macro(MetaMacro(type=type, name=name, info=info, regs=regs))

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

        # Guard against silent data-register exhaustion. tProc v2 has a limited
        # dreg pool (tproccfg['dreg_qty'], usually 32) shared with loop
        # counters, swept-time registers and user add_reg() calls. Without this
        # check, a deeply nested acquire_temp_reg would surface as an opaque
        # QICK add_reg() error instead of pointing at the real cause.
        dreg_qty = int(self.tproccfg.get("dreg_qty", 32))  # type: ignore[attr-defined]
        if total > dreg_qty:
            raise RuntimeError(
                f"acquire_temp_reg: requested {total} nested temp registers "
                f"but tProc v2 only has {dreg_qty} data registers (shared with "
                f"loop counters and user registers). Reduce nesting depth."
            )

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

    def wait(self, t: float | QickParam, tag: str | None = None) -> None:
        self.meta_macro("DISABLE_OPT_START", "")
        super().wait(t=t, tag=tag)  # type: ignore
        self.meta_macro("DISABLE_OPT_END", "")

    def wait_auto(
        self,
        t: float | QickParam = 0,
        gens: bool = False,
        ros: bool = True,
        tag: str | None = None,
        no_warn: bool = False,
    ) -> None:
        self.meta_macro("DISABLE_OPT_START", "")
        super().wait_auto(t=t, gens=gens, ros=ros, tag=tag, no_warn=no_warn)  # type: ignore
        self.meta_macro("DISABLE_OPT_END", "")

    def end(self) -> None:
        self.meta_macro("DISABLE_OPT_START", "")
        super().end()  # type: ignore
        self.meta_macro("DISABLE_OPT_END", "")


class ImproveAsmV2(AdditionalMacroMixin, AsmV2): ...


__all__ = ["ImproveAsmV2"]
