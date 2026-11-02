from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal, Optional, Union

from qick.asm_v2 import AsmV2, ConfigReadout, Pulse, QickParam, QickProgramV2

from .debug import PrintTimeStamp
from .delay import DelayRegAuto
from .loop import CloseInnerLoop, OpenInnerLoop
from .meta import MetaMacro
from .pluse_reg import PulseByReg
from .structure import MarkedCloseLoop, MarkedOpenLoop, SourcePulse
from .wmem import ConfigReadoutFromRegs, PulseFromLengthReg, PulseFromRegs
from .write_reg import WriteRegOp


class AdditionalMacroMixin(QickProgramV2):
    """Extend QICK's program with native loop and register-output macros.

    Constructor arguments and compile failures follow QickProgramV2.
    Every declaration rebuild resets compiler-owned loop metadata.
    loop_wave_tables maps loop names to WMEM addresses, field indexes 0..5,
    and rotated DMEM word offsets. loop_register_tables maps loop names and
    register addresses to offset (rotated DMEM offset) and values (ordered
    integer words). Opening and closing macros populate these maps at compile.
    """

    def __init__(self, *args, **kwargs):
        self._delay_disabled = False
        self._source_names: list[str] = []
        self._native_pulse_ordinal = 0
        self._temp_regs: list[str] = []
        self._reg_num_stack: list[int] = []

        super().__init__(*args, **kwargs)

    def _init_declarations(self) -> None:
        self._source_names = []
        self._native_pulse_ordinal = 0
        # Native loop discovery shares rotated DMEM offsets with its closing shell.
        self.loop_wave_tables: dict[str, dict[int, dict[int, int]]] = {}
        self.loop_register_tables: dict[
            str, dict[str, dict[str, int | tuple[int, ...]]]
        ] = {}
        super()._init_declarations()

    @contextmanager
    def source_scope(self, name: str) -> Generator[None]:
        """Associate emitted operations with a diagnostic source until scope exit.

        name is a Module diagnostic label. Nested scopes override the active
        source and restore it on exit. SOURCE_START/SOURCE_END are zero-word,
        artifact-only annotations; native pulse fallbacks are suppressed inside
        a scope and those calls do not consume native pulse ordinals.
        """
        self.meta_macro("SOURCE_START", name)
        self._source_names.append(name)
        try:
            yield
        finally:
            self._source_names.pop()
            self.meta_macro("SOURCE_END", name)

    def _make_asm(self) -> None:
        # QICK resets its low-level program (registers, prog_list) at the
        # start of _make_asm via _init_instructions; mirror that here so our
        # scratch register names are not retained across compiles.  Without
        # this, a second compile would skip add_reg() while QICK has already
        # forgotten the name, causing _get_reg() lookups to fail.
        self._temp_regs = []
        self._reg_num_stack = []
        super()._make_asm()

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

    def open_loop(self, n: int, name: str | None = None) -> None:
        """Open a native QICK loop and retain its structure in compiled_stream.

        n is the positive trip count; name is the loop register/label name.
        QICK allocates an automatic name when None is supplied.
        """
        self.append_macro(MarkedOpenLoop(n=n, name=name))

    def pulse(
        self,
        ch: int,
        name: str,
        t: float | QickParam | Literal["auto"] = 0,
        tag: str | None = None,
    ) -> None:
        """Play a native QICK pulse and annotate its output source.

        ch and name select a declared generator/pulse; t is us or "auto".
        Outside source_scope, tag is the source when provided; otherwise use
        native:pulse:<ordinal>. Every unscoped call consumes an ordinal starting
        at one per compile, including tagged calls. Scoped Module pulses keep
        the Module source and consume no ordinal. QICK timing errors propagate.
        """
        if self._source_names:
            self.append_macro(Pulse(ch=ch, name=name, t=t, tag=tag))
        else:
            self._native_pulse_ordinal += 1
            source = (
                tag if tag is not None else f"native:pulse:{self._native_pulse_ordinal}"
            )
            self.append_macro(
                SourcePulse(ch=ch, name=name, t=t, tag=tag, source_name=source)
            )

    def close_loop(self) -> None:
        """Close the innermost native QICK loop, including its sweep updates."""
        self.append_macro(MarkedCloseLoop())

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
        # tolerated: QickProgramV2.delay_auto accepts generator-end selection positionally.
        gens=True,  # noqa: FBT002
        # tolerated: QickProgramV2.delay_auto accepts readout-end selection positionally.
        ros=True,  # noqa: FBT002
        tag: str | None = None,
    ):
        if self._delay_disabled:
            raise RuntimeError("Delay macros are currently disabled.")
        # tolerated: QICK infers t as int from its default; the timing macro accepts float and QickParam microseconds.
        return super().delay_auto(t, gens, ros, tag)  # pyright: ignore[reportArgumentType]

    def delay(self, t: float | QickParam, tag: str | None = None):
        if self._delay_disabled:
            raise RuntimeError("Delay macros are currently disabled.")
        return super().delay(t, tag)

    def delay_reg_auto(self, time_reg: str, *, gens=True, ros=True) -> None:
        """Auto-align to timeline, then increment by runtime cycles from a register."""
        if self._delay_disabled:
            raise RuntimeError("Delay macros are currently disabled.")
        self.append_macro(DelayRegAuto(time_reg=time_reg, gens=gens, ros=ros))

    def pulse_by_reg(
        self,
        ch: int,
        addr_reg: str,
        t: float | QickParam = 0.0,
        *,
        flat_top_pulse: bool = False,
    ) -> None:
        """Play a pulse from wmem using a runtime-computed base address register.

        With ``flat_top_pulse=True``, fires 3 contiguous wmem entries (ramp_up
        at ``addr_reg``, flat at ``+1``, ramp_down at ``+2``) sharing the same
        TIME. The +1/+2 addresses are pre-computed into nested temp regs
        """
        source = PulseByReg.from_base(
            self, ch, addr_reg, t, flat_top_pulse=flat_top_pulse
        )
        with self.acquire_temp_reg(source.address_scratch_count) as registers:
            for operation in source.prepare_addresses(self, tuple(registers)):
                self.append_macro(operation)
            self.append_macro(source)

    def pulse_from_regs(
        self,
        ch: int,
        name: str,
        *,
        t: float | QickParam = 0.0,
        tag: str | None = None,
        freq_reg: str | None = None,
        gain_reg: str | None = None,
    ) -> None:
        self.append_macro(
            PulseFromRegs(
                ch=ch,
                name=name,
                t=t,
                tag=tag,
                freq_reg=freq_reg,
                gain_reg=gain_reg,
            )
        )

    def pulse_from_length_reg(
        self,
        ch: int,
        name: str,
        *,
        length_reg: str,
        flat_top: bool,
        t: float | QickParam = 0.0,
    ) -> None:
        """Play a const/flat-top pulse template with runtime length cycles."""
        self.append_macro(
            PulseFromLengthReg(
                ch=ch,
                name=name,
                length_reg=length_reg,
                flat_top=flat_top,
                t=t,
            )
        )

    def send_readoutconfig(
        self,
        ch: int,
        name: str,
        t: float | QickParam = 0.0,
        tag: str | None = None,
    ) -> None:
        """Send a registered ADC configuration at scalar or swept time t in us.

        ch is the ADC channel and name its add_readoutconfig name. Optional tag
        identifies timing for get_time_param. This preserves QICK's scheduling
        semantics while exposing its documented QickParam input type.
        """
        self.append_macro(ConfigReadout(ch=ch, name=name, t=t, tag=tag))

    def send_readoutconfig_from_regs(
        self,
        ch: int,
        name: str,
        *,
        t: float | QickParam = 0.0,
        tag: str | None = None,
        freq_reg: str | None = None,
    ) -> None:
        self.append_macro(
            ConfigReadoutFromRegs(
                ch=ch,
                name=name,
                t=t,
                tag=tag,
                freq_reg=freq_reg,
            )
        )

    def debug_macro(self, name: str, t: float | QickParam, prefix: str = "") -> None:
        """Insert a debug macro that prints the current time (cycle count) with a name."""
        self.append_macro(PrintTimeStamp(name, t, prefix=prefix))

    def meta_macro(
        self,
        type: str,
        name: str,
        info: dict[str, Any] | None = None,
        regs: dict[str, str] | None = None,
    ) -> None:
        """Insert a meta macro that emits a meta instruction for the IR builder.

        info maps string keys to arbitrary metadata values passed to MetaMacro.
        regs maps info keys to register names resolved by the compiler at
        translate time. Use it for logical names (e.g. a loop name), ASM
        addresses or QICK aliases; resolved addresses overwrite matching info
        keys without mutating the inputs.
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
        dreg_qty = int(self.tproccfg.get("dreg_qty", 32))
        if total > dreg_qty:
            raise RuntimeError(
                f"acquire_temp_reg: requested {total} nested temp registers "
                f"but tProc v2 only has {dreg_qty} data registers (shared with "
                f"loop counters and user registers). Reduce nesting depth."
            )

        while len(self._temp_regs) < total:
            reg_name = f"temp_reg_{len(self._temp_regs)}"
            self.add_reg(reg_name)
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
        super().wait(t=t, tag=tag)
        self.meta_macro("DISABLE_OPT_END", "")

    def wait_auto(
        self,
        t: float | QickParam = 0,
        # tolerated: QickProgramV2.wait_auto accepts generator-end selection positionally.
        gens: bool = False,  # noqa: FBT001, FBT002
        # tolerated: QickProgramV2.wait_auto accepts readout-end selection positionally.
        ros: bool = True,  # noqa: FBT001, FBT002
        tag: str | None = None,
        # tolerated: QickProgramV2.wait_auto accepts swept-wait warning control positionally.
        no_warn: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        self.meta_macro("DISABLE_OPT_START", "")
        # tolerated: QICK infers t as int from its default; the timing macro accepts float and QickParam microseconds.
        super().wait_auto(t=t, gens=gens, ros=ros, tag=tag, no_warn=no_warn)  # pyright: ignore[reportArgumentType]
        self.meta_macro("DISABLE_OPT_END", "")

    def end(self) -> None:
        self.meta_macro("DISABLE_OPT_START", "")
        super().end()
        self.meta_macro("DISABLE_OPT_END", "")


class ImproveAsmV2(AdditionalMacroMixin, AsmV2): ...


__all__ = ["ImproveAsmV2"]
