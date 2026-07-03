from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from qick.asm_v2 import AsmInst, Macro, WriteLabel

TraceKind = Literal[
    "add_pulse",
    "add_readout_config",
    "add_reg",
    "asm",
    "close_loop",
    "declare_gen",
    "declare_readout",
    "delay",
    "delay_auto",
    "delay_reg_auto",
    "disable_delay",
    "inc_reg",
    "jump",
    "label",
    "meta",
    "nop",
    "open_loop",
    "pulse",
    "patch_wmem",
    "read_dmem",
    "reg_write",
    "send_readout_config",
    "trigger",
    "write_label",
]

_FCLK = 430.08


@dataclass(frozen=True)
class TraceEvent:
    kind: TraceKind
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class _TracePulseRegistry:
    def __init__(self) -> None:
        self._next_id = 0
        self.registrations: list[tuple[str, Any]] = []

    def calc_name(self, cfg: Any) -> str:
        name = f"pulse_{self._next_id}"
        self._next_id += 1
        return name

    def register(self, name: str, cfg: Any) -> bool:
        self.registrations.append((name, cfg))
        return True


class ProgramTrace:
    """Tests-only semantic recorder for program emission calls."""

    def __init__(
        self,
        *,
        pmem_size: int = 512,
        reg_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self.events: list[TraceEvent] = []
        self.soccfg: dict[str, Any] = {
            "tprocs": [{"f_time": _FCLK}],
            "gens": [self._make_gen_entry() for _ in range(5)],
            "readouts": [self._make_readout_entry() for _ in range(5)],
        }
        self.tproccfg: dict[str, Any] = {"pmem_size": pmem_size, "dreg_qty": 32}
        self.pulse_registry = _TracePulseRegistry()
        self.max_timestamp = 0.0
        self.decremented_timestamps: list[Any] = []
        self.reg_resolver: Callable[[str], str] = reg_resolver or (lambda name: name)
        self.disable_delay_entries = 0
        self._dmem_offset = 0
        self._temp_regs: list[str] = []
        self._reg_num_stack: list[int] = []

    @staticmethod
    def _make_gen_entry(f_fabric: float = _FCLK) -> dict[str, Any]:
        return {"f_fabric": f_fabric, "tproc_ch": 0}

    @staticmethod
    def _make_readout_entry(f_output: float = _FCLK) -> dict[str, Any]:
        return {"f_output": f_output}

    def set_reg_map(self, reg_map: dict[str, str]) -> None:
        self.reg_resolver = lambda name: reg_map.get(name, name)

    def set_reg_resolver(self, reg_resolver: Callable[[str], str]) -> None:
        self.reg_resolver = reg_resolver

    def count(self, kind: TraceKind) -> int:
        return len(self.events_of(kind))

    def events_of(self, kind: TraceKind) -> list[TraceEvent]:
        return [event for event in self.events if event.kind == kind]

    def only(self, kind: TraceKind) -> TraceEvent:
        events = self.events_of(kind)
        if len(events) != 1:
            raise AssertionError(
                f"expected exactly one {kind!r} event, got {len(events)}"
            )
        return events[0]

    def has_no_events(self) -> bool:
        return len(self.events) == 0

    def _record(self, kind: TraceKind, *args: Any, **kwargs: Any) -> None:
        self.events.append(TraceEvent(kind=kind, args=args, kwargs=dict(kwargs)))

    def _get_reg(self, name: str) -> str:
        return self.reg_resolver(name)

    def _add_meta(self, *, type: str, name: str, info: dict[str, Any]) -> None:
        self._record("meta", type=type, name=name, info=dict(info))

    def get_max_timestamp(self, *, gens: bool = True, ros: bool = True) -> float:
        return self.max_timestamp

    def decrement_timestamps(self, value: Any) -> None:
        self.decremented_timestamps.append(value)

    def add_reg(self, name: str) -> None:
        self._record("add_reg", name=name)

    def declare_gen(self, ch: int, **kwargs: Any) -> None:
        self._record("declare_gen", ch=ch, **kwargs)

    def add_pulse(self, ch: int, name: str, **kwargs: Any) -> None:
        self._record("add_pulse", ch=ch, name=name, **kwargs)

    def add_cosine(self, ch: int, name: str, **kwargs: Any) -> None:
        self._record("asm", op="add_cosine", ch=ch, name=name, **kwargs)

    def add_gauss(self, ch: int, name: str, **kwargs: Any) -> None:
        self._record("asm", op="add_gauss", ch=ch, name=name, **kwargs)

    def add_DRAG(self, ch: int, name: str, **kwargs: Any) -> None:
        self._record("asm", op="add_DRAG", ch=ch, name=name, **kwargs)

    def add_envelope(self, ch: int, name: str, **kwargs: Any) -> None:
        self._record("asm", op="add_envelope", ch=ch, name=name, **kwargs)

    def declare_readout(self, *, ch: int, length: Any) -> None:
        self._record("declare_readout", ch=ch, length=length)

    def add_readoutconfig(
        self, *, ch: int, name: str, freq: Any, **kwargs: Any
    ) -> None:
        self._record("add_readout_config", ch=ch, name=name, freq=freq, **kwargs)

    def send_readoutconfig(self, ch: int, name: str, *, t: Any = 0.0) -> None:
        self._record("send_readout_config", ch=ch, name=name, t=t)

    def trigger(self, ros: Iterable[int], **kwargs: Any) -> None:
        self._record("trigger", ros=list(ros), **kwargs)

    def pulse(
        self,
        ch: int,
        name: str | None,
        *,
        t: Any = 0.0,
        tag: str | None = None,
    ) -> None:
        self._record("pulse", ch=ch, name=name, t=t, tag=tag)

    def patch_wmem_from_regs(
        self,
        name: str,
        *,
        freq_reg: str | None = None,
        gain_reg: str | None = None,
    ) -> None:
        self._record("patch_wmem", name=name, freq_reg=freq_reg, gain_reg=gain_reg)

    def delay(self, t: Any, tag: str | None = None) -> None:
        self._record("delay", t=t, tag=tag)

    def delay_auto(
        self,
        t: Any = 0.0,
        gens: bool = True,
        ros: bool = True,
        tag: str | None = None,
    ) -> None:
        self._record("delay_auto", t=t, gens=gens, ros=ros, tag=tag)

    def delay_reg_auto(
        self,
        time_reg: str,
        gens: bool = True,
        ros: bool = True,
    ) -> None:
        self._record("delay_reg_auto", time_reg=time_reg, gens=gens, ros=ros)

    @contextmanager
    def disable_delay(self) -> Generator[None]:
        self.disable_delay_entries += 1
        self._record("disable_delay", phase="start")
        try:
            yield
        finally:
            self._record("disable_delay", phase="end")

    def open_inner_loop(
        self,
        name: str,
        counter_reg: str,
        n: int | str,
        *,
        range_hint: tuple[int, int] | None = None,
    ) -> None:
        self._record(
            "open_loop",
            name=name,
            counter_reg=counter_reg,
            n=n,
            range_hint=range_hint,
        )

    def close_inner_loop(self, name: str, counter_reg: str, n: int | str) -> None:
        self._record("close_loop", name=name, counter_reg=counter_reg, n=n)

    def write_reg(self, dst: str, src: str | int) -> None:
        self._record("reg_write", mode="copy", dst=dst, src=src)

    def write_reg_op(
        self,
        dst: str,
        lhs: str,
        op: str,
        rhs: int | str | None = None,
    ) -> None:
        self._record("reg_write", mode="op", dst=dst, lhs=lhs, op=op, rhs=rhs)

    def inc_reg(self, reg: str, value: int) -> None:
        self._record("inc_reg", reg=reg, value=value)

    def read_dmem(self, *, dst: str, addr: str) -> None:
        self._record("read_dmem", dst=dst, addr=addr)

    def add_dmem(self, values: Iterable[int]) -> int:
        offset = self._dmem_offset
        stored = list(values)
        self._dmem_offset += len(stored)
        return offset

    @contextmanager
    def acquire_temp_reg(self, num: int = 1) -> Generator[list[str]]:
        if num < 0:
            raise ValueError(f"num must be greater than or equal to 0, got {num}")
        if num == 0:
            yield []
            return

        used = self._reg_num_stack[-1] if self._reg_num_stack else 0
        total = used + num
        dreg_qty = int(self.tproccfg.get("dreg_qty", 32))
        if total > dreg_qty:
            raise RuntimeError(
                f"acquire_temp_reg: requested {total} nested temp registers "
                f"but tProc v2 only has {dreg_qty} data registers"
            )

        while len(self._temp_regs) < total:
            reg_name = f"temp_reg_{len(self._temp_regs)}"
            self.add_reg(reg_name)
            self._temp_regs.append(reg_name)

        self._reg_num_stack.append(total)
        try:
            yield self._temp_regs[used:total]
        finally:
            self._reg_num_stack.pop()

    def jump(self, label: str) -> None:
        self._record("jump", label=label)

    def cond_jump(self, label: str, if_cond: str, op: str) -> None:
        self._record("jump", label=label, if_cond=if_cond, op=op)

    def nop(self) -> None:
        self._record("nop")

    def label(self, name: str) -> None:
        self._record("label", name=name)

    def meta_macro(
        self,
        type: str,
        name: str,
        info: dict[str, Any] | None = None,
        regs: dict[str, str] | None = None,
    ) -> None:
        resolved_info = dict(info or {})
        for key, reg_name in (regs or {}).items():
            resolved_info[key] = self._get_reg(reg_name)
        self._record("meta", type=type, name=name, info=resolved_info)

    def debug_macro(self, name: str, t: Any, prefix: str = "") -> None:
        return None

    def append_macro(self, macro: Macro) -> None:
        if isinstance(macro, WriteLabel):
            self._record("write_label", label=macro.label)
            return
        if isinstance(macro, AsmInst):
            self._record_asm_inst(macro.inst)
            return
        self._record("asm", macro=macro)

    def _record_asm_inst(self, inst: dict[str, Any]) -> None:
        command = inst.get("CMD")
        if command == "JUMP":
            self._record(
                "jump",
                label=inst.get("LABEL"),
                addr=inst.get("ADDR"),
                if_cond=inst.get("IF"),
            )
        elif command == "NOP":
            self._record("nop")
        elif command == "REG_WR":
            self._record("reg_write", mode="asm", inst=dict(inst))
        else:
            self._record("asm", inst=dict(inst))
