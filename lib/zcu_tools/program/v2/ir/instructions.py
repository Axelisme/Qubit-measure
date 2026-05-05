from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from .labels import Label

from .labels import PSEUDO_LABELS, is_pseudo_label_name, is_register_addr
from .utils import regs_from_value, strip_write_modifier


def _is_pseudo_label(value: Optional["Label"]) -> bool:
    if value is None:
        return False
    return is_pseudo_label_name(value.name)


def _serialize_addr(value: Optional[Union[str, "Label"]]) -> Optional[str]:
    from .labels import Label

    if value is None:
        return None
    if isinstance(value, Label):
        return value.name if is_pseudo_label_name(value.name) else f"&{value.name}"
    if not is_register_addr(value):
        raise ValueError(
            f"ADDR string must be a register address, got {value!r}. "
            "Use Label(...) for symbolic addresses."
        )
    return value


def _validate_addr_value(value: Optional[Union[str, "Label"]], *, field_name: str) -> None:
    from .labels import Label

    if value is None or isinstance(value, Label):
        return
    if isinstance(value, str) and is_register_addr(value):
        return
    if isinstance(value, str):
        raise ValueError(
            f"{field_name} string address must be a register address, got {value!r}. "
            "Use Label(...) for symbolic addresses."
        )
    raise TypeError(
        f"{field_name} must be str register address, Label, or None, got {type(value).__name__}."
    )


def _validate_jump_addr_target(value: Optional[Union[str, "Label"]]) -> None:
    from .labels import Label

    if value is None or isinstance(value, Label):
        return
    if value == "s15":
        return
    raise ValueError(
        f"JumpInst.addr must be 's15' or Label, got {value!r}."
    )

def _residual_fields(source: dict[str, Any], handled: set[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in source.items()
        if key not in handled and key not in {"CMD", "LINE", "P_ADDR"}
    }


@dataclass(frozen=True)
class Instruction:
    """Base class for all IR instructions."""

    @property
    def addr_inc(self) -> int:
        """Number of machine-code words this instruction will occupy."""
        return 1

    @property
    def reg_read(self) -> list[str]:
        """Registers read by this instruction."""
        return []

    @property
    def reg_write(self) -> list[str]:
        """Registers written by this instruction."""
        return []

    @property
    def need_label(self) -> Optional["Label"]:
        """Label name this instruction depends on (e.g. for JUMP or WR_ADDR)."""
        return None

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], label_map: Optional[dict[str, "Label"]] = None
    ) -> "Instruction":
        from .labels import Label

        def get_label(name: Any) -> Optional["Label"]:
            if not name:
                return None
            if isinstance(name, Label):
                return name
            if isinstance(name, str):
                if name.startswith("&"):
                    name = name[1:]
                if label_map is not None:
                    if name not in label_map and not is_pseudo_label_name(name):
                        label_map[name] = Label.make_new(name)
                    return label_map.get(name, Label(name))
                return Label(name)
            return None

        def resolve_addr(raw_addr: Any) -> Optional[Union[str, "Label"]]:
            if raw_addr is None:
                return None
            if isinstance(raw_addr, Label):
                return raw_addr
            if isinstance(raw_addr, str):
                if is_register_addr(raw_addr):
                    return raw_addr
                if raw_addr.startswith("&"):
                    return get_label(raw_addr)
                raise ValueError(
                    f"Invalid ADDR {raw_addr!r}: plain string labels are not supported. "
                    "Use register address (e.g. 's15') or '&label'."
                )
            raise ValueError(
                f"Invalid ADDR type {type(raw_addr).__name__}: expected register string, '&label', or Label."
            )

        if "LABEL" in d and "CMD" not in d:
            args = {k: v for k, v in d.items() if k not in ("LABEL", "LINE", "P_ADDR")}
            return LabelInst(
                name=get_label(d["LABEL"]),
                args=args,
            )

        cmd = d.get("CMD")
        if not cmd:
            raise ValueError(f"Unknown instruction format: {d}")

        # Dispatch to structured types for known opcodes
        if cmd == "TIME":
            extra_args = _residual_fields(d, {"C_OP", "LIT", "R1"})
            return TimeInst(
                c_op=d.get("C_OP", ""),
                lit=d.get("LIT"),
                r1=d.get("R1"),
                extra_args=extra_args,
            )
        elif cmd == "TEST":
            extra_args = _residual_fields(d, {"OP", "UF"})
            return TestInst(
                op=d.get("OP", ""),
                uf=d.get("UF"),
                extra_args=extra_args,
            )
        elif cmd == "JUMP":
            extra_args = _residual_fields(d, {"LABEL", "IF", "ADDR", "WR", "OP", "UF"})
            resolved_addr = resolve_addr(d.get("ADDR"))
            _validate_jump_addr_target(resolved_addr)
            return JumpInst(
                label=get_label(d.get("LABEL")),
                if_cond=d.get("IF"),
                addr=resolved_addr,
                wr=d.get("WR"),
                op=d.get("OP"),
                uf=d.get("UF"),
                extra_args=extra_args,
            )
        elif cmd == "REG_WR":
            src = d.get("SRC", "")
            wr = d.get("WR")
            if not d.get("DST") and wr:
                wr_parts = wr.split()
                if wr_parts:
                    d["DST"] = wr_parts[0]
                if len(wr_parts) > 1 and not src:
                    d["SRC"] = wr_parts[1]
                    src = wr_parts[1]

            resolved_addr = resolve_addr(d.get("ADDR"))

            if src == "dmem":
                extra_args = _residual_fields(
                    d,
                    {"DST", "SRC", "ADDR", "WR", "OP", "LIT", "UF", "IF", "LABEL"},
                )
                return DmemReadInst(
                    dst=d.get("DST", ""),
                    src="dmem",
                    addr=resolved_addr,
                    wr=wr,
                    op=d.get("OP"),
                    lit=d.get("LIT"),
                    uf=d.get("UF"),
                    if_cond=d.get("IF"),
                    label=get_label(d.get("LABEL")),
                    extra_args=extra_args,
                )
            extra_args = _residual_fields(
                d,
                {"DST", "SRC", "WR", "OP", "LIT", "ADDR", "UF", "IF", "LABEL"},
            )
            return RegWriteInst(
                dst=d.get("DST", ""),
                src=d.get("SRC", ""),
                wr=wr,
                op=d.get("OP"),
                lit=d.get("LIT"),
                addr=resolved_addr,
                uf=d.get("UF"),
                if_cond=d.get("IF"),
                label=get_label(d.get("LABEL")),
                extra_args=extra_args,
            )
        elif cmd == "WPORT_WR":
            extra_args = _residual_fields(
                d, {"DST", "SRC", "ADDR", "TIME", "WR", "OP", "UF", "IF"}
            )
            return PortWriteInst(
                dst=d.get("DST", ""),
                src=d.get("SRC"),
                addr=d.get("ADDR"),
                time=d.get("TIME"),
                wr=d.get("WR"),
                op=d.get("OP"),
                uf=d.get("UF"),
                if_cond=d.get("IF"),
                extra_args=extra_args,
            )
        elif cmd == "NOP":
            extra_args = _residual_fields(d, set())
            return NopInst(extra_args=extra_args)
        elif cmd == "DMEM_RD":
            resolved_addr = resolve_addr(d.get("ADDR"))

            extra_args = _residual_fields(
                d, {"DST", "SRC", "ADDR", "WR", "OP", "LIT", "UF", "IF", "LABEL"}
            )
            return DmemReadInst(
                dst=d.get("DST", ""),
                src=d.get("SRC", "dmem"),
                addr=resolved_addr,
                wr=d.get("WR"),
                op=d.get("OP"),
                lit=d.get("LIT"),
                uf=d.get("UF"),
                if_cond=d.get("IF"),
                label=get_label(d.get("LABEL")),
                extra_args=extra_args,
            )
        elif cmd == "DMEM_WR":
            extra_args = _residual_fields(
                d,
                {"DST", "SRC", "WR", "OP", "LIT", "UF", "IF"},
            )
            return DmemWriteInst(
                dst=d.get("DST", ""),
                src=d.get("SRC", ""),
                wr=d.get("WR"),
                op=d.get("OP"),
                lit=d.get("LIT"),
                uf=d.get("UF"),
                if_cond=d.get("IF"),
                extra_args=extra_args,
            )
        elif cmd == "WMEM_WR":
            extra_args = _residual_fields(
                d,
                {"ADDR", "TIME", "WR", "OP", "UF", "IF"},
            )
            return WmemWriteInst(
                addr=d.get("ADDR"),
                time=d.get("TIME"),
                wr=d.get("WR"),
                op=d.get("OP"),
                uf=d.get("UF"),
                if_cond=d.get("IF"),
                extra_args=extra_args,
            )
        elif cmd == "DPORT_WR":
            extra_args = _residual_fields(
                d,
                {"DST", "SRC", "DATA", "TIME", "WR", "OP", "UF", "IF"},
            )
            return DportWriteInst(
                dst=d.get("DST", ""),
                src=d.get("SRC"),
                data=d.get("DATA", ""),
                time=d.get("TIME"),
                wr=d.get("WR"),
                op=d.get("OP"),
                uf=d.get("UF"),
                if_cond=d.get("IF"),
                extra_args=extra_args,
            )
        elif cmd == "WAIT":
            extra_args = _residual_fields(d, {"C_OP", "TIME", "ADDR"})
            return WaitInst(
                c_op=d.get("C_OP", "time"),
                time=d.get("TIME"),
                addr=resolve_addr(d.get("ADDR")),
                extra_args=extra_args,
            )

        raise ValueError(f"Unknown instruction opcode: {cmd!r}")

    def to_dict(self) -> dict[str, Any]:
        """Convert back to QICK prog_list dict format."""
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.to_dict().items() if v and k not in ('CMD')])})"


@dataclass(frozen=True)
class LabelInst(Instruction):
    name: Optional["Label"] = None
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"LABEL": str(self.name)}
        d.update(self.args)
        return d


@dataclass(frozen=True)
class MetaInst(Instruction):
    """Meta instruction used for structural control like loops."""

    type: str = ""
    name: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def addr_inc(self) -> int:
        return 0  # MetaInst occupies no program memory

    def to_dict(self) -> dict[str, Any]:
        return {
            "CMD": "__META__",
            "TYPE": self.type,
            "NAME": self.name,
            "INFO": self.info,
        }


@dataclass(frozen=True)
class TimeInst(Instruction):
    """TIME instruction: advance timing counter."""

    c_op: str = ""  # inc_ref, trigger, etc.
    lit: Optional[str] = None  # e.g., "#10" for literal value
    r1: Optional[str] = None  # register operand
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def reg_read(self) -> list[str]:
        return sorted(list(regs_from_value(self.r1)))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "TIME", "C_OP": self.c_op}
        if self.lit is not None:
            d["LIT"] = self.lit
        if self.r1 is not None:
            d["R1"] = self.r1
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class TestInst(Instruction):
    """TEST instruction: evaluate condition for conditional branch."""

    __test__ = False
    op: str = ""  # The condition to test (e.g., "r1 == r2")
    uf: Optional[str] = None  # Overflow/underflow flag (usually "1")
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def reg_read(self) -> list[str]:
        return sorted(list(regs_from_value(self.op)))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "TEST", "OP": self.op}
        if self.uf is not None:
            d["UF"] = self.uf
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class JumpInst(Instruction):
    """JUMP instruction: unconditional or conditional jump."""

    label: Optional["Label"] = None  # Target label
    if_cond: Optional[str] = None  # Condition for conditional jump (e.g., "eq", "nz")
    addr: Optional[Union[str, "Label"]] = (
        None  # Direct address for large jumps (e.g., "s15" or label)
    )
    wr: Optional[str] = None  # Optional register write control string
    op: Optional[str] = None  # Optional ALU expression
    uf: Optional[str] = None  # Optional flag update control
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="JumpInst.addr")
        _validate_jump_addr_target(self.addr)

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        if isinstance(self.addr, str):
            reads.update(regs_from_value(self.addr))
        reads.update(regs_from_value(self.op))
        reads.update(regs_from_value(self.wr))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        if self.wr:
            return [strip_write_modifier(self.wr)]
        return []

    @property
    def need_label(self) -> Optional["Label"]:
        from .labels import Label

        if self.label and not _is_pseudo_label(self.label):
            return self.label

        if isinstance(self.addr, Label) and not _is_pseudo_label(self.addr):
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "JUMP"}
        if self.label:
            d["LABEL"] = str(self.label)
        if self.addr is not None:
            d["ADDR"] = _serialize_addr(self.addr)
        if self.if_cond is not None:
            d["IF"] = self.if_cond
        if self.wr is not None:
            d["WR"] = self.wr
        if self.op is not None:
            d["OP"] = self.op
        if self.uf is not None:
            d["UF"] = self.uf
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class RegWriteInst(Instruction):
    """REG_WR instruction: write to register."""

    dst: str = ""  # Destination register
    src: str = ""  # Source: 'op' (ALU operation), 'imm' (immediate), 'reg' (register), 'dmem' (memory)
    op: Optional[str] = None
    lit: Optional[str] = None
    addr: Optional[Union[str, "Label"]] = None
    uf: Optional[str] = None
    wr: Optional[str] = None
    if_cond: Optional[str] = None
    label: Optional["Label"] = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="RegWriteInst.addr")

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        reads.update(regs_from_value(self.op))
        if isinstance(self.addr, str):
            reads.update(regs_from_value(self.addr))
        reads.update(regs_from_value(self.extra_args.get("OP")))
        if isinstance(self.extra_args.get("ADDR"), str):
            reads.update(regs_from_value(self.extra_args.get("ADDR")))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return [strip_write_modifier(self.dst)]

    @property
    def need_label(self) -> Optional["Label"]:
        from .labels import Label

        if self.label and not _is_pseudo_label(self.label):
            return self.label

        if isinstance(self.addr, Label) and not _is_pseudo_label(self.addr):
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "REG_WR"}
        if self.wr is not None:
            d["WR"] = self.wr
        else:
            d["DST"] = self.dst
            d["SRC"] = self.src
        if self.op is not None:
            d["OP"] = self.op
        if self.lit is not None:
            d["LIT"] = self.lit
        if self.addr is not None:
            d["ADDR"] = _serialize_addr(self.addr)
        if self.uf is not None:
            d["UF"] = self.uf
        if self.if_cond is not None:
            d["IF"] = self.if_cond
        if self.label is not None:
            d["LABEL"] = str(self.label)
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class PortWriteInst(Instruction):
    """WPORT_WR instruction: write to output port."""

    dst: str = ""  # Destination (output port)
    src: Optional[str] = None
    addr: Optional[str] = None
    time: Optional[str] = None  # Timing reference
    wr: Optional[str] = None
    op: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    extra_args: dict[str, Any] = field(default_factory=dict)  # Other fields

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        reads.update(regs_from_value(self.addr))
        reads.update(regs_from_value(self.time))
        reads.update(regs_from_value(self.op))
        reads.update(regs_from_value(self.wr))
        for val in self.extra_args.values():
            reads.update(regs_from_value(val))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        dst = strip_write_modifier(self.dst)
        # DST can be a plain port number (e.g. "2") or a register (e.g. "p1").
        # Only return it as a register write when it is not a bare integer.
        return [dst] if dst and not dst.lstrip("-").isdigit() else []

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "WPORT_WR", "DST": self.dst}
        if self.src is not None:
            d["SRC"] = self.src
        if self.addr is not None:
            d["ADDR"] = self.addr
        if self.time is not None:
            d["TIME"] = self.time
        if self.wr is not None:
            d["WR"] = self.wr
        if self.op is not None:
            d["OP"] = self.op
        if self.uf is not None:
            d["UF"] = self.uf
        if self.if_cond is not None:
            d["IF"] = self.if_cond
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class NopInst(Instruction):
    """NOP instruction: no operation."""

    extra_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "NOP"}
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class DmemReadInst(Instruction):
    """DMEM read lowered to the native REG_WR form."""

    dst: str = ""  # Destination register
    src: str = "dmem"
    addr: Optional[Union[str, "Label"]] = None  # Memory address (register or label)
    wr: Optional[str] = None
    op: Optional[str] = None
    lit: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    label: Optional["Label"] = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="DmemReadInst.addr")

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        if isinstance(self.addr, str):
            reads.update(regs_from_value(self.addr))
        reads.update(regs_from_value(self.op))
        for val in self.extra_args.values():
            if isinstance(val, str):
                reads.update(regs_from_value(val))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return [strip_write_modifier(self.dst)]

    @property
    def need_label(self) -> Optional["Label"]:
        from .labels import Label

        if self.label and not _is_pseudo_label(self.label):
            return self.label

        if isinstance(self.addr, Label) and not _is_pseudo_label(self.addr):
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "REG_WR", "DST": self.dst, "SRC": self.src}
        if self.addr is not None:
            d["ADDR"] = _serialize_addr(self.addr)
        if self.wr is not None:
            d["WR"] = self.wr
        if self.op is not None:
            d["OP"] = self.op
        if self.lit is not None:
            d["LIT"] = self.lit
        if self.uf is not None:
            d["UF"] = self.uf
        if self.if_cond is not None:
            d["IF"] = self.if_cond
        if self.label is not None:
            d["LABEL"] = str(self.label)
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class DmemWriteInst(Instruction):
    """DMEM_WR instruction: write to data memory."""

    dst: str = ""  # Memory destination
    src: str = ""  # Source (register or literal)
    op: Optional[str] = None
    lit: Optional[str] = None
    wr: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        reads.update(regs_from_value(self.op))
        for val in self.extra_args.values():
            reads.update(regs_from_value(val))
        return sorted(list(reads))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "DMEM_WR", "DST": self.dst, "SRC": self.src}
        if self.op is not None:
            d["OP"] = self.op
        if self.lit is not None:
            d["LIT"] = self.lit
        if self.wr is not None:
            d["WR"] = self.wr
        if self.uf is not None:
            d["UF"] = self.uf
        if self.if_cond is not None:
            d["IF"] = self.if_cond
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class WmemWriteInst(Instruction):
    """WMEM_WR instruction: write wave registers into wave memory."""

    addr: Optional[str] = None
    time: Optional[str] = None
    wr: Optional[str] = None
    op: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.addr))
        reads.update(regs_from_value(self.time))
        reads.update(regs_from_value(self.op))
        reads.update(regs_from_value(self.wr))
        for val in self.extra_args.values():
            reads.update(regs_from_value(val))
        return sorted(list(reads))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "WMEM_WR"}
        if self.addr is not None:
            d["ADDR"] = self.addr
        if self.time is not None:
            d["TIME"] = self.time
        if self.wr is not None:
            d["WR"] = self.wr
        if self.op is not None:
            d["OP"] = self.op
        if self.uf is not None:
            d["UF"] = self.uf
        if self.if_cond is not None:
            d["IF"] = self.if_cond
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class DportWriteInst(Instruction):
    """DPORT_WR instruction: write to data port."""

    dst: str = ""  # Destination (port)
    src: Optional[str] = None
    data: str = ""  # Data to write (register or literal)
    time: Optional[str] = None
    wr: Optional[str] = None
    op: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        reads.update(regs_from_value(self.dst))
        reads.update(regs_from_value(self.data))
        reads.update(regs_from_value(self.time))
        reads.update(regs_from_value(self.op))
        reads.update(regs_from_value(self.wr))
        for val in self.extra_args.values():
            reads.update(regs_from_value(val))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        # DPORT_WR usually writes to a port, but if DST is a register...
        # In QICK DPORT_WR, DST is usually a literal port number.
        return [strip_write_modifier(self.dst)] if not self.dst.startswith("#") else []

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "DPORT_WR", "DST": self.dst, "DATA": self.data}
        if self.src is not None:
            d["SRC"] = self.src
        if self.time is not None:
            d["TIME"] = self.time
        if self.wr is not None:
            d["WR"] = self.wr
        if self.op is not None:
            d["OP"] = self.op
        if self.uf is not None:
            d["UF"] = self.uf
        if self.if_cond is not None:
            d["IF"] = self.if_cond
        d.update(self.extra_args)
        return d


@dataclass(frozen=True)
class WaitInst(Instruction):
    """WAIT instruction: wait for sync/trigger."""

    c_op: str = "time"
    time: Optional[str] = None
    addr: Optional[Union[str, "Label"]] = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="WaitInst.addr")

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.time))
        if isinstance(self.addr, str):
            reads.update(regs_from_value(self.addr))
        for val in self.extra_args.values():
            if isinstance(val, str):
                reads.update(regs_from_value(val))
        return sorted(list(reads))

    @property
    def need_label(self) -> Optional["Label"]:
        from .labels import Label

        if isinstance(self.addr, Label) and not _is_pseudo_label(self.addr):
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "WAIT", "C_OP": self.c_op}
        if self.time is not None:
            d["TIME"] = self.time
        if self.addr is not None:
            d["ADDR"] = _serialize_addr(self.addr)
        d.update(self.extra_args)
        return d

    @property
    def addr_inc(self) -> int:
        return 2
