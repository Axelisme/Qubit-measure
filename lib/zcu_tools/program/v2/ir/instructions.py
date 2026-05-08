from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .labels import Label, is_pseudo_label_name, is_register_addr
from .utils import regs_from_value, strip_write_modifier


def _is_pseudo_label(value: Optional[Label]) -> bool:
    if value is None:
        return False
    return is_pseudo_label_name(value.name)


def _serialize_addr(value: Optional[Union[str, Label]]) -> Optional[str]:
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


def _validate_addr_value(
    value: Optional[Union[str, Label]], *, field_name: str
) -> None:
    if value is None or isinstance(value, Label):
        return
    if isinstance(value, str) and is_register_addr(value):
        return
    if isinstance(value, str):
        raise ValueError(
            f"{field_name} string address must be a register address, got {value!r}. "
            "Use Label(...) for symbolic addresses."
        )


def _validate_jump_addr_target(value: Optional[Union[str, Label]]) -> None:
    if value is None or isinstance(value, Label):
        return
    if value == "s15":
        return
    raise ValueError(f"JumpInst.addr must be 's15' or Label, got {value!r}.")


def _get_label(name: Optional[str]) -> Optional[Label]:
    if isinstance(name, str):
        if name.startswith("&"):
            name = name[1:]
        return Label.use_existing(name)


def _resolve_addr(raw_addr: Any) -> Optional[Union[str, Label]]:
    if raw_addr is None:
        return None
    if isinstance(raw_addr, Label):
        return raw_addr
    if isinstance(raw_addr, str):
        if is_register_addr(raw_addr):
            return raw_addr
        if raw_addr.startswith("&"):
            return _get_label(raw_addr)
        raise ValueError(
            f"Invalid ADDR {raw_addr!r}: plain string labels are not supported. "
            "Use register address (e.g. 's15') or '&label'."
        )
    raise ValueError(
        f"Invalid ADDR type {type(raw_addr).__name__}: expected register string, '&label', or Label."
    )


def _normalize_reg_wr_fields(source: dict[str, Any]) -> dict[str, Any]:
    d = dict(source)
    src = d.get("SRC", "")
    wr = d.get("WR")
    if not d.get("DST") and wr:
        wr_parts = wr.split()
        if wr_parts:
            d["DST"] = wr_parts[0]
        if len(wr_parts) > 1 and not src:
            d["SRC"] = wr_parts[1]
    return d


@dataclass(frozen=True)
class Instruction(ABC):
    """Base class for all IR instructions."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> Instruction: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert back to QICK prog_list dict format."""
        ...

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.to_dict().items() if v and k not in ('CMD')])})"


@dataclass(frozen=True)
class BaseInst(Instruction):
    """Base class for real machine instructions that occupy program memory."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BaseInst:
        cmd = d.get("CMD")
        if not cmd:
            raise ValueError(f"Unknown instruction format: {d}")

        if cmd == "REG_WR":
            norm_d = _normalize_reg_wr_fields(d)
            if norm_d.get("SRC", "") == "dmem":
                return DmemReadInst.from_dict(norm_d)
            return RegWriteInst.from_dict(norm_d)

        dispatch: dict[str, type[BaseInst]] = {
            "TIME": TimeInst,
            "TEST": TestInst,
            "JUMP": JumpInst,
            "WPORT_WR": PortWriteInst,
            "NOP": NopInst,
            "DMEM_WR": DmemWriteInst,
            "WMEM_WR": WmemWriteInst,
            "DPORT_WR": DportWriteInst,
            "WAIT": WaitInst,
        }
        inst_cls = dispatch.get(cmd)
        if inst_cls is None:
            raise ValueError(f"Unknown instruction opcode: {cmd!r}")
        return inst_cls.from_dict(d)

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
    def need_label(self) -> Optional[Label]:
        """Label name this instruction depends on (e.g. for JUMP or WR_ADDR)."""
        return None


@dataclass(frozen=True)
class LabelInst(Instruction):
    name: Label
    can_remove: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelInst:
        if d.get("kind") != "label":
            raise ValueError(f"Invalid LabelInst format: {d}")

        return cls(
            name=Label.use_existing(d["name"]),
            can_remove=bool(d.get("can_remove", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "label",
            "name": str(self.name),
            "can_remove": self.can_remove,
        }


@dataclass(frozen=True)
class MetaInst(Instruction):
    """Meta instruction used for structural control like loops."""

    type: str = ""
    name: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetaInst:
        if d.get("kind") == "meta":
            return cls(
                type=d.get("type", ""),
                name=d.get("name", ""),
                info=d.get("info", {}),
            )
        if d.get("CMD") == "__META__":
            return cls(
                type=d.get("TYPE", ""),
                name=d.get("NAME", ""),
                info=d.get("INFO", {}),
            )
        raise ValueError(f"Invalid MetaInst format: {d}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "meta",
            "type": self.type,
            "name": self.name,
            "info": self.info,
        }


@dataclass(frozen=True)
class TimeInst(BaseInst):
    """TIME instruction: advance timing counter."""

    c_op: str
    lit: Optional[str] = None  # e.g., "#10" for literal value
    r1: Optional[str] = None  # register operand

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimeInst:
        return cls(
            c_op=d.get("C_OP", ""),
            lit=d.get("LIT"),
            r1=d.get("R1"),
        )

    @property
    def reg_read(self) -> list[str]:
        return sorted(list(regs_from_value(self.r1)))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "TIME",
            "C_OP": self.c_op,
            "LIT": self.lit,
            "R1": self.r1,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class TestInst(BaseInst):
    """TEST instruction: evaluate condition for conditional branch."""

    __test__ = False
    op: str
    uf: Optional[str] = None  # Overflow/underflow flag (usually "1")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TestInst:
        return cls(
            op=d.get("OP", ""),
            uf=d.get("UF"),
        )

    @property
    def reg_read(self) -> list[str]:
        return sorted(list(regs_from_value(self.op)))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "TEST",
            "OP": self.op,
            "UF": self.uf,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class JumpInst(BaseInst):
    """JUMP instruction: unconditional or conditional jump."""

    label: Optional[Label] = None  # Target label
    if_cond: Optional[str] = None  # Condition for conditional jump (e.g., "eq", "nz")
    addr: Optional[Union[str, Label]] = (
        None  # Direct address for large jumps (e.g., "s15" or label)
    )
    wr: Optional[str] = None  # Optional register write control string
    op: Optional[str] = None  # Optional ALU expression
    uf: Optional[str] = None  # Optional flag update control

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JumpInst:
        return cls(
            label=_get_label(d.get("LABEL")),
            if_cond=d.get("IF"),
            addr=_resolve_addr(d.get("ADDR")),
            wr=d.get("WR"),
            op=d.get("OP"),
            uf=d.get("UF"),
        )

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
    def need_label(self) -> Optional[Label]:
        if self.label and not _is_pseudo_label(self.label):
            return self.label

        if isinstance(self.addr, Label) and not _is_pseudo_label(self.addr):
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "JUMP",
            "LABEL": str(self.label) if self.label else None,
            "ADDR": _serialize_addr(self.addr),
            "IF": self.if_cond,
            "WR": self.wr,
            "OP": self.op,
            "UF": self.uf,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class RegWriteInst(BaseInst):
    """REG_WR instruction: write to register."""

    dst: str  # Destination register
    src: Optional[str] = None  # Source: 'op', 'imm', 'reg', 'dmem'
    op: Optional[str] = None
    lit: Optional[str] = None
    addr: Optional[Union[str, Label]] = None
    uf: Optional[str] = None
    wr: Optional[str] = None
    if_cond: Optional[str] = None
    label: Optional[Label] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegWriteInst:
        norm_d = _normalize_reg_wr_fields(d)
        return cls(
            dst=norm_d.get("DST", ""),
            src=norm_d.get("SRC"),
            wr=norm_d.get("WR"),
            op=norm_d.get("OP"),
            lit=norm_d.get("LIT"),
            addr=_resolve_addr(norm_d.get("ADDR")),
            uf=norm_d.get("UF"),
            if_cond=norm_d.get("IF"),
            label=_get_label(norm_d.get("LABEL")),
        )

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="RegWriteInst.addr")

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        reads.update(regs_from_value(self.op))
        if isinstance(self.addr, str):
            reads.update(regs_from_value(self.addr))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return [strip_write_modifier(self.dst)]

    @property
    def need_label(self) -> Optional[Label]:
        if self.label and not _is_pseudo_label(self.label):
            return self.label

        if isinstance(self.addr, Label) and not _is_pseudo_label(self.addr):
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "REG_WR",
            "DST": self.dst,
            "SRC": self.src,
            "WR": self.wr,
            "OP": self.op,
            "LIT": self.lit,
            "ADDR": _serialize_addr(self.addr),
            "UF": self.uf,
            "IF": self.if_cond,
            "LABEL": str(self.label) if self.label else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class PortWriteInst(BaseInst):
    """WPORT_WR instruction: write to output port."""

    dst: str  # Destination (output port)
    src: Optional[str] = None
    addr: Optional[str] = None
    time: Optional[str] = None  # Timing reference
    wr: Optional[str] = None
    op: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    data: Optional[str] = None
    phase: Optional[str] = None
    freq: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortWriteInst:
        return cls(
            dst=d.get("DST", ""),
            src=d.get("SRC"),
            addr=d.get("ADDR"),
            time=d.get("TIME"),
            wr=d.get("WR"),
            op=d.get("OP"),
            uf=d.get("UF"),
            if_cond=d.get("IF"),
            data=d.get("DATA"),
            phase=d.get("PHASE"),
            freq=d.get("FREQ"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        reads.update(regs_from_value(self.addr))
        reads.update(regs_from_value(self.time))
        reads.update(regs_from_value(self.op))
        reads.update(regs_from_value(self.wr))
        reads.update(regs_from_value(self.data))
        reads.update(regs_from_value(self.phase))
        reads.update(regs_from_value(self.freq))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        dst = strip_write_modifier(self.dst)
        # DST can be a plain port number (e.g. "2") or a register (e.g. "p1").
        # Only return it as a register write when it is not a bare integer.
        return [dst] if dst and not dst.lstrip("-").isdigit() else []

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "WPORT_WR",
            "DST": self.dst,
            "SRC": self.src,
            "ADDR": self.addr,
            "TIME": self.time,
            "WR": self.wr,
            "OP": self.op,
            "UF": self.uf,
            "IF": self.if_cond,
            "DATA": self.data,
            "PHASE": self.phase,
            "FREQ": self.freq,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class NopInst(BaseInst):
    """NOP instruction: no operation."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NopInst:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"CMD": "NOP"}


@dataclass(frozen=True)
class DmemReadInst(BaseInst):
    """DMEM read lowered to the native REG_WR form."""

    dst: str  # Destination register
    src: str = "dmem"
    addr: Optional[Union[str, Label]] = None  # Memory address (register or label)
    wr: Optional[str] = None
    op: Optional[str] = None
    lit: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    label: Optional[Label] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemReadInst:
        norm_d = _normalize_reg_wr_fields(d)
        return cls(
            dst=norm_d.get("DST", ""),
            src=norm_d.get("SRC", "dmem"),
            addr=_resolve_addr(norm_d.get("ADDR")),
            wr=norm_d.get("WR"),
            op=norm_d.get("OP"),
            lit=norm_d.get("LIT"),
            uf=norm_d.get("UF"),
            if_cond=norm_d.get("IF"),
            label=_get_label(norm_d.get("LABEL")),
        )

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="DmemReadInst.addr")

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        if isinstance(self.addr, str):
            reads.update(regs_from_value(self.addr))
        reads.update(regs_from_value(self.op))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return [strip_write_modifier(self.dst)]

    @property
    def need_label(self) -> Optional[Label]:
        if self.label and not _is_pseudo_label(self.label):
            return self.label

        if isinstance(self.addr, Label) and not _is_pseudo_label(self.addr):
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "REG_WR",
            "DST": self.dst,
            "SRC": self.src,
            "ADDR": _serialize_addr(self.addr),
            "WR": self.wr,
            "OP": self.op,
            "LIT": self.lit,
            "UF": self.uf,
            "IF": self.if_cond,
            "LABEL": str(self.label) if self.label else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DmemWriteInst(BaseInst):
    """DMEM_WR instruction: write to data memory."""

    dst: str  # Memory destination
    src: str  # Source (register or literal)
    op: Optional[str] = None
    lit: Optional[str] = None
    wr: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemWriteInst:
        return cls(
            dst=d.get("DST", ""),
            src=d.get("SRC", ""),
            wr=d.get("WR"),
            op=d.get("OP"),
            lit=d.get("LIT"),
            uf=d.get("UF"),
            if_cond=d.get("IF"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        reads.update(regs_from_value(self.op))
        return sorted(list(reads))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DMEM_WR",
            "DST": self.dst,
            "SRC": self.src,
            "OP": self.op,
            "LIT": self.lit,
            "WR": self.wr,
            "UF": self.uf,
            "IF": self.if_cond,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class WmemWriteInst(BaseInst):
    """WMEM_WR instruction: write wave registers into wave memory."""

    addr: Optional[str] = None
    time: Optional[str] = None
    wr: Optional[str] = None
    op: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    wp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WmemWriteInst:
        return cls(
            addr=d.get("ADDR"),
            time=d.get("TIME"),
            wr=d.get("WR"),
            op=d.get("OP"),
            uf=d.get("UF"),
            if_cond=d.get("IF"),
            wp=d.get("WP"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.addr))
        reads.update(regs_from_value(self.time))
        reads.update(regs_from_value(self.op))
        reads.update(regs_from_value(self.wr))
        reads.update(regs_from_value(self.wp))
        return sorted(list(reads))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "WMEM_WR",
            "ADDR": self.addr,
            "TIME": self.time,
            "WR": self.wr,
            "OP": self.op,
            "UF": self.uf,
            "IF": self.if_cond,
            "WP": self.wp,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DportWriteInst(BaseInst):
    """DPORT_WR instruction: write to data port."""

    dst: str  # Destination (port)
    src: Optional[str] = None
    data: str = ""  # Data to write (register or literal)
    time: Optional[str] = None
    wr: Optional[str] = None
    op: Optional[str] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DportWriteInst:
        return cls(
            dst=d.get("DST", ""),
            src=d.get("SRC"),
            data=d.get("DATA", ""),
            time=d.get("TIME"),
            wr=d.get("WR"),
            op=d.get("OP"),
            uf=d.get("UF"),
            if_cond=d.get("IF"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        reads.update(regs_from_value(self.dst))
        reads.update(regs_from_value(self.data))
        reads.update(regs_from_value(self.time))
        reads.update(regs_from_value(self.op))
        reads.update(regs_from_value(self.wr))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        # DPORT_WR usually writes to a port, but if DST is a register...
        # In QICK DPORT_WR, DST is usually a literal port number.
        return [strip_write_modifier(self.dst)] if not self.dst.startswith("#") else []

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DPORT_WR",
            "DST": self.dst,
            "SRC": self.src,
            "DATA": self.data,
            "TIME": self.time,
            "WR": self.wr,
            "OP": self.op,
            "UF": self.uf,
            "IF": self.if_cond,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class WaitInst(BaseInst):
    """WAIT instruction: wait for sync/trigger."""

    c_op: str = "time"
    time: Optional[str] = None
    addr: Optional[Union[str, Label]] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WaitInst:
        return cls(
            c_op=d.get("C_OP", "time"),
            time=d.get("TIME"),
            addr=_resolve_addr(d.get("ADDR")),
        )

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="WaitInst.addr")

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.time))
        if isinstance(self.addr, str):
            reads.update(regs_from_value(self.addr))
        return sorted(list(reads))

    @property
    def need_label(self) -> Optional[Label]:
        if isinstance(self.addr, Label) and not _is_pseudo_label(self.addr):
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "WAIT",
            "C_OP": self.c_op,
            "TIME": self.time,
            "ADDR": _serialize_addr(self.addr),
        }
        return {k: v for k, v in d.items() if v is not None}

    @property
    def addr_inc(self) -> int:
        return 2
