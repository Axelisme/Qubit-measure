from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Optional, Union

from .labels import Label, is_pseudo_label_name, is_register_addr
from .operands import (
    AluExpr,
    Literal,
    Register,
    SideWrite,
    parse_alu_expr,
    parse_register_or_literal,
    parse_side_write,
)


def _is_pseudo_label(value: Optional[Label]) -> bool:
    if value is None:
        return False
    return is_pseudo_label_name(value.name)


def _serialize_addr(value: Optional[Union[Register, Literal, Label]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Label):
        return value.name if is_pseudo_label_name(value.name) else f"&{value.name}"
    return str(value)


def _validate_addr_value(
    value: Optional[Union[Register, Literal, Label]], *, field_name: str
) -> None:
    if value is None or isinstance(value, Label):
        return
    if isinstance(value, Register) and is_register_addr(value.name):
        return
    if isinstance(value, Literal):
        return
    raise ValueError(
        f"{field_name} address must be a Register, Literal, or Label, got {value!r}."
    )


def _validate_jump_addr_target(
    value: Optional[Union[Register, Literal, Label]],
) -> None:
    if value is None or isinstance(value, Label):
        return
    if isinstance(value, Register) and value.name == "s15":
        return
    raise ValueError(f"JumpInst.addr must be 's15' or Label, got {value!r}.")


def _get_label(name: Optional[str]) -> Optional[Label]:
    if isinstance(name, str):
        if name.startswith("&"):
            name = name[1:]
        return Label.use_existing(name)


def _resolve_addr(raw_addr: Any) -> Optional[Union[Register, Literal, Label]]:
    if raw_addr is None:
        return None
    if isinstance(raw_addr, Label):
        return raw_addr
    if isinstance(raw_addr, int):
        return Literal(f"&{raw_addr}")
    if isinstance(raw_addr, str):
        if is_register_addr(raw_addr):
            return Register(raw_addr)
        if raw_addr.startswith("&"):
            if raw_addr[1:].isdigit():
                return Literal(raw_addr)
            return _get_label(raw_addr)
        # It might be a direct literal address
        if raw_addr.startswith("@") or raw_addr.lstrip("-").isdigit():
            return Literal(raw_addr)
        raise ValueError(
            f"Invalid ADDR {raw_addr!r}: plain string labels are not supported. "
            "Use register address (e.g. 's15') or '&label'."
        )
    if isinstance(raw_addr, (Register, Literal)):
        return raw_addr
    raise ValueError(
        f"Invalid ADDR type {type(raw_addr).__name__}: expected Register, Literal, '&label', or Label."
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
    lit: Optional[Literal] = None  # e.g., "#10" for literal value
    r1: Optional[Register] = None  # register operand

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimeInst:
        return cls(
            c_op=d.get("C_OP", ""),
            lit=Literal(d["LIT"]) if "LIT" in d else None,
            r1=Register(d["R1"]) if "R1" in d else None,
        )

    @property
    def reg_read(self) -> list[str]:
        reads = set(self.r1.get_read_regs()) if self.r1 else set()
        if self.c_op == "updt":
            reads.add("s11")
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return ["s14"]

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "TIME",
            "C_OP": self.c_op,
            "LIT": str(self.lit) if self.lit else None,
            "R1": str(self.r1) if self.r1 else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class TestInst(BaseInst):
    """TEST instruction: evaluate condition for conditional branch."""

    __test__ = False
    op: AluExpr
    uf: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TestInst:
        return cls(
            op=parse_alu_expr(d.get("OP", "")),
            uf=d.get("UF"),
        )

    @property
    def reg_read(self) -> list[str]:
        return sorted(list(self.op.get_read_regs()))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "TEST",
            "OP": str(self.op),
            "UF": self.uf,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class JumpInst(BaseInst):
    """JUMP instruction: unconditional or conditional jump."""

    label: Optional[Label] = None
    if_cond: Optional[str] = None
    addr: Optional[Union[Register, Literal, Label]] = None
    wr: Optional[SideWrite] = None
    op: Optional[AluExpr] = None
    uf: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JumpInst:
        return cls(
            label=_get_label(d.get("LABEL")),
            if_cond=d.get("IF"),
            addr=_resolve_addr(d.get("ADDR")),
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf=d.get("UF"),
        )

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="JumpInst.addr")
        _validate_jump_addr_target(self.addr)

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        if isinstance(self.addr, Register):
            reads.update(self.addr.get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        if self.wr:
            reads.update(self.wr.get_read_regs())
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        if self.wr:
            return sorted(list(self.wr.get_write_regs()))
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
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": self.uf,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class RegWriteInst(BaseInst):
    """REG_WR instruction: write to register."""

    dst: Register
    src: Optional[str] = None
    op: Optional[AluExpr] = None
    lit: Optional[Literal] = None
    addr: Optional[Union[Register, Literal, Label]] = None
    uf: Optional[str] = None
    wr: Optional[SideWrite] = None
    if_cond: Optional[str] = None
    label: Optional[Label] = None
    ww: Optional[str] = None
    wp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegWriteInst:
        norm_d = _normalize_reg_wr_fields(d)
        return cls(
            dst=Register(norm_d.get("DST", "")),
            src=norm_d.get("SRC"),
            wr=parse_side_write(norm_d["WR"]) if "WR" in norm_d else None,
            op=parse_alu_expr(norm_d["OP"]) if "OP" in norm_d else None,
            lit=Literal(norm_d["LIT"]) if "LIT" in norm_d else None,
            addr=_resolve_addr(norm_d.get("ADDR")),
            uf=norm_d.get("UF"),
            if_cond=norm_d.get("IF"),
            label=_get_label(norm_d.get("LABEL")),
            ww=norm_d.get("WW"),
            wp=norm_d.get("WP"),
        )

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="RegWriteInst.addr")

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        if self.src == "wmem":
            reads.add("s14")
        if (
            self.src
            and not self.src.startswith("#")
            and self.src not in ("op", "imm", "label", "dmem", "wmem")
        ):
            reads.update(Register(self.src).get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        if isinstance(self.addr, Register):
            reads.update(self.addr.get_read_regs())
        # wp often looks like "r_wave p0" or just port num, skip adding port names to reads
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return sorted(list(self.dst.get_write_regs()))

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
            "DST": str(self.dst),
            "SRC": self.src,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "LIT": str(self.lit) if self.lit else None,
            "ADDR": _serialize_addr(self.addr),
            "UF": self.uf,
            "IF": self.if_cond,
            "LABEL": str(self.label) if self.label else None,
            "WW": self.ww,
            "WP": self.wp,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class PortWriteInst(BaseInst):
    """WPORT_WR instruction: write to output port."""

    dst: Union[Register, Literal]
    src: Optional[str] = None
    addr: Optional[Union[Register, Literal]] = None
    time: Optional[Union[Register, Literal]] = None
    wr: Optional[SideWrite] = None
    op: Optional[AluExpr] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    ww: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortWriteInst:
        return cls(
            dst=parse_register_or_literal(str(d.get("DST", ""))),
            src=d.get("SRC"),
            addr=parse_register_or_literal(d["ADDR"]) if "ADDR" in d else None,
            time=parse_register_or_literal(d["TIME"]) if "TIME" in d else None,
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf=d.get("UF"),
            if_cond=d.get("IF"),
            ww=d.get("WW"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = {"s14"}
        if self.src == "r_wave":
            # WPORT_WR r_wave reads all wave registers (w0-w5 aliased by r_wave)
            reads.update({"r_wave", "w0", "w1", "w2", "w3", "w4", "w5"})
        elif (
            self.src
            and not self.src.startswith("#")
            and self.src not in ("op", "imm", "wmem")
        ):
            reads.update(Register(self.src).get_read_regs())
        if isinstance(self.addr, Register):
            reads.update(self.addr.get_read_regs())
        if isinstance(self.time, Register):
            reads.update(self.time.get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        if self.wr:
            reads.update(self.wr.get_read_regs())
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        if isinstance(self.dst, Register):
            return sorted(list(self.dst.get_write_regs()))
        return []

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "WPORT_WR",
            "DST": str(self.dst),
            "SRC": self.src,
            "ADDR": str(self.addr) if self.addr else None,
            "TIME": str(self.time) if self.time else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": self.uf,
            "IF": self.if_cond,
            "WW": self.ww,
        }
        # In QICK, DST for port is often just integer, so we check and restore that format.
        # Handle string serialization for others
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

    dst: Register
    src: str = "dmem"
    addr: Optional[Union[Register, Literal, Label]] = None
    wr: Optional[SideWrite] = None
    op: Optional[AluExpr] = None
    lit: Optional[Literal] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    label: Optional[Label] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemReadInst:
        norm_d = _normalize_reg_wr_fields(d)
        return cls(
            dst=Register(norm_d.get("DST", "")),
            src=norm_d.get("SRC", "dmem"),
            addr=_resolve_addr(norm_d.get("ADDR")),
            wr=parse_side_write(norm_d["WR"]) if "WR" in norm_d else None,
            op=parse_alu_expr(norm_d["OP"]) if "OP" in norm_d else None,
            lit=Literal(norm_d["LIT"]) if "LIT" in norm_d else None,
            uf=norm_d.get("UF"),
            if_cond=norm_d.get("IF"),
            label=_get_label(norm_d.get("LABEL")),
        )

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="DmemReadInst.addr")

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        if isinstance(self.addr, Register):
            reads.update(self.addr.get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return sorted(list(self.dst.get_write_regs()))

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
            "DST": str(self.dst),
            "SRC": self.src,
            "ADDR": _serialize_addr(self.addr),
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "LIT": str(self.lit) if self.lit else None,
            "UF": self.uf,
            "IF": self.if_cond,
            "LABEL": str(self.label) if self.label else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DmemWriteInst(BaseInst):
    """DMEM_WR instruction: write to data memory."""

    dst: Union[Register, Literal]
    src: Union[Register, Literal]
    op: Optional[AluExpr] = None
    lit: Optional[Literal] = None
    wr: Optional[SideWrite] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemWriteInst:
        return cls(
            dst=parse_register_or_literal(str(d.get("DST", ""))),
            src=parse_register_or_literal(str(d.get("SRC", ""))),
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            lit=Literal(d["LIT"]) if "LIT" in d else None,
            uf=d.get("UF"),
            if_cond=d.get("IF"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        if isinstance(self.src, Register):
            reads.update(self.src.get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        return sorted(list(reads))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DMEM_WR",
            "DST": str(self.dst),
            "SRC": str(self.src),
            "OP": str(self.op) if self.op else None,
            "LIT": str(self.lit) if self.lit else None,
            "WR": str(self.wr) if self.wr else None,
            "UF": self.uf,
            "IF": self.if_cond,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class WmemWriteInst(BaseInst):
    """WMEM_WR instruction: write wave registers into wave memory."""

    addr: Optional[Union[Register, Literal]] = None
    time: Optional[Union[Register, Literal]] = None
    wr: Optional[SideWrite] = None
    op: Optional[AluExpr] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None
    wp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WmemWriteInst:
        # QICK uses 'DST' for WMEM_WR address
        addr_raw = d.get("DST", d.get("ADDR"))
        return cls(
            addr=parse_register_or_literal(str(addr_raw))
            if addr_raw is not None
            else None,
            time=parse_register_or_literal(str(d["TIME"])) if "TIME" in d else None,
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf=d.get("UF"),
            if_cond=d.get("IF"),
            wp=d.get("WP"),
        )
    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = {"w0", "w1", "w2", "w3", "w4", "w5", "r_wave", "s14"}
        if isinstance(self.addr, Register):
            reads.update(self.addr.get_read_regs())
        if isinstance(self.time, Register):
            reads.update(self.time.get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        if self.wr:
            reads.update(self.wr.get_read_regs())
        return sorted(list(reads))
    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "WMEM_WR",
            "DST": str(self.addr) if self.addr else None,
            "TIME": str(self.time) if self.time else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": self.uf,
            "IF": self.if_cond,
            "WP": self.wp,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DportWriteInst(BaseInst):
    """DPORT_WR instruction: write to data port."""

    dst: Union[Register, Literal]
    src: Optional[str] = None
    data: Union[Register, Literal] = Literal("")
    time: Optional[Union[Register, Literal]] = None
    wr: Optional[SideWrite] = None
    op: Optional[AluExpr] = None
    uf: Optional[str] = None
    if_cond: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DportWriteInst:
        return cls(
            dst=parse_register_or_literal(str(d.get("DST", ""))),
            src=d.get("SRC"),
            data=parse_register_or_literal(str(d.get("DATA", ""))),
            time=parse_register_or_literal(str(d["TIME"])) if "TIME" in d else None,
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf=d.get("UF"),
            if_cond=d.get("IF"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = {"s14"}
        if self.src and not self.src.startswith("#"):
            reads.update(Register(self.src).get_read_regs())
        if isinstance(self.dst, Register):
            reads.update(self.dst.get_read_regs())
        if isinstance(self.data, Register):
            reads.update(self.data.get_read_regs())
        if isinstance(self.time, Register):
            reads.update(self.time.get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        if self.wr:
            reads.update(self.wr.get_read_regs())
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        if isinstance(self.dst, Register):
            return sorted(list(self.dst.get_write_regs()))
        return []

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DPORT_WR",
            "DST": str(self.dst),
            "SRC": self.src,
            "DATA": str(self.data),
            "TIME": str(self.time) if self.time else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": self.uf,
            "IF": self.if_cond,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class WaitInst(BaseInst):
    """WAIT instruction: wait for sync/trigger."""

    c_op: str = "time"
    time: Optional[Union[Register, Literal]] = None
    addr: Optional[Union[Register, Literal, Label]] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WaitInst:
        return cls(
            c_op=d.get("C_OP", "time"),
            time=parse_register_or_literal(str(d["TIME"])) if "TIME" in d else None,
            addr=_resolve_addr(d.get("ADDR")),
        )

    def __post_init__(self) -> None:
        _validate_addr_value(self.addr, field_name="WaitInst.addr")

    @property
    def reg_read(self) -> list[str]:
        # WAIT time: TEST (s11 - (TIME - OFFSET)), reads s11 and implicitly uses s14
        # WAIT port_dt/div_rdy/div_dt/qpa_*: TEST (s10 AND #mask), reads s10 only
        if self.c_op == "time":
            reads: set[str] = {"s11", "s14"}
        else:
            reads = {"s10"}
        if isinstance(self.time, Register):
            reads.update(self.time.get_read_regs())
        if isinstance(self.addr, Register):
            reads.update(self.addr.get_read_regs())
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
            "TIME": str(self.time) if self.time else None,
            "ADDR": _serialize_addr(self.addr),
        }
        return {k: v for k, v in d.items() if v is not None}

    @property
    def addr_inc(self) -> int:
        return 2
