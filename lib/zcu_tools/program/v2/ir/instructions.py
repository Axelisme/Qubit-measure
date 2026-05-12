from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Optional, Union

from .hw_semantics import STATUS_REG, TIMED_BASE_REG, USR_TIME_REG, WAVE_BUNDLE
from .labels import Label, is_pseudo_label_name
from .operands import (
    AddrType,
    AluExpr,
    ExprType,
    ImmValue,
    Register,
    SideWrite,
    SrcKeyword,
    SrcType,
    ValueType,
    parse_addr,
    parse_src,
    parse_value,
)

CondCode = str
UpdateFlag = str


def _parse_cond_code(val: Any) -> Optional[CondCode]:
    if not val:
        return None
    return str(val)


def _parse_update_flag(val: Any) -> Optional[UpdateFlag]:
    if not val:
        return None
    return str(val)


def _src_register_reads(src: Optional[SrcType]) -> set[str]:
    """If `src` names a register (not a keyword/literal), return its read regs."""
    if not src or isinstance(src, SrcKeyword):
        return set()
    if isinstance(src, Register):
        return src.get_read_regs()
    return set()


def _is_pseudo_label(value: Optional[Label]) -> bool:
    if value is None:
        return False
    return is_pseudo_label_name(value.name)


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
        fields = []
        for k, v in self.to_dict().items():
            if v is not None and k not in ("CMD",):
                fields.append(f"{k}={v}")
        return f"{self.__class__.__name__}({', '.join(fields)})"


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
            "name": self.name.name,
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
    lit: Optional[ImmValue] = None
    r1: Optional[Register] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimeInst:
        return cls(
            c_op=d.get("C_OP", ""),
            lit=ImmValue.parse(d["LIT"]) if "LIT" in d else None,
            r1=Register.parse(d["R1"]) if "R1" in d else None,
        )

    @property
    def reg_read(self) -> list[str]:
        reads = set(self.r1.get_read_regs()) if self.r1 else set()
        if self.c_op == "updt":
            reads.add(USR_TIME_REG)
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return [TIMED_BASE_REG]

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
    op: ExprType
    uf: Optional[UpdateFlag] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TestInst:
        return cls(
            op=AluExpr.parse(d.get("OP", "")),
            uf=_parse_update_flag(d.get("UF")),
        )

    @property
    def reg_read(self) -> list[str]:
        return sorted(list(self.op.get_read_regs())) if self.op else []

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "TEST",
            "OP": str(self.op) if self.op else None,
            "UF": self.uf if self.uf else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class JumpInst(BaseInst):
    """JUMP instruction: unconditional or conditional jump."""

    label: Optional[Label] = None
    if_cond: Optional[CondCode] = None
    addr: Optional[AddrType] = None
    wr: Optional[SideWrite] = None
    op: Optional[ExprType] = None
    uf: Optional[UpdateFlag] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JumpInst:
        return cls(
            label=Label.parse(d.get("LABEL")),
            if_cond=_parse_cond_code(d.get("IF")),
            addr=parse_addr(d.get("ADDR")),
            wr=SideWrite.parse(d["WR"]) if "WR" in d else None,
            op=AluExpr.parse(d["OP"]) if "OP" in d else None,
            uf=_parse_update_flag(d.get("UF")),
        )

    def __post_init__(self) -> None:
        if self.addr is not None and not isinstance(self.addr, Label):
            if not (isinstance(self.addr, Register) and self.addr.name == "s15"):
                raise ValueError(
                    f"JumpInst.addr must be 's15' or Label, got {self.addr!r}."
                )

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
            "LABEL": self.label.name if self.label else None,
            "ADDR": str(self.addr) if self.addr else None,
            "IF": self.if_cond if self.if_cond else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": self.uf if self.uf else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class RegWriteInst(BaseInst):
    """REG_WR instruction: write to register."""

    dst: Register
    src: Optional[SrcType] = None
    op: Optional[ExprType] = None
    lit: Optional[ImmValue] = None
    addr: Optional[AddrType] = None
    uf: Optional[UpdateFlag] = None
    wr: Optional[SideWrite] = None
    if_cond: Optional[CondCode] = None
    label: Optional[Label] = None
    ww: Optional[str] = None
    wp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegWriteInst:
        norm_d = _normalize_reg_wr_fields(d)
        return cls(
            dst=Register.parse(norm_d.get("DST", "")),
            src=parse_src(norm_d.get("SRC")),
            wr=SideWrite.parse(norm_d["WR"]) if "WR" in norm_d else None,
            op=AluExpr.parse(norm_d["OP"]) if "OP" in norm_d else None,
            lit=ImmValue.parse(norm_d["LIT"]) if "LIT" in norm_d else None,
            addr=parse_addr(norm_d.get("ADDR")),
            uf=_parse_update_flag(norm_d.get("UF")),
            if_cond=_parse_cond_code(norm_d.get("IF")),
            label=Label.parse(norm_d.get("LABEL")),
            ww=norm_d.get("WW"),
            wp=norm_d.get("WP"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        if self.src == SrcKeyword.WMEM:
            reads.add(TIMED_BASE_REG)
        reads |= _src_register_reads(self.src)
        if self.op:
            reads.update(self.op.get_read_regs())
        if isinstance(self.addr, Register):
            reads.update(self.addr.get_read_regs())
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return sorted(list(self.dst.get_write_regs())) if self.dst else []

    @property
    def need_label(self) -> Optional[Label]:
        if self.label and not _is_pseudo_label(self.label):
            return self.label
        if isinstance(self.addr, Label) and not _is_pseudo_label(self.addr):
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        src_val = (
            self.src.value
            if isinstance(self.src, SrcKeyword)
            else str(self.src)
            if self.src
            else None
        )
        d = {
            "CMD": "REG_WR",
            "DST": str(self.dst) if self.dst else None,
            "SRC": src_val,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "LIT": str(self.lit) if self.lit else None,
            "ADDR": str(self.addr) if self.addr else None,
            "UF": self.uf if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
            "LABEL": self.label.name if self.label else None,
            "WW": self.ww,
            "WP": self.wp,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class PortWriteInst(BaseInst):
    """WPORT_WR instruction: write to output port."""

    dst: ValueType
    src: Optional[SrcType] = None
    addr: Optional[ValueType] = None
    time: Optional[ValueType] = None
    wr: Optional[SideWrite] = None
    op: Optional[ExprType] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None
    ww: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortWriteInst:
        return cls(
            dst=parse_value(str(d.get("DST", ""))),
            src=parse_src(d.get("SRC")),
            addr=parse_value(d["ADDR"]) if "ADDR" in d else None,
            time=parse_value(d["TIME"]) if "TIME" in d else None,
            wr=SideWrite.parse(d["WR"]) if "WR" in d else None,
            op=AluExpr.parse(d["OP"]) if "OP" in d else None,
            uf=_parse_update_flag(d.get("UF")),
            if_cond=_parse_cond_code(d.get("IF")),
            ww=d.get("WW"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = {TIMED_BASE_REG}
        if isinstance(self.src, Register) and self.src.name == "r_wave":
            reads |= WAVE_BUNDLE
        else:
            reads |= _src_register_reads(self.src)

        for op in [self.dst, self.addr, self.time]:
            if isinstance(op, Register):
                reads.update(op.get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        if self.wr:
            reads.update(self.wr.get_read_regs())
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return []

    def to_dict(self) -> dict[str, Any]:
        src_val = (
            self.src.value
            if isinstance(self.src, SrcKeyword)
            else str(self.src)
            if self.src
            else None
        )
        d = {
            "CMD": "WPORT_WR",
            "DST": str(self.dst) if self.dst else None,
            "SRC": src_val,
            "ADDR": str(self.addr) if self.addr else None,
            "TIME": str(self.time) if self.time else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": self.uf if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
            "WW": self.ww,
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

    dst: Register
    src: SrcKeyword = SrcKeyword.DMEM
    addr: Optional[AddrType] = None
    wr: Optional[SideWrite] = None
    op: Optional[ExprType] = None
    lit: Optional[ImmValue] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None
    label: Optional[Label] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemReadInst:
        norm_d = _normalize_reg_wr_fields(d)
        src_val = parse_src(norm_d.get("SRC", "dmem"))
        if not isinstance(src_val, SrcKeyword):
            src_val = SrcKeyword.DMEM
        return cls(
            dst=Register.parse(norm_d.get("DST", "")),
            src=src_val,
            addr=parse_addr(norm_d.get("ADDR")),
            wr=SideWrite.parse(norm_d["WR"]) if "WR" in norm_d else None,
            op=AluExpr.parse(norm_d["OP"]) if "OP" in norm_d else None,
            lit=ImmValue.parse(norm_d["LIT"]) if "LIT" in norm_d else None,
            uf=_parse_update_flag(norm_d.get("UF")),
            if_cond=_parse_cond_code(norm_d.get("IF")),
            label=Label.parse(norm_d.get("LABEL")),
        )

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
        return sorted(list(self.dst.get_write_regs())) if self.dst else []

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
            "DST": str(self.dst) if self.dst else None,
            "SRC": self.src.value,
            "ADDR": str(self.addr) if self.addr else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "LIT": str(self.lit) if self.lit else None,
            "UF": self.uf if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
            "LABEL": self.label.name if self.label else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DmemWriteInst(BaseInst):
    """DMEM_WR instruction: write to data memory."""

    dst: ValueType
    src: ValueType
    op: Optional[ExprType] = None
    lit: Optional[ImmValue] = None
    wr: Optional[SideWrite] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemWriteInst:
        return cls(
            dst=parse_value(str(d.get("DST", ""))),
            src=parse_value(str(d.get("SRC", ""))),
            wr=SideWrite.parse(d["WR"]) if "WR" in d else None,
            op=AluExpr.parse(d["OP"]) if "OP" in d else None,
            lit=ImmValue.parse(d["LIT"]) if "LIT" in d else None,
            uf=_parse_update_flag(d.get("UF")),
            if_cond=_parse_cond_code(d.get("IF")),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        for op in [self.dst, self.src]:
            if isinstance(op, Register):
                reads.update(op.get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        return sorted(list(reads))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DMEM_WR",
            "DST": str(self.dst) if self.dst else None,
            "SRC": str(self.src) if self.src else None,
            "OP": str(self.op) if self.op else None,
            "LIT": str(self.lit) if self.lit else None,
            "WR": str(self.wr) if self.wr else None,
            "UF": self.uf if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class WmemWriteInst(BaseInst):
    """WMEM_WR instruction: write wave registers into wave memory."""

    addr: Optional[ValueType] = None
    time: Optional[ValueType] = None
    wr: Optional[SideWrite] = None
    op: Optional[ExprType] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None
    wp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WmemWriteInst:
        addr_raw = d.get("DST", d.get("ADDR"))
        return cls(
            addr=parse_value(str(addr_raw)) if addr_raw is not None else None,
            time=parse_value(str(d["TIME"])) if "TIME" in d else None,
            wr=SideWrite.parse(d["WR"]) if "WR" in d else None,
            op=AluExpr.parse(d["OP"]) if "OP" in d else None,
            uf=_parse_update_flag(d.get("UF")),
            if_cond=_parse_cond_code(d.get("IF")),
            wp=d.get("WP"),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set(WAVE_BUNDLE) | {TIMED_BASE_REG}
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
            "UF": self.uf if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
            "WP": self.wp,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DportWriteInst(BaseInst):
    """DPORT_WR instruction: write to data port."""

    dst: ValueType
    src: Optional[SrcType] = None
    data: ValueType = ImmValue(0, prefix="")
    time: Optional[ValueType] = None
    wr: Optional[SideWrite] = None
    op: Optional[ExprType] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DportWriteInst:
        data_val = d.get("DATA", "")
        parsed_data = (
            parse_value(str(data_val)) if data_val != "" else ImmValue(0, prefix="")
        )
        if parsed_data is None:
            parsed_data = ImmValue(0, prefix="")
        return cls(
            dst=parse_value(str(d.get("DST", ""))),
            src=parse_src(d.get("SRC")),
            data=parsed_data,
            time=parse_value(str(d["TIME"])) if "TIME" in d else None,
            wr=SideWrite.parse(d["WR"]) if "WR" in d else None,
            op=AluExpr.parse(d["OP"]) if "OP" in d else None,
            uf=_parse_update_flag(d.get("UF")),
            if_cond=_parse_cond_code(d.get("IF")),
        )

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = {TIMED_BASE_REG}
        reads |= _src_register_reads(self.src)

        for op in [self.dst, self.data, self.time]:
            if isinstance(op, Register):
                reads.update(op.get_read_regs())
        if self.op:
            reads.update(self.op.get_read_regs())
        if self.wr:
            reads.update(self.wr.get_read_regs())
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return []

    def to_dict(self) -> dict[str, Any]:
        src_val = (
            self.src.value
            if isinstance(self.src, SrcKeyword)
            else str(self.src)
            if self.src
            else None
        )
        d = {
            "CMD": "DPORT_WR",
            "DST": str(self.dst) if self.dst else None,
            "SRC": src_val,
            "DATA": str(self.data),
            "TIME": str(self.time) if self.time else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": self.uf if self.uf else None,
            "IF": self.if_cond if self.if_cond else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class WaitInst(BaseInst):
    """WAIT instruction: wait for sync/trigger."""

    c_op: str = "time"
    time: Optional[ValueType] = None
    addr: Optional[AddrType] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WaitInst:
        return cls(
            c_op=d.get("C_OP", "time"),
            time=parse_value(str(d["TIME"])) if "TIME" in d else None,
            addr=parse_addr(d.get("ADDR")),
        )

    @property
    def reg_read(self) -> list[str]:
        if self.c_op == "time":
            reads: set[str] = {USR_TIME_REG, TIMED_BASE_REG}
        else:
            reads = {STATUS_REG}
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
            "ADDR": str(self.addr) if self.addr else None,
        }
        return {k: v for k, v in d.items() if v is not None}

    @property
    def addr_inc(self) -> int:
        return 2
