from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Optional, Union

from .hw_semantics import STATUS_REG, TIMED_BASE_REG, USR_TIME_REG, WAVE_BUNDLE
from .labels import Label
from .operands import (
    AddrType,
    AluExpr,
    ExprType,
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    SideWrite,
    SrcKeyword,
    SrcType,
    TimeType,
    parse_addr,
    parse_alu_expr,
    parse_imm_value,
    parse_immediate,
    parse_label,
    parse_mem_addr,
    parse_register,
    parse_side_write,
    parse_src,
    parse_time,
    parse_value,
)

CondCode = str
UpdateFlag = str


def _require_register(val: str, field: str) -> Register:
    reg = parse_register(val)
    if reg is None:
        raise ValueError(f"{field}: {val!r} is not a valid register name")
    return reg


def _require_alu_expr(val: str, field: str) -> AluExpr:
    expr = parse_alu_expr(val)
    if expr is None:
        raise ValueError(f"{field}: {val!r} is not a valid ALU expression")
    return expr


def _parse_port_dst(val: str) -> Union[Register, ImmValue]:
    """Parse a port DST: either a register or a bare integer (port number)."""
    reg = parse_register(val)
    if reg is not None:
        return reg
    imm = parse_imm_value(val)
    if imm is not None:
        return imm
    raise ValueError(f"DST: {val!r} is not a register or port number")


def _parse_mem_addr_field(val: str, field: str) -> Union[Register, MemAddr]:
    """Parse an address field: register or &N numeric address."""
    reg = parse_register(val)
    if reg is not None:
        return reg
    addr = parse_mem_addr(val)
    if addr is not None:
        return addr
    raise ValueError(f"{field}: {val!r} is not a register or memory address")


def _parse_dmem_src(val: str) -> Union[Register, Immediate]:
    """Parse a DMEM_WR SRC: register or #N immediate."""
    reg = parse_register(val)
    if reg is not None:
        return reg
    imm = parse_immediate(val)
    if imm is not None:
        return imm
    raise ValueError(f"SRC: {val!r} is not a register or immediate value")



def _src_register_reads(src: Optional[SrcType]) -> set[str]:
    """If `src` names a register (not a keyword/literal), return its read regs."""
    if not src or isinstance(src, SrcKeyword):
        return set()
    if isinstance(src, Register):
        return src.get_read_regs()
    return set()





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
            if d.get("SRC") == "dmem":
                return DmemReadInst.from_dict(d)
            return RegWriteInst.from_dict(d)

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
        if d.get("kind") != "meta":
            raise ValueError(f"Invalid MetaInst format: {d}")
        return cls(
            type=d["type"],
            name=d["name"],
            info=d.get("info", {}),
        )

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
    lit: Optional[Immediate] = None
    r1: Optional[Register] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimeInst:
        return cls(
            c_op=d["C_OP"],
            lit=parse_immediate(d["LIT"]) if "LIT" in d else None,
            r1=parse_register(d["R1"]) if "R1" in d else None,
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
            op=_require_alu_expr(d["OP"], "OP"),
            uf=d.get("UF") or None,
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
            label=parse_label(d.get("LABEL")),
            if_cond=d.get("IF") or None,
            addr=parse_addr(d.get("ADDR")),
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf=d.get("UF") or None,
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
        if self.label and not self.label.is_pseudo_name():
            return self.label
        if isinstance(self.addr, Label) and not self.addr.is_pseudo_name():
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
    lit: Optional[Immediate] = None
    addr: Optional[AddrType] = None
    uf: Optional[UpdateFlag] = None
    wr: Optional[SideWrite] = None
    if_cond: Optional[CondCode] = None
    label: Optional[Label] = None
    ww: Optional[str] = None
    wp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegWriteInst:
        return cls(
            dst=_require_register(d["DST"], "DST"),
            src=parse_src(d.get("SRC")),
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            lit=parse_immediate(d["LIT"]) if "LIT" in d else None,
            addr=parse_addr(d.get("ADDR")),
            uf=d.get("UF") or None,
            if_cond=d.get("IF") or None,
            label=parse_label(d.get("LABEL")),
            ww=d.get("WW"),
            wp=d.get("WP"),
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
        if self.label and not self.label.is_pseudo_name():
            return self.label
        if isinstance(self.addr, Label) and not self.addr.is_pseudo_name():
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

    dst: Union[Register, ImmValue]
    src: Optional[SrcType] = None
    addr: Optional[Union[Register, MemAddr]] = None
    time: Optional[TimeType] = None
    wr: Optional[SideWrite] = None
    op: Optional[ExprType] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None
    ww: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortWriteInst:
        return cls(
            dst=_parse_port_dst(str(d["DST"])),
            src=parse_src(d.get("SRC")),
            addr=_parse_mem_addr_field(str(d["ADDR"]), "ADDR") if "ADDR" in d else None,
            time=parse_time(d["TIME"]) if "TIME" in d else None,
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf=d.get("UF") or None,
            if_cond=d.get("IF") or None,
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
    lit: Optional[Immediate] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None
    label: Optional[Label] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemReadInst:
        return cls(
            dst=_require_register(d["DST"], "DST"),
            src=SrcKeyword.DMEM,
            addr=parse_addr(d.get("ADDR")),
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            lit=parse_immediate(d["LIT"]) if "LIT" in d else None,
            uf=d.get("UF") or None,
            if_cond=d.get("IF") or None,
            label=parse_label(d.get("LABEL")),
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
        if self.label and not self.label.is_pseudo_name():
            return self.label
        if isinstance(self.addr, Label) and not self.addr.is_pseudo_name():
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

    dst: Union[Register, MemAddr]
    src: Union[Register, Immediate]
    op: Optional[ExprType] = None
    lit: Optional[Immediate] = None
    wr: Optional[SideWrite] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemWriteInst:
        return cls(
            dst=_parse_mem_addr_field(str(d["DST"]), "DST"),
            src=_parse_dmem_src(str(d["SRC"])),
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            lit=parse_immediate(d["LIT"]) if "LIT" in d else None,
            uf=d.get("UF") or None,
            if_cond=d.get("IF") or None,
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

    addr: Optional[Union[Register, MemAddr]] = None
    time: Optional[TimeType] = None
    wr: Optional[SideWrite] = None
    op: Optional[ExprType] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None
    wp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WmemWriteInst:
        return cls(
            addr=_parse_mem_addr_field(str(d["DST"]), "DST") if "DST" in d else None,
            time=parse_time(str(d["TIME"])) if "TIME" in d else None,
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf=d.get("UF") or None,
            if_cond=d.get("IF") or None,
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

    dst: Union[Register, ImmValue]
    src: Optional[SrcType] = None
    data: Union[Register, Immediate, ImmValue] = ImmValue(0)
    time: Optional[TimeType] = None
    wr: Optional[SideWrite] = None
    op: Optional[ExprType] = None
    uf: Optional[UpdateFlag] = None
    if_cond: Optional[CondCode] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DportWriteInst:
        data_raw = d.get("DATA")
        parsed_data: Union[Register, Immediate, ImmValue]
        if data_raw is None:
            parsed_data = ImmValue(0)
        else:
            result = parse_value(str(data_raw))
            if result is None:
                raise ValueError(f"DATA: {data_raw!r} is not a valid value")
            parsed_data = result
        return cls(
            dst=_parse_port_dst(str(d["DST"])),
            src=parse_src(d.get("SRC")),
            data=parsed_data,
            time=parse_time(str(d["TIME"])) if "TIME" in d else None,
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf=d.get("UF") or None,
            if_cond=d.get("IF") or None,
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
    time: Optional[TimeType] = None
    addr: Optional[AddrType] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WaitInst:
        return cls(
            c_op=d["C_OP"],
            time=parse_time(str(d["TIME"])) if "TIME" in d else None,
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
        if isinstance(self.addr, Label) and not self.addr.is_pseudo_name():
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
