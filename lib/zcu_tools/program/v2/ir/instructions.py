from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from typing_extensions import Any, Optional, Union

from .hw_semantics import STATUS_REG, TIMED_BASE_REG, USR_TIME_REG
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

CondCode = Literal["Z", "S", "NZ", "NS"]
_VALID_COND_CODES = frozenset({"Z", "S", "NZ", "NS"})


def _parse_cond_code(val: Optional[str]) -> Optional[CondCode]:
    if val is None:
        return None
    if val not in _VALID_COND_CODES:
        raise ValueError(
            f"IF: {val!r} is not a valid condition code, expected one of {sorted(_VALID_COND_CODES)}"
        )
    return val  # type: ignore[return-value]


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
    def reg_read(self) -> frozenset[str]:
        """Registers read by this instruction."""
        return frozenset()

    @property
    def reg_write(self) -> frozenset[str]:
        """Registers written by this instruction."""
        return frozenset()

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
            name=Label(d["name"]),
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
    def reg_read(self) -> frozenset[str]:
        reads = self.r1.regs() if self.r1 else frozenset()
        if self.c_op == "updt":
            reads = reads | {USR_TIME_REG}
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return frozenset({TIMED_BASE_REG})

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
    uf: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TestInst:
        return cls(
            op=_require_alu_expr(d["OP"], "OP"),
            uf="UF" in d,
        )

    @property
    def reg_read(self) -> frozenset[str]:
        return self.op.regs() if self.op else frozenset()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "TEST",
            "OP": str(self.op) if self.op else None,
            "UF": "1" if self.uf else None,
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
    uf: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JumpInst:
        return cls(
            label=parse_label(d.get("LABEL")),
            if_cond=_parse_cond_code(d.get("IF")),
            addr=parse_addr(d.get("ADDR")),
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf="UF" in d,
        )

    def __post_init__(self) -> None:
        if self.addr is not None and not isinstance(self.addr, Label):
            if self.addr != Register("s15"):
                raise ValueError(
                    f"JumpInst.addr must be 's15' or Label, got {self.addr!r}."
                )

    @property
    def reg_read(self) -> frozenset[str]:
        reads = frozenset[str]()
        if isinstance(self.addr, Register):
            reads = reads | self.addr.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return self.wr.regs() if self.wr else frozenset()

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
            "UF": "1" if self.uf else None,
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
    uf: bool = False
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
            uf="UF" in d,
            if_cond=_parse_cond_code(d.get("IF")),
            label=parse_label(d.get("LABEL")),
            ww=d.get("WW"),
            wp=d.get("WP"),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        if self.src == SrcKeyword.WMEM:
            reads = reads | {TIMED_BASE_REG}
        if isinstance(self.src, Register):
            reads = reads | self.src.regs()
        if self.op:
            reads = reads | self.op.regs()
        if isinstance(self.addr, Register):
            reads = reads | self.addr.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return self.dst.regs() if self.dst else frozenset()

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
            "UF": "1" if self.uf else None,
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
    uf: bool = False
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
            uf="UF" in d,
            if_cond=_parse_cond_code(d.get("IF")),
            ww=d.get("WW"),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset({TIMED_BASE_REG})
        if isinstance(self.src, Register):
            reads = reads | self.src.regs()
        for op in [self.dst, self.addr, self.time]:
            if isinstance(op, Register):
                reads = reads | op.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return frozenset()

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
            "UF": "1" if self.uf else None,
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
    uf: bool = False
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
            uf="UF" in d,
            if_cond=_parse_cond_code(d.get("IF")),
            label=parse_label(d.get("LABEL")),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        if isinstance(self.addr, Register):
            reads = reads | self.addr.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return self.dst.regs() if self.dst else frozenset()

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
            "UF": "1" if self.uf else None,
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
    uf: bool = False
    if_cond: Optional[CondCode] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemWriteInst:
        return cls(
            dst=_parse_mem_addr_field(str(d["DST"]), "DST"),
            src=_parse_dmem_src(str(d["SRC"])),
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            lit=parse_immediate(d["LIT"]) if "LIT" in d else None,
            uf="UF" in d,
            if_cond=_parse_cond_code(d.get("IF")),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        for op in [self.dst, self.src]:
            if isinstance(op, Register):
                reads = reads | op.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DMEM_WR",
            "DST": str(self.dst) if self.dst else None,
            "SRC": str(self.src) if self.src else None,
            "OP": str(self.op) if self.op else None,
            "LIT": str(self.lit) if self.lit else None,
            "WR": str(self.wr) if self.wr else None,
            "UF": "1" if self.uf else None,
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
    uf: bool = False
    if_cond: Optional[CondCode] = None
    wp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WmemWriteInst:
        return cls(
            addr=_parse_mem_addr_field(str(d["DST"]), "DST") if "DST" in d else None,
            time=parse_time(str(d["TIME"])) if "TIME" in d else None,
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            uf="UF" in d,
            if_cond=_parse_cond_code(d.get("IF")),
            wp=d.get("WP"),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads = Register("r_wave").regs() | frozenset({TIMED_BASE_REG})
        if isinstance(self.addr, Register):
            reads = reads | self.addr.regs()
        if isinstance(self.time, Register):
            reads = reads | self.time.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "WMEM_WR",
            "DST": str(self.addr) if self.addr else None,
            "TIME": str(self.time) if self.time else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": "1" if self.uf else None,
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
    uf: bool = False
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
            uf="UF" in d,
            if_cond=_parse_cond_code(d.get("IF")),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset({TIMED_BASE_REG})
        if isinstance(self.src, Register):
            reads = reads | self.src.regs()
        for op in [self.dst, self.data, self.time]:
            if isinstance(op, Register):
                reads = reads | op.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return frozenset()

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
            "UF": "1" if self.uf else None,
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
    def reg_read(self) -> frozenset[str]:
        if self.c_op == "time":
            reads: frozenset[str] = frozenset({USR_TIME_REG, TIMED_BASE_REG})
        else:
            reads = frozenset({STATUS_REG})
        if isinstance(self.time, Register):
            reads = reads | self.time.regs()
        if isinstance(self.addr, Register):
            reads = reads | self.addr.regs()
        return reads

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
