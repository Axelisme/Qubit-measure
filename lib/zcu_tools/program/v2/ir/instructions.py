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

TimeCOp = Literal["inc_ref", "set_ref", "updt", "rst"]
_VALID_TIME_COPS = frozenset({"inc_ref", "set_ref", "updt", "rst"})

FlagCOp = Literal["set", "clr", "inv"]
_VALID_FLAG_COPS = frozenset({"set", "clr", "inv"})

WaitCOp = Literal["time", "port_dt", "div_rdy", "div_dt", "qpa_rdy", "qpa_dt"]
_VALID_WAIT_COPS = frozenset(
    {"time", "port_dt", "div_rdy", "div_dt", "qpa_rdy", "qpa_dt"}
)

ClearCOp = Literal["arith", "div", "qnet", "qcom", "qpa", "qpb", "port", "all"]
_VALID_CLEAR_COPS = frozenset(
    {"arith", "div", "qnet", "qcom", "qpa", "qpb", "port", "all"}
)

NetCOp = Literal[
    "set_net", "sync_net", "updt_offset", "set_dt", "get_dt", "set_flag", "get_flag"
]
_VALID_NET_COPS = frozenset(
    {"set_net", "sync_net", "updt_offset", "set_dt", "get_dt", "set_flag", "get_flag"}
)

ComCOp = Literal[
    "set_flag",
    "sync",
    "reset",
    "set_byte_1",
    "set_byte_2",
    "set_hw_1",
    "set_hw_2",
    "set_word_1",
    "set_word_2",
]
_VALID_COM_COPS = frozenset(
    {
        "set_flag",
        "sync",
        "reset",
        "set_byte_1",
        "set_byte_2",
        "set_hw_1",
        "set_hw_2",
        "set_word_1",
        "set_word_2",
    }
)

ArithCOp = Literal["T", "TP", "TM", "PT", "PTP", "PTM", "MT", "MTP", "MTM"]
_VALID_ARITH_COPS = frozenset({"T", "TP", "TM", "PT", "PTP", "PTM", "MT", "MTP", "MTM"})

TrigSrc = Literal["set", "clr"]
_VALID_TRIG_SRCS = frozenset({"set", "clr"})

PACOp = Literal["PA", "PB"]
_VALID_PA_CMDS = frozenset({"PA", "PB"})

ComFlagVal = Literal["0", "1"]
_VALID_COM_FLAG_VALS = frozenset({"0", "1"})


def _require_literal(val: str, field: str, valid: frozenset[str]) -> str:
    if val not in valid:
        raise ValueError(
            f"{field}: {val!r} is not valid, expected one of {sorted(valid)}"
        )
    return val


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
            "DPORT_RD": DportReadInst,
            "TRIG": TrigInst,
            "CALL": CallInst,
            "RET": RetInst,
            "FLAG": FlagInst,
            "ARITH": ArithInst,
            "DIV": DivInst,
            "NET": NetInst,
            "COM": ComInst,
            "PA": CustomPeripheralInst,
            "PB": CustomPeripheralInst,
            "CLEAR": ClearInst,
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

    c_op: TimeCOp
    lit: Optional[Immediate] = None
    r1: Optional[Register] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimeInst:
        return cls(
            c_op=_require_literal(str(d["C_OP"]), "TIME.C_OP", _VALID_TIME_COPS),  # type: ignore[arg-type]
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


DmemSrc = Literal["imm", "op"]


def _parse_dmem_src_keyword(val: str) -> DmemSrc:
    if val not in ("imm", "op"):
        raise ValueError(f"DMEM_WR.SRC must be 'imm' or 'op', got {val!r}")
    return val  # type: ignore[return-value]


@dataclass(frozen=True)
class DmemWriteInst(BaseInst):
    """DMEM_WR instruction: write to data memory."""

    dst: Union[Register, MemAddr]
    src: DmemSrc = "imm"
    op: Optional[ExprType] = None
    lit: Optional[Immediate] = None
    wr: Optional[SideWrite] = None
    uf: bool = False
    if_cond: Optional[CondCode] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DmemWriteInst:
        dst_raw = str(d["DST"])
        if dst_raw.startswith("[") and dst_raw.endswith("]"):
            dst_raw = dst_raw[1:-1]
        return cls(
            dst=_parse_mem_addr_field(dst_raw, "DST"),
            src=_parse_dmem_src_keyword(str(d["SRC"])),
            wr=parse_side_write(d["WR"]) if "WR" in d else None,
            op=parse_alu_expr(d["OP"]) if "OP" in d else None,
            lit=parse_immediate(d["LIT"]) if "LIT" in d else None,
            uf="UF" in d,
            if_cond=_parse_cond_code(d.get("IF")),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        if isinstance(self.dst, Register):
            reads = reads | self.dst.regs()
        if self.op:
            reads = reads | self.op.regs()
        return reads

    def to_dict(self) -> dict[str, Any]:
        dst_str = str(self.dst)
        if isinstance(self.dst, MemAddr):
            dst_str = f"[{dst_str}]"
        d = {
            "CMD": "DMEM_WR",
            "DST": dst_str,
            "SRC": self.src,
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
class DportReadInst(BaseInst):
    """DPORT_RD instruction: read I/Q result from a data port."""

    dst: Union[Register, ImmValue]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DportReadInst:
        return cls(dst=_parse_port_dst(str(d["DST"])))

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset({STATUS_REG})
        if isinstance(self.dst, Register):
            reads = reads | self.dst.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return frozenset({"s8", "s9"})

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DPORT_RD",
            "DST": str(self.dst) if self.dst else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class TrigInst(BaseInst):
    """TRIG instruction: set or clear a trigger port."""

    dst: Union[Register, ImmValue]
    src: TrigSrc
    time: Optional[TimeType] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrigInst:
        return cls(
            dst=_parse_port_dst(str(d["DST"])),
            src=_require_literal(str(d["SRC"]), "TRIG.SRC", _VALID_TRIG_SRCS),  # type: ignore[arg-type]
            time=parse_time(d["TIME"]) if "TIME" in d else None,
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset({TIMED_BASE_REG})
        if isinstance(self.dst, Register):
            reads = reads | self.dst.regs()
        if isinstance(self.time, Register):
            reads = reads | self.time.regs()
        return reads

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "TRIG",
            "DST": str(self.dst) if self.dst else None,
            "SRC": self.src,
            "TIME": str(self.time) if self.time else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class CallInst(BaseInst):
    """CALL instruction: call a subroutine (stores return address before jumping)."""

    label: Optional[Label] = None
    addr: Optional[AddrType] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CallInst:
        return cls(
            label=parse_label(d.get("LABEL")),
            addr=parse_addr(d.get("ADDR")),
        )

    def __post_init__(self) -> None:
        if self.addr is not None and not isinstance(self.addr, Label):
            if self.addr != Register("s15"):
                raise ValueError(
                    f"CallInst.addr must be 's15' or Label, got {self.addr!r}."
                )

    @property
    def reg_read(self) -> frozenset[str]:
        if isinstance(self.addr, Register):
            return self.addr.regs()
        return frozenset()

    @property
    def need_label(self) -> Optional[Label]:
        if self.label and not self.label.is_pseudo_name():
            return self.label
        if isinstance(self.addr, Label) and not self.addr.is_pseudo_name():
            return self.addr
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "CALL",
            "LABEL": self.label.name if self.label else None,
            "ADDR": str(self.addr) if self.addr else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class RetInst(BaseInst):
    """RET instruction: return from a subroutine call."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetInst:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"CMD": "RET"}


@dataclass(frozen=True)
class FlagInst(BaseInst):
    """FLAG instruction: manipulate the external flag."""

    c_op: FlagCOp

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FlagInst:
        return cls(
            c_op=_require_literal(str(d["C_OP"]), "FLAG.C_OP", _VALID_FLAG_COPS),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "FLAG",
            "C_OP": self.c_op,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class ArithInst(BaseInst):
    """ARITH instruction: high-precision multiply-accumulate operation."""

    c_op: ArithCOp
    r1: Optional[Register] = None
    r2: Optional[Register] = None
    r3: Optional[Register] = None
    r4: Optional[Register] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArithInst:
        return cls(
            c_op=_require_literal(str(d["C_OP"]), "ARITH.C_OP", _VALID_ARITH_COPS),  # type: ignore[arg-type]
            r1=parse_register(d["R1"]) if "R1" in d else None,
            r2=parse_register(d["R2"]) if "R2" in d else None,
            r3=parse_register(d["R3"]) if "R3" in d else None,
            r4=parse_register(d["R4"]) if "R4" in d else None,
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        for r in [self.r1, self.r2, self.r3, self.r4]:
            if r is not None:
                reads = reads | r.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return frozenset({"s3"})

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "ARITH",
            "C_OP": self.c_op,
            "R1": str(self.r1) if self.r1 else None,
            "R2": str(self.r2) if self.r2 else None,
            "R3": str(self.r3) if self.r3 else None,
            "R4": str(self.r4) if self.r4 else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class DivInst(BaseInst):
    """DIV instruction: integer division (async, result in s4/s5)."""

    num: Register
    den: Union[Register, Immediate]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DivInst:
        num_raw = str(d["NUM"])
        den_raw = str(d["DEN"])
        num = parse_register(num_raw)
        if num is None:
            raise ValueError(f"DIV.NUM: {num_raw!r} is not a register")
        den_reg = parse_register(den_raw)
        if den_reg is not None:
            return cls(num=num, den=den_reg)
        den_imm = parse_immediate(den_raw)
        if den_imm is not None:
            return cls(num=num, den=den_imm)
        # Fallback: den may be a register name without # prefix
        return cls(num=num, den=Register(den_raw))

    @property
    def reg_read(self) -> frozenset[str]:
        reads = self.num.regs()
        if isinstance(self.den, Register):
            reads = reads | self.den.regs()
        return reads

    @property
    def reg_write(self) -> frozenset[str]:
        return frozenset({"s4", "s5"})

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "DIV",
            "NUM": str(self.num) if self.num else None,
            "DEN": str(self.den) if self.den else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class NetInst(BaseInst):
    """NET instruction: QNET network peripheral control."""

    c_op: NetCOp
    r1: Optional[Register] = None
    r2: Optional[Register] = None
    r3: Optional[Register] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NetInst:
        return cls(
            c_op=_require_literal(str(d["C_OP"]), "NET.C_OP", _VALID_NET_COPS),  # type: ignore[arg-type]
            r1=parse_register(d["R1"]) if "R1" in d else None,
            r2=parse_register(d["R2"]) if "R2" in d else None,
            r3=parse_register(d["R3"]) if "R3" in d else None,
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        for r in [self.r1, self.r2, self.r3]:
            if r is not None:
                reads = reads | r.regs()
        return reads

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "NET",
            "C_OP": self.c_op,
            "R1": str(self.r1) if self.r1 else None,
            "R2": str(self.r2) if self.r2 else None,
            "R3": str(self.r3) if self.r3 else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class ComInst(BaseInst):
    """COM instruction: QCOM communication peripheral control."""

    c_op: ComCOp
    r1: Optional[Register] = None
    flag_val: Optional[ComFlagVal] = None  # '0' or '1' for COM set_flag
    lit: Optional[Immediate] = None
    if_cond: Optional[CondCode] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ComInst:
        r1_raw = d.get("R1")
        flag_val: Optional[ComFlagVal] = None
        r1 = None
        if r1_raw is not None and isinstance(r1_raw, str):
            if r1_raw in _VALID_COM_FLAG_VALS:
                # COM set_flag encodes flag value as bare '0'/'1' in R1 field;
                # preserve it so to_dict can emit R1 exactly as assembler expects.
                flag_val = r1_raw  # type: ignore[assignment]
            else:
                r1 = parse_register(r1_raw)
        else:
            r1 = parse_register(r1_raw)

        return cls(
            c_op=_require_literal(str(d["C_OP"]), "COM.C_OP", _VALID_COM_COPS),  # type: ignore[arg-type]
            r1=r1,
            flag_val=flag_val,
            lit=parse_immediate(d["LIT"]) if "LIT" in d else None,
            if_cond=_parse_cond_code(d.get("IF")),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        if self.r1 is not None:
            return self.r1.regs()
        return frozenset()

    def to_dict(self) -> dict[str, Any]:
        r1_out = (
            self.flag_val
            if self.flag_val is not None
            else (str(self.r1) if self.r1 else None)
        )
        d = {
            "CMD": "COM",
            "C_OP": self.c_op,
            "R1": r1_out,
            "LIT": str(self.lit) if self.lit else None,
            "IF": self.if_cond if self.if_cond else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class CustomPeripheralInst(BaseInst):
    """PA / PB instruction: custom peripheral A or B."""

    cmd: PACOp
    c_op: int
    r1: Optional[Register] = None
    r2: Optional[Register] = None
    r3: Optional[Register] = None
    r4: Optional[Register] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CustomPeripheralInst:
        return cls(
            cmd=_require_literal(str(d["CMD"]), "PA/PB.CMD", _VALID_PA_CMDS),  # type: ignore[arg-type]
            c_op=int(d["C_OP"]),
            r1=parse_register(d["R1"]) if "R1" in d else None,
            r2=parse_register(d["R2"]) if "R2" in d else None,
            r3=parse_register(d["R3"]) if "R3" in d else None,
            r4=parse_register(d["R4"]) if "R4" in d else None,
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads: frozenset[str] = frozenset()
        for r in [self.r1, self.r2, self.r3, self.r4]:
            if r is not None:
                reads = reads | r.regs()
        return reads

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": self.cmd,
            "C_OP": str(self.c_op),  # assembler integer2bin() requires str
            "R1": str(self.r1) if self.r1 else None,
            "R2": str(self.r2) if self.r2 else None,
            "R3": str(self.r3) if self.r3 else None,
            "R4": str(self.r4) if self.r4 else None,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class ClearInst(BaseInst):
    """CLEAR instruction: clear peripheral data-new flags.

    Expands to REG_WR s2 imm at the binary level, but appears as CLEAR in
    prog_list before binary encoding.
    """

    c_op: ClearCOp

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClearInst:
        return cls(
            c_op=_require_literal(str(d["C_OP"]), "CLEAR.C_OP", _VALID_CLEAR_COPS),  # type: ignore[arg-type]
        )

    @property
    def reg_write(self) -> frozenset[str]:
        return frozenset({"s2"})

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "CLEAR",
            "C_OP": self.c_op,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class WaitInst(BaseInst):
    """WAIT instruction: wait for sync/trigger."""

    c_op: WaitCOp
    time: Optional[TimeType] = None
    addr: Optional[AddrType] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WaitInst:
        return cls(
            c_op=_require_literal(str(d["C_OP"]), "WAIT.C_OP", _VALID_WAIT_COPS),  # type: ignore[arg-type]
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
