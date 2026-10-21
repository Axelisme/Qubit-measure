"""Core, compute, control and timing instructions with the central QICK dispatcher."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, override

from .hw_semantics import ADDR_REG, STATUS_REG, TIMED_BASE_REG, USR_TIME_REG
from .instruction_fields import (
    VALID_ARITH_COPS,
    VALID_CLEAR_COPS,
    VALID_COM_COPS,
    VALID_COM_FLAG_VALS,
    VALID_FLAG_COPS,
    VALID_NET_COPS,
    VALID_PA_CMDS,
    VALID_TIME_COPS,
    VALID_WAIT_COPS,
    ArithCOp,
    ClearCOp,
    ComCOp,
    ComFlagVal,
    CondCode,
    FlagCOp,
    NetCOp,
    PACOp,
    TimeCOp,
    WaitCOp,
    is_valid_literal,
    parse_cond_code,
    parse_optional_addr,
    parse_optional_immediate,
    parse_optional_register,
    parse_optional_side_write,
    parse_optional_src,
    parse_optional_time,
    parse_wp_port,
    remap_label_ref,
    require_alu_expr,
    require_literal,
    require_register,
)
from .labels import Label, LabelRef
from .operands import (
    AddrType,
    AluExpr,
    DmemAddr,
    ExprType,
    Immediate,
    Register,
    SideWrite,
    SrcKeyword,
    SrcType,
    TimeType,
    parse_immediate,
    parse_label,
    parse_register,
)


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
        # Import concrete IO instructions here to avoid a base/IO initialization cycle.
        from .io_instructions import (
            DmemReadInst,
            DmemWriteInst,
            DportReadInst,
            DportWriteInst,
            PortWriteInst,
            TrigInst,
            WmemWriteInst,
        )

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
    def need_label(self) -> Label | None:
        """Non-pseudo Label this instruction references (for dead-label analysis)."""
        return None

    @property
    def need_labels(self) -> frozenset[Label]:
        """All non-pseudo Labels this instruction references.

        Default: the single ``need_label`` if any. Instructions that reference
        several labels (e.g. a dmem dispatch table via ``DmemAddr``) override
        this so dead-label analysis sees every referenced label.
        """
        lbl = self.need_label
        return frozenset({lbl}) if lbl is not None else frozenset()

    # tolerated: No-label instructions share the remap signature with label-bearing overrides.
    def remap_labels(self, remap: dict[str, Label]) -> BaseInst:  # noqa: ARG002
        """Return a copy with LabelRef targets remapped via ``remap``.

        Default implementation returns ``self`` unchanged; override in any
        instruction that holds a ``LabelRef`` (JumpInst, RegWriteInst, …).
        ``remap`` is keyed by label name strings (``str(label)``).
        """
        return self


def _remap_op_labels(op: ExprType | None, remap: dict[str, Label]) -> ExprType | None:
    """Remap DmemAddr entry labels inside an AluExpr, or return op unchanged."""
    if isinstance(op, AluExpr) and isinstance(op.rhs, DmemAddr):
        new_table = tuple(remap.get(str(lbl), lbl) for lbl in op.rhs.table_labels)
        if new_table != op.rhs.table_labels:
            return dataclasses.replace(op, rhs=DmemAddr(table_labels=new_table))
    return op


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
    lit: Immediate | None = None
    r1: Register | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimeInst:
        return cls(
            c_op=require_literal(str(d["C_OP"]), "TIME.C_OP", VALID_TIME_COPS),
            lit=parse_optional_immediate(d, "LIT", "TIME.LIT"),
            r1=parse_optional_register(d, "R1", "TIME.R1"),
        )

    @property
    def reg_read(self) -> frozenset[str]:
        reads = self.r1.regs() if self.r1 else frozenset()
        if self.c_op == "inc_ref":
            reads = reads | {TIMED_BASE_REG}
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
            op=require_alu_expr(d["OP"], "OP"),
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

    label: LabelRef | None = None
    if_cond: CondCode | None = None
    addr: AddrType | None = None
    wr: SideWrite | None = None
    op: ExprType | None = None
    uf: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JumpInst:
        return cls(
            label=parse_label(d.get("LABEL")),
            if_cond=parse_cond_code(d.get("IF")),
            addr=parse_optional_addr(d, "ADDR", "JUMP.ADDR"),
            wr=parse_optional_side_write(d, "WR", "JUMP.WR"),
            op=require_alu_expr(d["OP"], "JUMP.OP") if "OP" in d else None,
            uf="UF" in d,
        )

    def __post_init__(self) -> None:
        if isinstance(self.addr, Register) and self.addr.canonical_name != ADDR_REG:
            raise ValueError(f"JumpInst.addr must be {ADDR_REG!r}, got {self.addr!r}.")
        if self.addr is not None and not isinstance(self.addr, Register):
            raise ValueError(f"JumpInst.addr must be {ADDR_REG!r}, got {self.addr!r}.")

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
    def need_label(self) -> Label | None:
        if self.label is not None and not self.label.is_pseudo():
            return self.label.as_label()
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "JUMP",
            "LABEL": str(self.label) if self.label is not None else None,
            "ADDR": str(self.addr) if self.addr else None,
            "IF": self.if_cond if self.if_cond else None,
            "WR": str(self.wr) if self.wr else None,
            "OP": str(self.op) if self.op else None,
            "UF": "1" if self.uf else None,
        }
        return {k: v for k, v in d.items() if v is not None}

    def remap_labels(self, remap: dict[str, Label]) -> JumpInst:
        new_label = remap_label_ref(self.label, remap)
        if new_label is self.label:
            return self
        return dataclasses.replace(self, label=new_label)


@dataclass(frozen=True)
class RegWriteInst(BaseInst):
    """REG_WR instruction: write to register."""

    dst: Register
    src: SrcType | None = None
    op: ExprType | None = None
    lit: Immediate | None = None
    addr: AddrType | None = None
    uf: bool = False
    wr: SideWrite | None = None
    if_cond: CondCode | None = None
    label: LabelRef | None = None
    ww: str | None = None
    wp: str | None = None
    port: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegWriteInst:
        wp, port = parse_wp_port(d, "REG_WR")
        return cls(
            dst=require_register(d["DST"], "DST"),
            src=parse_optional_src(d, "SRC", "REG_WR.SRC"),
            wr=parse_optional_side_write(d, "WR", "REG_WR.WR"),
            op=require_alu_expr(d["OP"], "REG_WR.OP") if "OP" in d else None,
            lit=parse_optional_immediate(d, "LIT", "REG_WR.LIT"),
            addr=parse_optional_addr(d, "ADDR", "REG_WR.ADDR"),
            uf="UF" in d,
            if_cond=parse_cond_code(d.get("IF")),
            label=parse_label(d.get("LABEL")),
            ww=d.get("WW"),
            wp=wp,
            port=port,
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
        writes = self.dst.regs() if self.dst else frozenset()
        if self.wr is not None:
            writes = writes | self.wr.regs()
        return writes

    @property
    def need_label(self) -> Label | None:
        if self.label is not None and not self.label.is_pseudo():
            return self.label.as_label()
        return None

    @property
    def need_labels(self) -> frozenset[Label]:
        labels = super().need_labels
        # A dmem dispatch table address (DmemAddr) references every case-entry
        # label; surface them so dead-label analysis keeps the entries alive.
        if isinstance(self.op, AluExpr) and isinstance(self.op.rhs, DmemAddr):
            labels = labels | frozenset(self.op.rhs.table_labels)
        return labels

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
            "LABEL": str(self.label) if self.label is not None else None,
            "WW": self.ww,
            "WP": self.wp,
            "PORT": self.port,
        }
        return {k: v for k, v in d.items() if v is not None}

    def remap_labels(self, remap: dict[str, Label]) -> RegWriteInst:
        new_label = remap_label_ref(self.label, remap)
        new_op = _remap_op_labels(self.op, remap)
        if new_label is self.label and new_op is self.op:
            return self
        return dataclasses.replace(self, label=new_label, op=new_op)


@dataclass(frozen=True)
class NopInst(BaseInst):
    """NOP instruction: no operation."""

    @classmethod
    @override
    def from_dict(cls, d: dict[str, Any]) -> NopInst:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"CMD": "NOP"}


@dataclass(frozen=True)
class CallInst(BaseInst):
    """CALL instruction: call a subroutine (stores return address before jumping)."""

    label: LabelRef | None = None
    addr: AddrType | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CallInst:
        return cls(
            label=parse_label(d.get("LABEL")),
            addr=parse_optional_addr(d, "ADDR", "CALL.ADDR"),
        )

    def __post_init__(self) -> None:
        if isinstance(self.addr, Register) and self.addr.canonical_name != ADDR_REG:
            raise ValueError(f"CallInst.addr must be {ADDR_REG!r}, got {self.addr!r}.")
        if self.addr is not None and not isinstance(self.addr, Register):
            raise ValueError(f"CallInst.addr must be {ADDR_REG!r}, got {self.addr!r}.")

    @property
    def reg_read(self) -> frozenset[str]:
        if isinstance(self.addr, Register):
            return self.addr.regs()
        return frozenset()

    @property
    def need_label(self) -> Label | None:
        if self.label is not None and not self.label.is_pseudo():
            return self.label.as_label()
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "CALL",
            "LABEL": str(self.label) if self.label is not None else None,
            "ADDR": str(self.addr) if self.addr else None,
        }
        return {k: v for k, v in d.items() if v is not None}

    def remap_labels(self, remap: dict[str, Label]) -> CallInst:
        new_label = remap_label_ref(self.label, remap)
        if new_label is self.label:
            return self
        return dataclasses.replace(self, label=new_label)


@dataclass(frozen=True)
class RetInst(BaseInst):
    """RET instruction: return from a subroutine call."""

    @classmethod
    @override
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
            c_op=require_literal(str(d["C_OP"]), "FLAG.C_OP", VALID_FLAG_COPS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "CMD": "FLAG",
            "C_OP": self.c_op,
        }


@dataclass(frozen=True)
class ArithInst(BaseInst):
    """ARITH instruction: high-precision multiply-accumulate operation."""

    c_op: ArithCOp
    r1: Register | None = None
    r2: Register | None = None
    r3: Register | None = None
    r4: Register | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArithInst:
        return cls(
            c_op=require_literal(str(d["C_OP"]), "ARITH.C_OP", VALID_ARITH_COPS),
            r1=parse_optional_register(d, "R1", "ARITH.R1"),
            r2=parse_optional_register(d, "R2", "ARITH.R2"),
            r3=parse_optional_register(d, "R3", "ARITH.R3"),
            r4=parse_optional_register(d, "R4", "ARITH.R4"),
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
    den: Register | Immediate

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
        raise ValueError(f"DIV.DEN: {den_raw!r} is not a register or immediate")

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
    r1: Register | None = None
    r2: Register | None = None
    r3: Register | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NetInst:
        return cls(
            c_op=require_literal(str(d["C_OP"]), "NET.C_OP", VALID_NET_COPS),
            r1=parse_optional_register(d, "R1", "NET.R1"),
            r2=parse_optional_register(d, "R2", "NET.R2"),
            r3=parse_optional_register(d, "R3", "NET.R3"),
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
    r1: Register | None = None
    flag_val: ComFlagVal | None = None  # '0' or '1' for COM set_flag
    lit: Immediate | None = None
    if_cond: CondCode | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ComInst:
        flag_val: ComFlagVal | None = None
        r1 = None
        if "R1" in d:
            r1_raw = d["R1"]
            if isinstance(r1_raw, str) and is_valid_literal(
                r1_raw, VALID_COM_FLAG_VALS
            ):
                flag_val = r1_raw
            else:
                r1 = parse_register(r1_raw)
                if r1 is None:
                    raise ValueError(
                        f"COM.R1: {r1_raw!r} is not a valid register name or flag value"
                    )

        return cls(
            c_op=require_literal(str(d["C_OP"]), "COM.C_OP", VALID_COM_COPS),
            r1=r1,
            flag_val=flag_val,
            lit=parse_optional_immediate(d, "LIT", "COM.LIT"),
            if_cond=parse_cond_code(d.get("IF")),
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
    r1: Register | None = None
    r2: Register | None = None
    r3: Register | None = None
    r4: Register | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CustomPeripheralInst:
        return cls(
            cmd=require_literal(str(d["CMD"]), "PA/PB.CMD", VALID_PA_CMDS),
            c_op=int(d["C_OP"]),
            r1=parse_optional_register(d, "R1", f"{d['CMD']}.R1"),
            r2=parse_optional_register(d, "R2", f"{d['CMD']}.R2"),
            r3=parse_optional_register(d, "R3", f"{d['CMD']}.R3"),
            r4=parse_optional_register(d, "R4", f"{d['CMD']}.R4"),
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
            c_op=require_literal(str(d["C_OP"]), "CLEAR.C_OP", VALID_CLEAR_COPS),
        )

    @property
    def reg_write(self) -> frozenset[str]:
        return frozenset({"s2"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "CMD": "CLEAR",
            "C_OP": self.c_op,
        }


@dataclass(frozen=True)
class WaitInst(BaseInst):
    """WAIT instruction: wait for sync/trigger."""

    c_op: WaitCOp
    time: TimeType | None = None
    addr: AddrType | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WaitInst:
        return cls(
            c_op=require_literal(str(d["C_OP"]), "WAIT.C_OP", VALID_WAIT_COPS),
            time=parse_optional_time(d, "TIME", "WAIT.TIME"),
            addr=parse_optional_addr(d, "ADDR", "WAIT.ADDR"),
        )

    def __post_init__(self) -> None:
        if isinstance(self.addr, Register) and self.addr.canonical_name != ADDR_REG:
            raise ValueError(f"WaitInst.addr must be {ADDR_REG!r}, got {self.addr!r}.")
        if self.addr is not None and not isinstance(self.addr, Register):
            raise ValueError(f"WaitInst.addr must be {ADDR_REG!r}, got {self.addr!r}.")

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
