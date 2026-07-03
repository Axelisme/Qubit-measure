from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from .hw_semantics import GENERAL_REGS, VOLATILE_REGS, WAVE_REGS

if TYPE_CHECKING:
    from .labels import Label, LabelRef

# Pattern to extract registers, literals and operators
_OP_TOKEN_RE = re.compile(
    r"([a-zA-Z_]\w*|#(?:u|b|h)?\-?[0-9A-Fa-f_xX]+|&[a-zA-Z0-9_]+|@[0-9\-_]+|<<|>>|AND|OR|XOR|ASR|SR|SL|ABS|MSH|LSH|SWP|CAT|::|NOT|!|PAR|[\+\-\*&\|\^=<>]+)"
)


class SrcKeyword(str, Enum):
    OP = "op"
    IMM = "imm"
    LABEL = "label"
    DMEM = "dmem"
    WMEM = "wmem"
    REG = "reg"


class AluOp(str, Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    AND_BIT = "&"
    OR_BIT = "|"
    XOR_BIT = "^"
    EQ = "="
    LT = "<"
    GT = ">"
    EQ_EQ = "=="
    NEQ = "!="
    LTE = "<="
    GTE = ">="
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    ASR = "ASR"
    SR = "SR"
    SL = "SL"
    ABS = "ABS"
    MSH = "MSH"
    LSH = "LSH"
    SWP = "SWP"
    CAT = "CAT"
    NOT = "NOT"
    PAR = "PAR"
    COLON2 = "::"
    BANG = "!"
    LSHIFT = "<<"
    RSHIFT = ">>"
    NONE = ""


class Operand(ABC):
    """Base class for all instruction operands."""

    @abstractmethod
    def regs(self) -> frozenset[str]:
        """Return canonical names of all registers referenced by this operand.

        This is register-neutral: whether the operand is a read source or write
        destination is determined by the instruction that holds it, not here.
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Format the operand for QICK assembly."""
        ...


# QICK register aliases
_REG_ALIAS: dict[str, str] = {
    "w_freq": "w0",
    "w_phase": "w1",
    "w_env": "w2",
    "w_gain": "w3",
    "w_length": "w4",
    "w_conf": "w5",
    "s_zero": "s0",
    "s_rand": "s1",
    "s_cfg": "s2",
    "s_ctrl": "s2",
    "s_arith_l": "s3",
    "s_div_q": "s4",
    "s_div_r": "s5",
    "s_core_r1": "s6",
    "s_core_r2": "s7",
    "s_port_l": "s8",
    "s_port_h": "s9",
    "s_status": "s10",
    "s_usr_time": "s11",
    "s_core_w1": "s12",
    "s_core_w2": "s13",
    "s_out_time": "s14",
    "s_addr": "s15",
    "zero": "s0",
    "curr_usr_time": "s11",
    "out_usr_time": "s14",
}


def canonical_reg(name: str) -> str:
    """Resolve a register alias (e.g. 'w_freq' -> 'w0') to its canonical name."""
    return _REG_ALIAS.get(name, name)


def _parse_prefixed_int(raw: str) -> int:
    if raw.startswith("#b"):
        return int(raw[2:], 2)
    if raw.startswith("#h"):
        return int(raw[2:], 16)
    if raw.startswith("#u"):
        return _parse_int_literal(raw[2:])
    if raw.startswith("#"):
        return _parse_int_literal(raw[1:])
    return _parse_int_literal(raw)


def _parse_int_literal(raw: str) -> int:
    try:
        return int(raw, 0)
    except ValueError:
        if re.fullmatch(r"[+-]?[0-9][0-9_]*", raw):
            return int(raw, 10)
        raise


@dataclass
class Register(Operand):
    name: str

    @property
    def canonical_name(self) -> str:
        """Single canonical name after alias resolution (e.g. 'w_freq' → 'w0')."""
        return _REG_ALIAS.get(self.name, self.name)

    def regs(self) -> frozenset[str]:
        """Expand 'r_wave' to all wave registers; otherwise return {canonical_name}."""
        c = self.canonical_name
        if c == "r_wave":
            return frozenset(WAVE_REGS)
        return frozenset({c})

    def is_general_reg(self) -> bool:
        """Return True if this is a general-purpose 'r' register."""
        return self.canonical_name in GENERAL_REGS

    def is_wave_reg(self) -> bool:
        """Return True if this is a wave register ('w0'-'w5') or 'r_wave'."""
        c = self.canonical_name
        return c == "r_wave" or c in WAVE_REGS

    def is_volatile_reg(self) -> bool:
        """True if writes have hardware side effects or ordering constraints (s0-s15)."""
        return self.canonical_name in VOLATILE_REGS

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ImmValue(Operand):
    """Bare integer value (no prefix), e.g. port number."""

    value: int

    def regs(self) -> frozenset[str]:
        return frozenset()

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Immediate(Operand):
    """Immediate integer value: #N (for ALU ops, REG_WR imm, etc.)"""

    value: int

    def regs(self) -> frozenset[str]:
        return frozenset()

    def __str__(self) -> str:
        return f"#{self.value}"


@dataclass(frozen=True)
class TimeOffset(Operand):
    """Time offset: @N (for WPORT_WR/DPORT_WR/WMEM_WR time field)"""

    value: int

    def regs(self) -> frozenset[str]:
        return frozenset()

    def __str__(self) -> str:
        return f"@{self.value}"


@dataclass(frozen=True)
class MemAddr(Operand):
    """Memory address: &N (for DMEM/WMEM addr field)"""

    value: int

    def regs(self) -> frozenset[str]:
        return frozenset()

    def __str__(self) -> str:
        return f"&{self.value}"


@dataclass(frozen=True)
class DmemAddr(Operand):
    """An unresolved reference to a dispatch jump table stored in data memory.

    A ``DmemAddr`` denotes "one dmem table" — a contiguous run of dmem words,
    each holding the resolved program address of one ``table_labels`` entry.
    It is a *reference*, not a resource: it carries no numeric offset.

    The pipeline's resolve step (running after every clone-capable pass) is
    what commits the resource: it scans for all ``DmemAddr`` references,
    allocates a dmem run for each unique one, and replaces the reference with
    a concrete ``Immediate(base_offset)``.  Cloning a ``DmemAddr`` (e.g. when
    UnrollLoopPass duplicates a loop body) is therefore always safe — the copy
    is just another reference, and its ``table_labels`` are remapped like any
    other label, so each copy resolves to its own distinct dmem run.

    ``__str__`` raises: a ``DmemAddr`` must never reach instruction
    serialization. If it does, the resolve step was skipped — a pipeline bug.
    """

    table_labels: tuple[Label, ...]

    def regs(self) -> frozenset[str]:
        return frozenset()

    def __str__(self) -> str:
        raise RuntimeError(
            "DmemAddr reached serialization unresolved; the pipeline resolve "
            "step must replace every DmemAddr with a concrete Immediate first."
        )


# Type aliases
ValueType: TypeAlias = Register | Immediate | ImmValue
AddrType: TypeAlias = Register | MemAddr
TimeType: TypeAlias = Register | TimeOffset
SrcType: TypeAlias = SrcKeyword | Register


@dataclass(frozen=True)
class AluExpr(Operand):
    """Represents an ALU operation (e.g., 'r1 + #1', 'ABS r2').

    ``rhs`` may transiently hold a ``DmemAddr`` placeholder while a dispatch
    table address is still unresolved; the pipeline resolve step rewrites it
    to an ``Immediate`` before serialization.
    """

    lhs: Register
    op: AluOp
    rhs: Register | Immediate | DmemAddr | None = None

    def regs(self) -> frozenset[str]:
        result = self.lhs.regs()
        if self.rhs is not None:
            result = result | self.rhs.regs()
        return result

    def __str__(self) -> str:
        if self.rhs is None:
            if self.op == AluOp.NONE:
                return str(self.lhs)
            return f"{self.op.value} {self.lhs}"
        return f"{self.lhs} {self.op.value} {self.rhs}"


ExprType: TypeAlias = AluExpr


@dataclass(frozen=True)
class SideWrite(Operand):
    """Represents the -wr() modifier (e.g., 'r1 op', 's2 imm')."""

    dst: Register
    src_type: str

    def regs(self) -> frozenset[str]:
        return self.dst.regs()

    def __str__(self) -> str:
        return f"{self.dst} {self.src_type}"


# ---------------------------------------------------------------------------
# External factory / parse functions
# ---------------------------------------------------------------------------


def parse_register(val: Register | str | None) -> Register | None:
    if isinstance(val, Register):
        return val
    if not isinstance(val, str):
        return None
    # asm_v2.py encodes dmem register addresses as '&rN'; strip the leading '&'
    core = val[1:] if val.startswith("&") else val
    # Accept only hardware-backed registers and known QICK aliases. This keeps
    # malformed names out of the IR instead of deferring the failure to assembler.
    if (
        core == "r_wave"
        or core in _REG_ALIAS
        or core in GENERAL_REGS
        or core in VOLATILE_REGS
        or core in WAVE_REGS
    ):
        return Register(name=core)
    return None


def parse_immediate(val: Immediate | str | int | None) -> Immediate | None:
    if isinstance(val, Immediate):
        return val
    if isinstance(val, int):
        return Immediate(value=val)
    if not isinstance(val, str):
        return None
    val = val.strip()
    if not val.startswith("#"):
        return None
    try:
        return Immediate(value=_parse_prefixed_int(val))
    except ValueError:
        return None


def parse_time_offset(val: TimeOffset | str | int | None) -> TimeOffset | None:
    if isinstance(val, TimeOffset):
        return val
    if isinstance(val, int):
        return TimeOffset(value=val)
    if not isinstance(val, str):
        return None
    val = val.strip()
    if not val.startswith("@"):
        return None
    try:
        return TimeOffset(value=int(val[1:], 0))
    except ValueError:
        return None


def parse_mem_addr(val: MemAddr | str | int | None) -> MemAddr | None:
    if isinstance(val, MemAddr):
        return val
    if isinstance(val, int):
        return MemAddr(value=val)
    if not isinstance(val, str):
        return None
    val = val.strip()
    if not val.startswith("&"):
        return None
    num_str = val[1:]
    # Must be numeric, not a register name
    try:
        return MemAddr(value=int(num_str, 0))
    except ValueError:
        return None


def parse_imm_value(val: ImmValue | str | int | None) -> ImmValue | None:
    """Parse a bare integer (no prefix)."""
    if isinstance(val, ImmValue):
        return val
    if isinstance(val, int):
        return ImmValue(value=val)
    if not isinstance(val, str):
        return None
    val = val.strip()
    try:
        return ImmValue(value=_parse_int_literal(val))
    except ValueError:
        pass
    return None


def parse_label(val: Label | LabelRef | str | None) -> LabelRef | None:
    from .labels import PSEUDO_LABELS, Label, LabelRef

    if isinstance(val, LabelRef):
        return val
    if isinstance(val, Label):
        return LabelRef(val)
    if not isinstance(val, str):
        return None
    name = val[1:] if val.startswith("&") else val
    if not name:
        return None
    if name in PSEUDO_LABELS:
        return LabelRef(name)  # type: ignore[arg-type]
    return LabelRef(Label(name))


def parse_alu_expr(val: AluExpr | str | None) -> AluExpr | None:
    if isinstance(val, AluExpr):
        return val
    if not isinstance(val, str):
        return None

    tokens = _OP_TOKEN_RE.findall(val)
    if not tokens:
        raise ValueError(f"Invalid ALU expression: {val!r}")

    if len(tokens) >= 3:
        if len(tokens) > 3:
            raise ValueError(f"ALU expression has too many tokens: {val!r}")
        lhs = parse_register(tokens[0])
        try:
            op = AluOp(tokens[1])
        except ValueError:
            raise ValueError(f"Unknown ALU operator: {tokens[1]!r}")
        rhs = _parse_alu_rhs(tokens[2])
        if lhs and rhs is not None:
            return AluExpr(lhs, op, rhs)
        raise ValueError(f"Cannot parse ALU operands: {tokens[0]} and {tokens[2]}")

    if len(tokens) == 2:
        lhs = parse_register(tokens[1])
        try:
            op = AluOp(tokens[0])
        except ValueError:
            raise ValueError(f"Unknown ALU unary operator: {tokens[0]!r}")
        if lhs:
            return AluExpr(lhs, op=op)
        raise ValueError(f"Cannot parse ALU operand: {tokens[1]}")

    if len(tokens) == 1:
        lhs = parse_register(tokens[0])
        if lhs:
            return AluExpr(lhs, op=AluOp.NONE)
        raise ValueError(f"Cannot parse ALU expression: {val!r}")

    raise ValueError(f"Cannot parse ALU expression: {val!r}")


def _parse_alu_rhs(token: str) -> Register | Immediate | None:
    imm = parse_immediate(token)
    if imm is not None:
        return imm
    reg = parse_register(token)
    return reg


def parse_side_write(val: SideWrite | str | None) -> SideWrite | None:
    if isinstance(val, SideWrite):
        return val
    if not isinstance(val, str):
        return None
    parts = val.split()
    if not parts:
        return None
    dst = parse_register(parts[0])
    if not dst:
        return None
    if len(parts) >= 2:
        return SideWrite(dst, parts[1])
    return SideWrite(dst, "op")


def parse_value(
    val: Register | Immediate | ImmValue | str | int | None,
) -> ValueType | None:
    """Parse into ValueType: Register | Immediate | ImmValue.

    Priority: Immediate (#N) > Register > ImmValue (bare int).
    """
    if val is None:
        return None
    if isinstance(val, (Register, Immediate, ImmValue)):
        return val
    if isinstance(val, int):
        return ImmValue(value=val)

    val_s = val.strip()
    imm = parse_immediate(val_s)
    if imm is not None:
        return imm
    reg = parse_register(val_s)
    if reg is not None:
        return reg
    bare = parse_imm_value(val_s)
    if bare is not None:
        return bare
    # Fallback: treat as register name
    return Register(name=val_s)


def parse_addr(val: Register | MemAddr | str | None) -> AddrType | None:
    """Parse into AddrType: Register | MemAddr."""
    if val is None:
        return None
    if isinstance(val, (Register, MemAddr)):
        return val

    if isinstance(val, str):
        val_s = val.strip()
        # &N (numeric) → MemAddr
        mem = parse_mem_addr(val_s)
        if mem is not None:
            return mem
        # &rN / &sN / &r_wave → Register (asm_v2 encodes dmem reg addresses as '&rN')
        reg = parse_register(val_s)
        if reg is not None:
            return reg
    return None


def parse_time(val: Register | TimeOffset | str | int | None) -> TimeType | None:
    """Parse into TimeType: Register | TimeOffset."""
    if val is None:
        return None
    if isinstance(val, (Register, TimeOffset)):
        return val
    if isinstance(val, int):
        return TimeOffset(value=val)

    val_s = val.strip()
    t = parse_time_offset(val_s)
    if t is not None:
        return t
    reg = parse_register(val_s)
    if reg is not None:
        return reg
    return None


def parse_src(val: SrcKeyword | Register | str | None) -> SrcType | None:
    if val is None:
        return None
    if isinstance(val, SrcKeyword):
        return val
    if isinstance(val, Register):
        return val
    if isinstance(val, str):
        try:
            return SrcKeyword(val)
        except ValueError:
            pass
        reg = parse_register(val)
        if reg is not None:
            return reg
    return None
