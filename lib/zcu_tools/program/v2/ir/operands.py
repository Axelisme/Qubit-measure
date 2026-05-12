from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from typing_extensions import Optional, TypeAlias, Union

from .hw_semantics import VOLATILE_REGS, WAVE_REGS, GENERAL_REGS

if TYPE_CHECKING:
    from .labels import Label

# Pattern to extract registers, literals and operators
_OP_TOKEN_RE = re.compile(
    r"([a-zA-Z_]\w*|#u?\-?[0-9A-Fa-f]+|&[a-zA-Z0-9_]+|@[0-9\-]+|<<|>>|AND|OR|XOR|ASR|SR|SL|ABS|MSH|LSH|SWP|CAT|::|NOT|!|PAR|[\+\-\*&\|\^=<>]+)"
)


class SrcKeyword(str, Enum):
    OP = "op"
    IMM = "imm"
    LABEL = "label"
    DMEM = "dmem"
    WMEM = "wmem"


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
}


def canonical_reg(name: str) -> str:
    """Resolve a register alias (e.g. 'w_freq' -> 'w0') to its canonical name."""
    return _REG_ALIAS.get(name, name)


@dataclass
class Register(Operand):
    name: str

    def __post_init__(self):
        if self.name.startswith("&"):
            self.name = self.name[1:]

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
        """True if writes have hardware side effects (s0-s14)."""
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


# Type aliases
ValueType: TypeAlias = Union[Register, Immediate, ImmValue]
AddrType: TypeAlias = Union[Register, MemAddr, "Label"]
TimeType: TypeAlias = Union[Register, TimeOffset]
SrcType: TypeAlias = Union[SrcKeyword, Register]


@dataclass(frozen=True)
class AluExpr(Operand):
    """Represents an ALU operation (e.g., 'r1 + #1', 'ABS r2')."""

    lhs: Register
    op: AluOp
    rhs: Optional[Union[Register, Immediate]] = None

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


def parse_register(val: Union[Register, str, None]) -> Optional[Register]:
    if isinstance(val, Register):
        return val
    if not isinstance(val, str):
        return None
    core_name = val[1:] if val.startswith("&") else val
    if (
        core_name.startswith("r_wave")
        or core_name.startswith("s_")
        or core_name.startswith("w_")
        or (bool(core_name) and core_name[0] in "rswp" and core_name[1:].isdigit())
        or core_name in _REG_ALIAS
    ):
        return Register(name=val)
    return None


def parse_immediate(val: Union[Immediate, str, int, None]) -> Optional[Immediate]:
    if isinstance(val, Immediate):
        return val
    if isinstance(val, int):
        return Immediate(value=val)
    if not isinstance(val, str):
        return None
    val = val.strip()
    if val.startswith("#u"):
        num_str = val[2:]
    elif val.startswith("#"):
        num_str = val[1:]
    else:
        return None
    try:
        return Immediate(value=int(num_str, 0))
    except ValueError:
        return None


def parse_time_offset(val: Union[TimeOffset, str, int, None]) -> Optional[TimeOffset]:
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


def parse_mem_addr(val: Union[MemAddr, str, int, None]) -> Optional[MemAddr]:
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


def parse_imm_value(val: Union[ImmValue, str, int, None]) -> Optional[ImmValue]:
    """Parse a bare integer (no prefix)."""
    if isinstance(val, ImmValue):
        return val
    if isinstance(val, int):
        return ImmValue(value=val)
    if not isinstance(val, str):
        return None
    val = val.strip()
    try:
        if (
            val.isdigit()
            or (val.startswith("-") and val[1:].isdigit())
            or val.startswith("0x")
        ):
            return ImmValue(value=int(val, 0))
    except ValueError:
        pass
    return None


def parse_label(val: Union["Label", str, None]) -> Optional["Label"]:
    from .labels import Label

    if isinstance(val, Label):
        return val
    if not isinstance(val, str):
        return None
    name = val[1:] if val.startswith("&") else val
    if not name:
        return None
    try:
        return Label.use_existing(name)
    except ValueError:
        return Label.make_new(name)


def parse_alu_expr(val: Union[AluExpr, str, None]) -> Optional[AluExpr]:
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


def _parse_alu_rhs(token: str) -> Optional[Union[Register, Immediate]]:
    imm = parse_immediate(token)
    if imm is not None:
        return imm
    reg = parse_register(token)
    return reg


def parse_side_write(val: Union[SideWrite, str, None]) -> Optional[SideWrite]:
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
    val: Union[Register, Immediate, ImmValue, str, int, None],
) -> Optional[ValueType]:
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


def parse_addr(val: Union[Register, MemAddr, str, None]) -> Optional[AddrType]:
    """Parse into AddrType: Register | MemAddr | Label."""
    from .labels import Label

    if val is None:
        return None
    if isinstance(val, (Register, MemAddr)):
        return val
    if isinstance(val, Label):
        return val

    if isinstance(val, str):
        val_s = val.strip()
        # &N (numeric) → MemAddr
        mem = parse_mem_addr(val_s)
        if mem is not None:
            return mem
        # &name → Label
        if val_s.startswith("&"):
            lbl = parse_label(val_s)
            if lbl is not None:
                return lbl
        reg = parse_register(val_s)
        if reg is not None:
            return reg
    return None


def parse_time(val: Union[Register, TimeOffset, str, int, None]) -> Optional[TimeType]:
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
    # Fallback: treat any non-empty string as a register name
    if val_s:
        return Register(name=val_s)
    return None


def parse_src(val: Union[SrcKeyword, Register, str, None]) -> Optional[SrcType]:
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
