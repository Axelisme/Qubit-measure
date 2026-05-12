from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from typing_extensions import Any, Optional, Union, TypeAlias

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
    def get_read_regs(self) -> set[str]:
        """Return the names of all registers read by this operand."""
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
    name = name[1:] if name.startswith("&") else name
    return _REG_ALIAS.get(name, name)


@dataclass(frozen=True)
class Register(Operand):
    name: str

    @classmethod
    def parse(cls, val: Any) -> Optional[Register]:
        if isinstance(val, Register):
            return val
        if not isinstance(val, str):
            return None
        core_name = val[1:] if val.startswith("&") else val

        # Simple test to verify it looks like a register
        if (
            core_name.startswith("r_wave")
            or core_name.startswith("s_")
            or core_name.startswith("w_")
            or (core_name and core_name[0] in "rswp" and core_name[1:].isdigit())
            or core_name in _REG_ALIAS
        ):
            return cls(name=val)
        return None

    def is_general_reg(self) -> bool:
        """Return True if this is a general-purpose 'r' register."""
        canon = canonical_reg(self.name)
        return canon.startswith("r") and canon[1:].isdigit()

    def is_wave_reg(self) -> bool:
        """Return True if this is a wave register ('w0'-'w5') or 'r_wave'."""
        canon = canonical_reg(self.name)
        return canon == "r_wave" or (canon.startswith("w") and canon[1:].isdigit())

    def get_read_regs(self) -> set[str]:
        name = self.name[1:] if self.name.startswith("&") else self.name
        if name.startswith("#"):
            return set()
        canon = _REG_ALIAS.get(name, name)
        if canon == "r_wave":
            return {"r_wave", "w0", "w1", "w2", "w3", "w4", "w5"}
        if canon in {"w0", "w1", "w2", "w3", "w4", "w5"}:
            return {canon, "r_wave"}
        if canon != name:
            return {name, canon}
        return {name}

    def get_write_regs(self) -> set[str]:
        name = self.name[1:] if self.name.startswith("&") else self.name
        canon = _REG_ALIAS.get(name, name)
        if canon == "r_wave":
            return {"r_wave", "w0", "w1", "w2", "w3", "w4", "w5"}
        if canon in {"w0", "w1", "w2", "w3", "w4", "w5"}:
            return {canon, "r_wave"}
        if canon != name:
            return {name, canon}
        return {name}

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ImmValue(Operand):
    value: int
    prefix: str = ""

    @classmethod
    def parse(cls, val: Any) -> Optional[ImmValue]:
        if isinstance(val, ImmValue):
            return val
        if isinstance(val, int):
            return cls(value=val)
        if not isinstance(val, str):
            return None

        val = val.strip()
        prefix = ""
        num_str = val

        if val.startswith("#u"):
            prefix = "#u"
            num_str = val[2:]
        elif val.startswith(("#", "@", "&")):
            prefix = val[0]
            num_str = val[1:]
        elif not (
            num_str.isdigit()
            or (num_str.startswith("-") and num_str[1:].isdigit())
            or num_str.startswith("0x")
        ):
            return None  # Not an immediate value

        try:
            parsed_val = int(num_str, 0)
            return cls(value=parsed_val, prefix=prefix)
        except ValueError:
            return None

    def get_read_regs(self) -> set[str]:
        return set()

    def __str__(self) -> str:
        return f"{self.prefix}{self.value}"


# Forward declare type aliases
ValueType: TypeAlias = Union[Register, ImmValue]
AddrType: TypeAlias = Union[Register, ImmValue, "Label"]
SrcType: TypeAlias = Union[SrcKeyword, Register]


@dataclass(frozen=True)
class AluExpr(Operand):
    """Represents an ALU operation (e.g., 'r1 + #1', 'ABS r2')."""

    lhs: Register
    op: AluOp
    rhs: Optional[ValueType] = None

    @classmethod
    def parse(cls, op_str: Any) -> Optional[AluExpr]:
        if isinstance(op_str, AluExpr):
            return op_str
        if not isinstance(op_str, str):
            return None

        tokens = _OP_TOKEN_RE.findall(op_str)
        if not tokens:
            raise ValueError(f"Invalid ALU expression: {op_str!r}")

        if len(tokens) >= 3:
            if len(tokens) > 3:
                raise ValueError(f"ALU expression has too many tokens: {op_str!r}")
            lhs = Register.parse(tokens[0])
            try:
                op = AluOp(tokens[1])
            except ValueError:
                raise ValueError(f"Unknown ALU operator: {tokens[1]!r}")
            rhs = parse_value(tokens[2])
            if lhs and rhs is not None:
                return cls(lhs, op, rhs)
            raise ValueError(f"Cannot parse ALU operands: {tokens[0]} and {tokens[2]}")

        if len(tokens) == 2:
            lhs = Register.parse(tokens[1])
            try:
                op = AluOp(tokens[0])
            except ValueError:
                raise ValueError(f"Unknown ALU unary operator: {tokens[0]!r}")
            if lhs:
                return cls(lhs, op=op)
            raise ValueError(f"Cannot parse ALU operand: {tokens[1]}")

        if len(tokens) == 1:
            lhs = Register.parse(tokens[0])
            if lhs:
                return cls(lhs, op=AluOp.NONE)
            raise ValueError(f"Cannot parse ALU expression: {op_str!r}")

        raise ValueError(f"Cannot parse ALU expression: {op_str!r}")

    def get_read_regs(self) -> set[str]:
        regs = self.lhs.get_read_regs()
        if self.rhs:
            regs.update(self.rhs.get_read_regs())
        return regs

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

    @classmethod
    def parse(cls, wr_str: Any) -> Optional[SideWrite]:
        if isinstance(wr_str, SideWrite):
            return wr_str
        if not isinstance(wr_str, str):
            return None
        parts = wr_str.split()
        if not parts:
            return None
        dst = Register.parse(parts[0])
        if not dst:
            return None
        if len(parts) >= 2:
            return cls(dst, parts[1])
        return cls(dst, "op")

    def get_read_regs(self) -> set[str]:
        return set()

    def get_write_regs(self) -> set[str]:
        return self.dst.get_write_regs()

    def __str__(self) -> str:
        return f"{self.dst} {self.src_type}"


# --- Parsers ---


def parse_value(val: Any) -> Optional[ValueType]:
    if val is None:
        return None
    if isinstance(val, (Register, ImmValue)):
        return val
    if (parsed := ImmValue.parse(val)) is not None:
        return parsed
    if (parsed := Register.parse(val)) is not None:
        return parsed
    if isinstance(val, str):
        # Fallback for plain string register names that didn't strictly match parse rules
        return Register(name=val)
    return None


def parse_addr(val: Any) -> Optional[AddrType]:
    if val is None:
        return None
    if isinstance(val, (Register, ImmValue)):
        return val
    if val.__class__.__name__ == "Label":
        return val

    if isinstance(val, str) and val.startswith("&"):
        from .labels import Label

        if (parsed := Label.parse(val)) is not None:
            return parsed

    if (parsed := ImmValue.parse(val)) is not None:
        return parsed
    if (parsed := Register.parse(val)) is not None:
        return parsed
    return None


def parse_src(val: Any) -> Optional[SrcType]:
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
    if (parsed := Register.parse(val)) is not None:
        return parsed
    return None
