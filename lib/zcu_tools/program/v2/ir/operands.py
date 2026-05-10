from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

# Pattern to extract registers, literals and operators
_OP_TOKEN_RE = re.compile(
    r"([a-zA-Z_]\w*|#u?\-?[0-9A-Fa-f]+|&[a-zA-Z0-9_]+|@[0-9\-]+|<<|>>|AND|OR|XOR|ASR|SR|SL|ABS|MSH|LSH|SWP|CAT|::|NOT|!|PAR|[\+\-\*&\|\^=<>]+)"
)


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


@dataclass(frozen=True)
class Register(Operand):
    name: str

    def get_read_regs(self) -> set[str]:
        # 'r_wave' is special, it aliases with w0-w5
        if self.name.startswith("#"):
            return set()
        canon = _REG_ALIAS.get(self.name, self.name)
        if canon == "r_wave":
            return {"r_wave", "w0", "w1", "w2", "w3", "w4", "w5"}
        if canon in {"w0", "w1", "w2", "w3", "w4", "w5"}:
            return {canon, "r_wave"}
        # Always include both alias and canonical so callers using either name
        # see the same set (mirrors how the assembler resolves aliases).
        if canon != self.name:
            return {self.name, canon}
        return {self.name}

    def get_write_regs(self) -> set[str]:
        canon = _REG_ALIAS.get(self.name, self.name)
        if canon == "r_wave":
            return {"r_wave", "w0", "w1", "w2", "w3", "w4", "w5"}
        if canon in {"w0", "w1", "w2", "w3", "w4", "w5"}:
            return {canon, "r_wave"}
        if canon != self.name:
            return {self.name, canon}
        return {self.name}

    def __str__(self) -> str:
        return self.name


def canonical_reg(name: str) -> str:
    """Resolve a register alias (e.g. 'w_freq' -> 'w0') to its canonical name.

    Names not present in the alias table are returned unchanged.  Use this
    whenever a pass compares a stored register name against a set of
    read/written regs, since some passes record the alias while others
    record the canonical form.
    """
    return _REG_ALIAS.get(name, name)


# QICK register aliases (e.g. w_freq -> w0, s_zero -> s0).  Mirrors
# QickProgramV2.REG_ALIASES; declared here so IR analysis stays
# self-contained.
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


@dataclass(frozen=True)
class Literal(Operand):
    value: str

    def get_read_regs(self) -> set[str]:
        return set()

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class AluExpr(Operand):
    """Represents an ALU operation (e.g., 'r1 + #1', 'ABS r2')."""

    lhs: Register
    op: str
    rhs: Optional[Union[Register, Literal]] = None

    def get_read_regs(self) -> set[str]:
        regs = self.lhs.get_read_regs()
        if self.rhs:
            regs.update(self.rhs.get_read_regs())
        return regs

    def __str__(self) -> str:
        if self.rhs is None:
            # Unary operators usually come first (e.g., 'ABS r1')
            if not self.op:
                return str(self.lhs)
            return f"{self.op} {self.lhs}"
        return f"{self.lhs} {self.op} {self.rhs}"


@dataclass(frozen=True)
class SideWrite(Operand):
    """Represents the -wr() modifier (e.g., 'r1 op', 's2 imm')."""

    dst: Register
    src_type: str

    def get_read_regs(self) -> set[str]:
        return set()

    def get_write_regs(self) -> set[str]:
        return self.dst.get_write_regs()

    def __str__(self) -> str:
        return f"{self.dst} {self.src_type}"


# --- Parsers ---


def parse_register_or_literal(val: str) -> Union[Register, Literal]:
    if (
        val.startswith("#")
        or val.startswith("&")
        or val.startswith("@")
        or val.startswith("0x")
    ):
        return Literal(val)
    # Some numbers without '#' might sneak in depending on context,
    # but QICK literals usually have prefixes. If it's a raw number, treat as literal.
    if val.lstrip("-").isdigit():
        return Literal(val)
    return Register(val)


def parse_alu_expr(op_str: str) -> AluExpr:
    tokens = _OP_TOKEN_RE.findall(op_str)
    if not tokens:
        raise ValueError(f"Invalid ALU expression: {op_str!r}")

    # Binary op: <lhs> <op> <rhs>
    if len(tokens) >= 3:
        if len(tokens) > 3:
            raise ValueError(f"ALU expression has too many tokens: {op_str!r}")
        lhs = Register(tokens[0])
        op = tokens[1]
        rhs = parse_register_or_literal(tokens[2])
        return AluExpr(lhs, op, rhs)

    # Unary op: <op> <lhs> (e.g., "ABS r1")
    if len(tokens) == 2:
        return AluExpr(Register(tokens[1]), op=tokens[0])

    # Single token (e.g. just a register pass-through like "r3")
    if len(tokens) == 1:
        return AluExpr(Register(tokens[0]), op="")

    raise ValueError(f"Cannot parse ALU expression: {op_str!r}")


def parse_side_write(wr_str: str) -> SideWrite:
    parts = wr_str.split()
    if len(parts) >= 2:
        return SideWrite(Register(parts[0]), parts[1])
    # Fallback if only reg is provided, default to "op"
    return SideWrite(Register(parts[0]), "op")
