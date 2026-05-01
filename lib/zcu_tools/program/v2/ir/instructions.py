from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .node import IRNode


@dataclass(frozen=True)
class Instruction(IRNode):
    """Base class for all IR instructions."""

    # Optional metadata from QICK assembler
    line: Optional[int] = None
    p_addr: Optional[int] = None

    def emit(self, prog_list: list[dict[str, Any]]) -> None:
        prog_list.append(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Instruction":
        if "LABEL" in d and "CMD" not in d:
            args = {k: v for k, v in d.items() if k not in ("LABEL", "LINE", "P_ADDR")}
            return LabelInst(
                name=d["LABEL"],
                args=args,
                line=d.get("LINE"),
                p_addr=d.get("P_ADDR"),
            )

        cmd = d.get("CMD")
        if not cmd:
            raise ValueError(f"Unknown instruction format: {d}")

        if cmd == "__META__":
            return MetaInst(
                type=d["TYPE"],
                name=d["NAME"],
                line=d.get("LINE"),
                p_addr=d.get("P_ADDR"),
                args=d.get("ARGS", {}),
            )

        # Dispatch to structured types for known opcodes
        if cmd == "TIME":
            return TimeInst(
                c_op=d.get("C_OP", ""),
                lit=d.get("LIT"),
                r1=d.get("R1"),
                line=d.get("LINE"),
                p_addr=d.get("P_ADDR"),
            )
        elif cmd == "TEST":
            return TestInst(
                op=d.get("OP", ""),
                uf=d.get("UF"),
                line=d.get("LINE"),
                p_addr=d.get("P_ADDR"),
            )
        elif cmd == "JUMP":
            return JumpInst(
                label=d.get("LABEL", ""),
                if_cond=d.get("IF"),
                addr=d.get("ADDR"),
                line=d.get("LINE"),
                p_addr=d.get("P_ADDR"),
            )
        elif cmd == "REG_WR":
            extra_args = {
                k: v
                for k, v in d.items()
                if k not in ("CMD", "DST", "SRC", "LINE", "P_ADDR")
            }
            return RegWriteInst(
                dst=d.get("DST", ""),
                src=d.get("SRC", ""),
                extra_args=extra_args,
                line=d.get("LINE"),
                p_addr=d.get("P_ADDR"),
            )
        elif cmd == "WPORT_WR":
            extra_args = {
                k: v
                for k, v in d.items()
                if k not in ("CMD", "DST", "TIME", "LINE", "P_ADDR")
            }
            return PortWriteInst(
                dst=d.get("DST", ""),
                time=d.get("TIME", ""),
                extra_args=extra_args,
                line=d.get("LINE"),
                p_addr=d.get("P_ADDR"),
            )

        # Default to GenericInst for unmapped opcodes
        args = {k: v for k, v in d.items() if k not in ("CMD", "LINE", "P_ADDR")}
        return GenericInst(
            cmd=cmd,
            args=args,
            line=d.get("LINE"),
            p_addr=d.get("P_ADDR"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert back to QICK prog_list dict format."""
        raise NotImplementedError


@dataclass(frozen=True)
class GenericInst(Instruction):
    """Fallback for instructions without a specific model."""

    cmd: str = ""
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": self.cmd}
        d.update(self.args)
        if self.line is not None:
            d["LINE"] = self.line
        if self.p_addr is not None:
            d["P_ADDR"] = self.p_addr
        return d


@dataclass(frozen=True)
class LabelInst(Instruction):
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"LABEL": self.name}
        d.update(self.args)
        if self.line is not None:
            d["LINE"] = self.line
        if self.p_addr is not None:
            d["P_ADDR"] = self.p_addr
        return d


@dataclass(frozen=True)
class MetaInst(Instruction):
    """Meta instruction used for structural control like loops."""

    type: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "CMD": "__META__",
            "TYPE": self.type,
            "NAME": self.name,
            "LINE": self.line,
            "P_ADDR": self.p_addr,
            "ARGS": self.args,
        }

    def emit(self, prog_list: list[dict[str, Any]]) -> None:
        pass


@dataclass(frozen=True)
class TimeInst(Instruction):
    """TIME instruction: advance timing counter."""

    c_op: str = ""  # inc_ref, trigger, etc.
    lit: Optional[str] = None  # e.g., "#10" for literal value
    r1: Optional[str] = None  # register operand

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "TIME", "C_OP": self.c_op}
        if self.lit is not None:
            d["LIT"] = self.lit
        if self.r1 is not None:
            d["R1"] = self.r1
        if self.line is not None:
            d["LINE"] = self.line
        if self.p_addr is not None:
            d["P_ADDR"] = self.p_addr
        return d


@dataclass(frozen=True)
class TestInst(Instruction):
    """TEST instruction: evaluate condition for conditional branch."""

    op: str = ""  # The condition to test (e.g., "r1 == r2")
    uf: Optional[str] = None  # Overflow/underflow flag (usually "1")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "TEST", "OP": self.op}
        if self.uf is not None:
            d["UF"] = self.uf
        if self.line is not None:
            d["LINE"] = self.line
        if self.p_addr is not None:
            d["P_ADDR"] = self.p_addr
        return d


@dataclass(frozen=True)
class JumpInst(Instruction):
    """JUMP instruction: unconditional or conditional jump."""

    label: str = ""  # Target label
    if_cond: Optional[str] = None  # Condition for conditional jump (e.g., "eq", "nz")
    addr: Optional[str] = None  # Direct address for large jumps (e.g., "s15")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "JUMP"}
        if self.label:
            d["LABEL"] = self.label
        if self.addr is not None:
            d["ADDR"] = self.addr
        if self.if_cond is not None:
            d["IF"] = self.if_cond
        if self.line is not None:
            d["LINE"] = self.line
        if self.p_addr is not None:
            d["P_ADDR"] = self.p_addr
        return d


@dataclass(frozen=True)
class RegWriteInst(Instruction):
    """REG_WR instruction: write to register."""

    dst: str = ""  # Destination register
    src: str = ""  # Source: 'op' (ALU operation), 'imm' (immediate), 'reg' (register)
    extra_args: dict[str, Any] = field(default_factory=dict)  # OP, LIT, UF, UDF, etc.

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "REG_WR", "DST": self.dst, "SRC": self.src}
        d.update(self.extra_args)
        if self.line is not None:
            d["LINE"] = self.line
        if self.p_addr is not None:
            d["P_ADDR"] = self.p_addr
        return d


@dataclass(frozen=True)
class PortWriteInst(Instruction):
    """WPORT_WR instruction: write to output port."""

    dst: str = ""  # Destination (output port)
    time: str = ""  # Timing reference
    extra_args: dict[str, Any] = field(default_factory=dict)  # Other fields

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "WPORT_WR", "DST": self.dst, "TIME": self.time}
        d.update(self.extra_args)
        if self.line is not None:
            d["LINE"] = self.line
        if self.p_addr is not None:
            d["P_ADDR"] = self.p_addr
        return d
