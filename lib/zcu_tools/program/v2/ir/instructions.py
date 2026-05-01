from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .utils import regs_from_value, strip_write_modifier


@dataclass(frozen=True)
class Instruction:
    """Base class for all IR instructions."""

    # Optional metadata from QICK assembler
    line: Optional[int] = None
    annotations: dict[str, Any] = field(default_factory=dict)

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
    def need_label(self) -> Optional[str]:
        """Label name this instruction depends on (e.g. for JUMP or WR_ADDR)."""
        return None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Instruction":
        # Extract internal annotations (starting with IR_)
        annotations = {k: v for k, v in d.items() if k.startswith("IR_")}
        clean_d = {k: v for k, v in d.items() if not k.startswith("IR_")}

        if "LABEL" in clean_d and "CMD" not in clean_d:
            args = {
                k: v for k, v in clean_d.items() if k not in ("LABEL", "LINE", "P_ADDR")
            }
            return LabelInst(
                name=clean_d["LABEL"],
                args=args,
                line=clean_d.get("LINE"),
                annotations=annotations,
            )

        cmd = clean_d.get("CMD")
        if not cmd:
            raise ValueError(f"Unknown instruction format: {d}")

        if cmd == "__META__":
            return MetaInst(
                type=clean_d["TYPE"],
                name=clean_d["NAME"],
                line=clean_d.get("LINE"),
                args=clean_d.get("ARGS", {}),
                annotations=annotations,
            )

        # Dispatch to structured types for known opcodes
        if cmd == "TIME":
            return TimeInst(
                c_op=clean_d.get("C_OP", ""),
                lit=clean_d.get("LIT"),
                r1=clean_d.get("R1"),
                line=clean_d.get("LINE"),
                annotations=annotations,
            )
        elif cmd == "TEST":
            return TestInst(
                op=clean_d.get("OP", ""),
                uf=clean_d.get("UF"),
                line=clean_d.get("LINE"),
                annotations=annotations,
            )
        elif cmd == "JUMP":
            return JumpInst(
                label=clean_d.get("LABEL", ""),
                if_cond=clean_d.get("IF"),
                addr=clean_d.get("ADDR"),
                line=clean_d.get("LINE"),
                annotations=annotations,
            )
        elif cmd == "REG_WR":
            extra_args = {
                k: v
                for k, v in clean_d.items()
                if k not in ("CMD", "DST", "SRC", "LINE", "P_ADDR")
            }
            return RegWriteInst(
                dst=clean_d.get("DST", ""),
                src=clean_d.get("SRC", ""),
                extra_args=extra_args,
                line=clean_d.get("LINE"),
                annotations=annotations,
            )
        elif cmd == "WPORT_WR":
            extra_args = {
                k: v
                for k, v in clean_d.items()
                if k not in ("CMD", "DST", "TIME", "LINE", "P_ADDR")
            }
            return PortWriteInst(
                dst=clean_d.get("DST", ""),
                time=clean_d.get("TIME", ""),
                extra_args=extra_args,
                line=clean_d.get("LINE"),
                annotations=annotations,
            )
        elif cmd == "NOP":
            extra_args = {
                k: v for k, v in clean_d.items() if k not in ("CMD", "LINE", "P_ADDR")
            }
            return NopInst(
                extra_args=extra_args,
                line=clean_d.get("LINE"),
                annotations=annotations,
            )
        elif cmd == "DMEM_RD":
            extra_args = {
                k: v
                for k, v in clean_d.items()
                if k not in ("CMD", "DST", "ADDR", "LINE", "P_ADDR")
            }
            return DmemReadInst(
                dst=clean_d.get("DST", ""),
                addr=clean_d.get("ADDR", ""),
                extra_args=extra_args,
                line=clean_d.get("LINE"),
                annotations=annotations,
            )
        elif cmd == "DMEM_WR":
            extra_args = {
                k: v
                for k, v in clean_d.items()
                if k not in ("CMD", "SRC", "ADDR", "LINE", "P_ADDR")
            }
            return DmemWriteInst(
                src=clean_d.get("SRC", ""),
                addr=clean_d.get("ADDR", ""),
                extra_args=extra_args,
                line=clean_d.get("LINE"),
                annotations=annotations,
            )
        elif cmd == "DPORT_WR":
            extra_args = {
                k: v
                for k, v in clean_d.items()
                if k not in ("CMD", "DST", "DATA", "LINE", "P_ADDR")
            }
            return DportWriteInst(
                dst=clean_d.get("DST", ""),
                data=clean_d.get("DATA", ""),
                extra_args=extra_args,
                line=clean_d.get("LINE"),
                annotations=annotations,
            )
        elif cmd == "WAIT":
            extra_args = {
                k: v for k, v in clean_d.items() if k not in ("CMD", "LINE", "P_ADDR")
            }
            return WaitInst(
                extra_args=extra_args,
                line=clean_d.get("LINE"),
                annotations=annotations,
            )

        # Default to GenericInst for unmapped opcodes
        args = {k: v for k, v in clean_d.items() if k not in ("CMD", "LINE", "P_ADDR")}
        return GenericInst(
            cmd=cmd,
            args=args,
            line=clean_d.get("LINE"),
            annotations=annotations,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert back to QICK prog_list dict format."""
        raise NotImplementedError


@dataclass(frozen=True)
class GenericInst(Instruction):
    """Fallback for instructions without a specific model."""

    cmd: str = ""
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        for key in ("R1", "R2", "R3", "ADDR"):
            value = self.args.get(key)
            if isinstance(value, str) and not value.startswith("#"):
                reads.add(value)

        for key in ("SRC", "OP", "TIME"):
            reads.update(regs_from_value(self.args.get(key)))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        writes: set[str] = set()
        for key in ("DST", "WR"):
            value = self.args.get(key)
            if isinstance(value, str):
                writes.add(strip_write_modifier(value))
        return sorted(list(writes))

    @property
    def need_label(self) -> Optional[str]:
        label = self.args.get("LABEL")
        if isinstance(label, str) and label not in ("HERE", "NEXT", "PREV", "SKIP"):
            return label
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": self.cmd}
        d.update(self.args)
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class LabelInst(Instruction):
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"LABEL": self.name}
        d.update(self.args)
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class MetaInst(Instruction):
    """Meta instruction used for structural control like loops."""

    type: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "CMD": "__META__",
            "TYPE": self.type,
            "NAME": self.name,
            "LINE": self.line,
            "ARGS": self.args,
        }
        d.update(self.annotations)
        return d


@dataclass(frozen=True)
class TimeInst(Instruction):
    """TIME instruction: advance timing counter."""

    c_op: str = ""  # inc_ref, trigger, etc.
    lit: Optional[str] = None  # e.g., "#10" for literal value
    r1: Optional[str] = None  # register operand

    @property
    def reg_read(self) -> list[str]:
        if self.r1 and not self.r1.startswith("#"):
            return [self.r1]
        return []

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "TIME", "C_OP": self.c_op}
        if self.lit is not None:
            d["LIT"] = self.lit
        if self.r1 is not None:
            d["R1"] = self.r1
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class TestInst(Instruction):
    """TEST instruction: evaluate condition for conditional branch."""

    __test__ = False
    op: str = ""  # The condition to test (e.g., "r1 == r2")
    uf: Optional[str] = None  # Overflow/underflow flag (usually "1")

    @property
    def reg_read(self) -> list[str]:
        return sorted(list(regs_from_value(self.op)))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "TEST", "OP": self.op}
        if self.uf is not None:
            d["UF"] = self.uf
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class JumpInst(Instruction):
    """JUMP instruction: unconditional or conditional jump."""

    label: str = ""  # Target label
    if_cond: Optional[str] = None  # Condition for conditional jump (e.g., "eq", "nz")
    addr: Optional[str] = None  # Direct address for large jumps (e.g., "s15")

    @property
    def need_label(self) -> Optional[str]:
        if self.label and self.label not in ("HERE", "NEXT", "PREV", "SKIP"):
            return self.label
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "JUMP"}
        if self.label:
            d["LABEL"] = self.label
        if self.addr is not None:
            d["ADDR"] = self.addr
        if self.if_cond is not None:
            d["IF"] = self.if_cond
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class RegWriteInst(Instruction):
    """REG_WR instruction: write to register."""

    dst: str = ""  # Destination register
    src: str = ""  # Source: 'op' (ALU operation), 'imm' (immediate), 'reg' (register), 'dmem' (memory)
    extra_args: dict[str, Any] = field(
        default_factory=dict
    )  # OP, LIT, UF, UDF, ADDR (for dmem), etc.

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        if "OP" in self.extra_args:
            reads.update(regs_from_value(self.extra_args["OP"]))
        if "ADDR" in self.extra_args:
            reads.update(regs_from_value(self.extra_args["ADDR"]))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return [strip_write_modifier(self.dst)]

    @property
    def need_label(self) -> Optional[str]:
        label = self.extra_args.get("LABEL")
        if isinstance(label, str) and label not in ("HERE", "NEXT", "PREV", "SKIP"):
            return label
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "REG_WR", "DST": self.dst, "SRC": self.src}
        d.update(self.extra_args)
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class PortWriteInst(Instruction):
    """WPORT_WR instruction: write to output port."""

    dst: str = ""  # Destination (output port)
    time: str = ""  # Timing reference
    extra_args: dict[str, Any] = field(default_factory=dict)  # Other fields

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.time))
        for val in self.extra_args.values():
            reads.update(regs_from_value(val))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return [strip_write_modifier(self.dst)]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "WPORT_WR", "DST": self.dst, "TIME": self.time}
        d.update(self.extra_args)
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class NopInst(Instruction):
    """NOP instruction: no operation."""

    extra_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "NOP"}
        d.update(self.extra_args)
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class DmemReadInst(Instruction):
    """DMEM_RD instruction: read from data memory.
    Note: QICK often maps this to REG_WR with SRC='dmem'.
    """

    dst: str = ""  # Destination register
    addr: str = ""  # Memory address (register or literal)
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.addr))
        for val in self.extra_args.values():
            reads.update(regs_from_value(val))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        return [strip_write_modifier(self.dst)]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "DMEM_RD", "DST": self.dst, "ADDR": self.addr}
        d.update(self.extra_args)
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class DmemWriteInst(Instruction):
    """DMEM_WR instruction: write to data memory."""

    src: str = ""  # Source (register or literal)
    addr: str = ""  # Memory address (register or literal)
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.src))
        reads.update(regs_from_value(self.addr))
        for val in self.extra_args.values():
            reads.update(regs_from_value(val))
        return sorted(list(reads))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "DMEM_WR", "SRC": self.src, "ADDR": self.addr}
        d.update(self.extra_args)
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class DportWriteInst(Instruction):
    """DPORT_WR instruction: write to data port."""

    dst: str = ""  # Destination (port)
    data: str = ""  # Data to write (register or literal)
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def reg_read(self) -> list[str]:
        reads: set[str] = set()
        reads.update(regs_from_value(self.dst))
        reads.update(regs_from_value(self.data))
        for val in self.extra_args.values():
            reads.update(regs_from_value(val))
        return sorted(list(reads))

    @property
    def reg_write(self) -> list[str]:
        # DPORT_WR usually writes to a port, but if DST is a register...
        # In QICK DPORT_WR, DST is usually a literal port number.
        return [strip_write_modifier(self.dst)] if not self.dst.startswith("#") else []

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "DPORT_WR", "DST": self.dst, "DATA": self.data}
        d.update(self.extra_args)
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d


@dataclass(frozen=True)
class WaitInst(Instruction):
    """WAIT instruction: wait for sync/trigger."""

    extra_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"CMD": "WAIT"}
        d.update(self.extra_args)
        d.update(self.annotations)
        if self.line is not None:
            d["LINE"] = self.line
        return d

    @property
    def addr_inc(self) -> int:
        return 2
