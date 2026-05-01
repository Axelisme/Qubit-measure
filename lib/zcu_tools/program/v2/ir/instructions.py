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
                line=d["LINE"],
                p_addr=d["P_ADDR"],
                args=d["ARGS"],
            )

        # Default to GenericInst for now
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
