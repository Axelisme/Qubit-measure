from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .instructions import BaseInst, JumpInst, LabelInst, MetaInst
from .operands import Register


class IRNode:
    """Base class for all IR nodes."""

    def _into_str(self, indent: int = 0) -> str:
        """Helper for __str__ that takes an indent level."""
        return f"{'    ' * indent}{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self._into_str()


@dataclass
class BasicBlockNode(IRNode):
    """A basic block: a straight-line sequence with an optional terminal jump.

    labels: LabelInst(s) that mark the entry point of this block.
    insts:  Linear instructions (no labels, no jumps except TestInst).
    branch: Optional terminal JumpInst that ends this block.
    disable_opt: When True, the instruction count is frozen (set by jump-table
                  lowering). Post-LIR passes must NOP-pad instead of removing.
    """

    labels: list[LabelInst] = field(default_factory=list)
    insts: list[BaseInst] = field(default_factory=list)
    branch: Optional[JumpInst] = None
    disable_opt: bool = False

    def __post_init__(self) -> None:
        for inst in self.insts:
            if isinstance(inst, MetaInst):
                raise ValueError(
                    f"BasicBlockNode.insts must not contain MetaInst; "
                    f"use standalone MetaInst entries in the chunked stream instead. Got: {inst}"
                )
            if isinstance(inst, LabelInst):
                raise ValueError(
                    f"BasicBlockNode.insts must not contain LabelInst; "
                    f"use BasicBlockNode.labels instead. Got: {inst}"
                )
            if isinstance(inst, JumpInst):
                raise ValueError(
                    f"BasicBlockNode.insts must not contain JumpInst; "
                    f"use BasicBlockNode.branch instead. Got: {inst}"
                )

    @property
    def addr_size(self) -> int:
        size = sum(inst.addr_inc for inst in self.insts)
        if self.branch is not None:
            size += self.branch.addr_inc
        return size

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        lines = []
        if self.disable_opt:
            lines.append(f"{prefix}BasicBlockNode(disable_opt={self.addr_size}):")
        for lbl in self.labels:
            lines.append(f"{prefix}{lbl}:")
        for inst in self.insts:
            lines.append(f"{prefix}  {inst}")
        if self.branch is not None:
            lines.append(f"{prefix}  -> {self.branch}")
        return "\n".join(lines) + "\n"


@dataclass
class BlockNode(IRNode):
    """A sequence of IR nodes (structural container).

    Children must be BasicBlockNode | IRLoop | IRBranch | BlockNode.
    """

    insts: list[IRNode] = field(default_factory=list)

    def append(self, item: IRNode) -> None:
        self.insts.append(item)

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return f"{prefix} {self.__class__.__name__}()\n" + "\n".join(
            i._into_str(indent + 1) for i in self.insts
        )


@dataclass
class RootNode(BlockNode):
    """The root of the IR tree."""


@dataclass
class IRLoop(IRNode):
    """A loop node (pure data — lowering is handled by IRParser.unparse).

    `body` is treated as one full logical iteration, including the loop-carried
    counter update. Later linear passes may merge or reorder that update, so
    callers must not assume it remains the final instruction physically.
    """

    name: str
    counter_reg: Register
    n: Union[int, Register]
    body: BlockNode
    range_hint: Optional[tuple[int, int]] = None

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return (
            f"{prefix} IRLoop(name={self.name}, n={self.n}, counter={str(self.counter_reg)}, range_hint={self.range_hint})\n"
            + "\n".join(i._into_str(indent + 1) for i in self.body.insts)
        )


@dataclass
class IRBranch(IRNode):
    """A branch node (pure data — lowering is handled by IRParser.unparse)."""

    name: str
    compare_reg: Register
    cases: list[BlockNode]

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return (
            f"{prefix} IRBranch(name={self.name}, compare_reg={str(self.compare_reg)})\n"
            + "\n".join(i._into_str(indent + 1) for i in self.cases)
        )
