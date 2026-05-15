from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

from .instructions import BaseInst, JumpInst, LabelInst, MetaInst
from .labels import Label
from .operands import Register


class IRNode(ABC):
    """Base class for all IR nodes.

    Every IRNode participates in the IR tree via the children() /
    replace_child() interface for uniform recursive traversal.
    """

    @abstractmethod
    def children(self) -> list[IRNode]:
        """Return the direct IRNode children of this node."""
        ...

    @abstractmethod
    def replace_child(self, old: IRNode, new: IRNode) -> None:
        """Replace a direct child in-place.

        Raises ValueError if `old` is not a direct child.
        Raises TypeError on leaf nodes that have no children.
        """
        ...

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

    def children(self) -> list[IRNode]:
        return []

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        raise TypeError("BasicBlockNode is a leaf node and has no children")

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
    """A sequence of IR nodes (structural container)."""

    insts: list[IRNode] = field(default_factory=list)

    def append(self, item: IRNode) -> None:
        self.insts.append(item)

    def children(self) -> list[IRNode]:
        return self.insts

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        idx = self.insts.index(old)
        self.insts[idx] = new

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return f"{prefix} {self.__class__.__name__}()\n" + "\n".join(
            i._into_str(indent + 1) for i in self.insts
        )


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
    body: IRNode
    range_hint: Optional[tuple[int, int]] = None

    def children(self) -> list[IRNode]:
        return [self.body]

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        if self.body is old:
            self.body = new
        else:
            raise ValueError(
                f"IRLoop.replace_child: {old!r} is not a child of this node"
            )

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return (
            f"{prefix} IRLoop(name={self.name}, n={self.n}, counter={str(self.counter_reg)}, range_hint={self.range_hint})\n"
            + self.body._into_str(indent + 1)
        )


@dataclass
class IRBranch(IRNode):
    """A branch node (pure data — lowering is handled by IRParser.unparse)."""

    name: str
    compare_reg: Register
    cases: list[IRNode]

    def children(self) -> list[IRNode]:
        return list(self.cases)

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        idx = self.cases.index(old)
        self.cases[idx] = new

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return (
            f"{prefix} IRBranch(name={self.name}, compare_reg={str(self.compare_reg)})\n"
            + "\n".join(i._into_str(indent + 1) for i in self.cases)
        )


@dataclass
class IRDispatch(IRNode):
    """A dispatch table node (pure data — lowering is handled by IRParser.unparse).

    Represents a value-indexed dispatch: ``value_reg`` selects which target
    label to jump to.  A mandatory out-of-range guard is always emitted: if
    ``value_reg >= len(target_labels)``, control falls through to
    ``target_labels[-1]`` (the last case).  This behaviour is intentional and
    must be documented at call sites that rely on it.

    IRDispatch is a leaf node — case bodies are **not** stored inside it.
    The caller (IRParser._lower_branch) is responsible for emitting the bodies
    after the dispatch table in the chunk stream.
    """

    name: str
    value_reg: Register
    target_labels: list[Label]

    def children(self) -> list[IRNode]:
        return []

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        raise TypeError("IRDispatch is a leaf node and has no children")

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        targets = ", ".join(str(lbl) for lbl in self.target_labels)
        return f"{prefix} IRDispatch(name={self.name}, value_reg={self.value_reg}, targets=[{targets}])\n"
