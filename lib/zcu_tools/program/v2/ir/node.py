from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Any, Iterator, Optional


class IRNode:
    """Base class for all IR nodes."""

    def children(self) -> Iterator[IRNode]:
        """Yield all immediate child nodes."""
        return iter([])

    def emit(self, prog_list: list[dict[str, Any]]) -> None:
        """Flatten this node into a list of QICK instructions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement emit()"
        )


@dataclass
class BlockNode(IRNode):
    """A sequence of IR nodes."""

    insts: list[IRNode] = field(default_factory=list)

    def append(self, item: IRNode) -> None:
        self.insts.append(item)

    def children(self) -> Iterator[IRNode]:
        yield from self.insts

    def emit(self, prog_list: list[dict[str, Any]]) -> None:
        for item in self.insts:
            item.emit(prog_list)


@dataclass
class RootNode(BlockNode):
    """The root of the IR tree."""

    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class IRLoop(IRNode):
    """A loop node separated into sections."""

    name: str = ""
    trip_count: Optional[int] = None

    # Structural Labels (Attributes)
    start_label: Optional[str] = None
    end_label: Optional[str] = None

    initial: BlockNode = field(default_factory=BlockNode)
    stop_check: BlockNode = field(default_factory=BlockNode)
    body: BlockNode = field(default_factory=BlockNode)
    update: BlockNode = field(default_factory=BlockNode)
    jump_back: BlockNode = field(default_factory=BlockNode)

    def children(self) -> Iterator[IRNode]:
        yield self.initial
        yield self.stop_check
        yield self.body
        yield self.update
        yield self.jump_back

    def emit(self, prog_list: list[dict[str, Any]]) -> None:
        from .instructions import LabelInst

        if self.start_label:
            LabelInst(name=self.start_label).emit(prog_list)

        self.initial.emit(prog_list)
        self.stop_check.emit(prog_list)
        self.body.emit(prog_list)
        self.update.emit(prog_list)
        self.jump_back.emit(prog_list)

        if self.end_label:
            LabelInst(name=self.end_label).emit(prog_list)


@dataclass
class IRBranchCase(BlockNode):
    """A branch case with a stable logical identity."""

    name: str = ""


@dataclass
class IRBranch(IRNode):
    """A branch node containing multiple cases."""

    name: str = ""
    # dispatch contains the binary tree of cond_jump and labels
    dispatch: BlockNode = field(default_factory=BlockNode)
    cases: list[IRBranchCase] = field(default_factory=list)

    def children(self) -> Iterator[IRNode]:
        yield self.dispatch
        yield from self.cases

    def emit(self, prog_list: list[dict[str, Any]]) -> None:
        self.dispatch.emit(prog_list)
        for case in self.cases:
            case.emit(prog_list)
