from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Iterator, Optional, Union


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


@dataclass
class IRLoop(IRNode):
    """A loop node."""

    name: str = ""
    counter_reg: str = ""
    n: Union[int, str] = 0

    # Structural Labels (Attributes)
    start_label: Optional[str] = None
    end_label: Optional[str] = None

    body: BlockNode = field(default_factory=BlockNode)

    def children(self) -> Iterator[IRNode]:
        yield self.body

    def emit(self, prog_list: list[dict[str, Any]]) -> None:
        from .instructions import JumpInst, LabelInst, RegWriteInst, TestInst

        start = self.start_label or f"{self.name}_start"
        end = self.end_label or f"{self.name}_end"

        # Initialize counter
        RegWriteInst(dst=self.counter_reg, src="imm", extra_args={"LIT": "#0"}).emit(
            prog_list
        )

        # Loop start label
        LabelInst(name=start).emit(prog_list)

        # Stop check
        op_str = (
            f"{self.counter_reg} - #{self.n}"
            if isinstance(self.n, int)
            else f"{self.counter_reg} - {self.n}"
        )
        TestInst(op=op_str, uf="0").emit(prog_list)
        JumpInst(label=end, if_cond="NS").emit(prog_list)

        self.body.emit(prog_list)

        # Jump back
        JumpInst(label=start).emit(prog_list)

        # Loop end label
        LabelInst(name=end).emit(prog_list)


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
