from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import Iterator, Optional, Union

if TYPE_CHECKING:
    from .instructions import Instruction


class IRNode:
    """Base class for all IR nodes."""

    def children(self) -> Iterator[IRNode]:
        """Yield all immediate child nodes."""
        return iter([])

    def emit(self, inst_list: list[Instruction]) -> None:
        """Flatten this node into a list of Instruction objects."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement emit()"
        )


@dataclass
class InstNode(IRNode):
    """A wrapper for a single linear instruction."""

    inst: Instruction

    def emit(self, inst_list: list[Instruction]) -> None:
        inst_list.append(self.inst)


@dataclass
class BlockNode(IRNode):
    """A sequence of IR nodes."""

    insts: list[IRNode] = field(default_factory=list)

    def append(self, item: IRNode) -> None:
        self.insts.append(item)

    def children(self) -> Iterator[IRNode]:
        yield from self.insts

    def emit(self, inst_list: list[Instruction]) -> None:
        for item in self.insts:
            item.emit(inst_list)


@dataclass
class RootNode(BlockNode):
    """The root of the IR tree."""


@dataclass
class IRLoop(IRNode):
    """A loop node."""

    name: str = ""
    counter_reg: str = ""
    n: Union[int, str] = 0
    range_hint: Optional[tuple[int, int]] = None

    # Structural Labels (Attributes)
    start_label: Optional[str] = None
    end_label: Optional[str] = None

    body: BlockNode = field(default_factory=BlockNode)

    def children(self) -> Iterator[IRNode]:
        yield self.body

    def emit(self, inst_list: list[Instruction]) -> None:
        from .instructions import JumpInst, LabelInst, RegWriteInst, TestInst

        start = self.start_label or f"{self.name}_start"
        end = self.end_label or f"{self.name}_end"

        # Initialize counter
        inst_list.append(
            RegWriteInst(dst=self.counter_reg, src="imm", extra_args={"LIT": "#0"})
        )

        # Loop start label
        inst_list.append(LabelInst(name=start))

        # Stop check
        op_str = (
            f"{self.counter_reg} - #{self.n}"
            if isinstance(self.n, int)
            else f"{self.counter_reg} - {self.n}"
        )
        inst_list.append(TestInst(op=op_str, uf="0"))
        inst_list.append(JumpInst(label=end, if_cond="NS"))

        self.body.emit(inst_list)

        # Jump back
        inst_list.append(JumpInst(label=start))

        # Loop end label
        inst_list.append(LabelInst(name=end))


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

    def emit(self, inst_list: list[Instruction]) -> None:
        self.dispatch.emit(inst_list)
        for case in self.cases:
            case.emit(inst_list)
