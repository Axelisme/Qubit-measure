from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .instructions import Instruction


@dataclass
class IRNode:
    """Base class for all IR nodes."""

    pass


@dataclass
class BlockNode(IRNode):
    """A sequence of instructions and other nodes."""

    insts: list[Union[Instruction, "IRNode"]] = field(default_factory=list)

    def append(self, item: Union[Instruction, "IRNode"]) -> None:
        self.insts.append(item)


@dataclass
class RootNode(BlockNode):
    """The root of the IR tree."""

    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class IRLoop(IRNode):
    """A loop node separated into sections."""

    name: str = ""
    trip_count: int | None = None
    initial: BlockNode = field(default_factory=BlockNode)
    update: BlockNode = field(default_factory=BlockNode)
    stop_check: BlockNode = field(default_factory=BlockNode)
    body: BlockNode = field(default_factory=BlockNode)
    jump_back: BlockNode = field(default_factory=BlockNode)


@dataclass
class IRBranchCase(BlockNode):
    """A branch case with a stable logical identity."""

    name: str = ""


@dataclass
class IRBranch(BlockNode):
    """A branch node containing multiple cases."""

    name: str = ""
    cases: list[IRBranchCase] = field(default_factory=list)
