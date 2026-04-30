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
class LoopNode(BlockNode):
    """A loop node containing an inner block."""
    name: str = ""
