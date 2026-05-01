from __future__ import annotations

from typing_extensions import Any

from .factory import InstructionStream, parse_root
from .instructions import Instruction
from .linker import IRLinker
from .node import RootNode


class IRBuilder:
    def __init__(self):
        self.linker = IRLinker()

    def build(
        self, prog_list: list[dict[str, Any]], labels: dict[str, Any]
    ) -> RootNode:
        source_prog_list = self.linker.unlink(prog_list, labels)

        inst_list = [Instruction.from_dict(d) for d in source_prog_list]
        stream = InstructionStream(inst_list)
        root = parse_root(stream)

        if stream.peek() is not None:
            raise ValueError("Unparsed instructions remaining in stream")

        return root

    def unbuild(self, ir: RootNode) -> tuple[list[dict], dict[str, str]]:
        return self.linker.link(ir)
