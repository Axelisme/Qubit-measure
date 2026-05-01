from __future__ import annotations

from .factory import InstructionStream, parse_root
from .instructions import Instruction
from .linker import IRLinker
from .node import IRNode, RootNode


class IRBuilder:
    def build(self, prog_list: list[dict]) -> RootNode:
        inst_list = [Instruction.from_dict(d) for d in prog_list]
        stream = InstructionStream(inst_list)
        root = parse_root(stream)

        if stream.peek() is not None:
            raise ValueError("Unparsed instructions remaining in stream")

        return root

    def unbuild(self, ir: IRNode) -> tuple[list[dict], dict[str, str]]:
        if not isinstance(ir, RootNode):
            raise ValueError("IR node passed to unbuild must be a RootNode")

        linker = IRLinker()
        return linker.link(ir)
