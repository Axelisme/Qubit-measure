from __future__ import annotations

from typing_extensions import Any

from .factory import InstructionStream, parse_root
from .instructions import Instruction
from .labels import Label
from .linker import IRCursor, IRLinker
from .node import RootNode


class IRBuilder:
    def __init__(self):
        self.linker = IRLinker()

    def build(
        self,
        prog_list: list[dict[str, Any]],
        labels: dict[str, Any],
        meta_infos: list[dict[str, Any]],
    ) -> RootNode:
        Label.reset()  # Must precede all Label.make_new() calls; see Label.label_set docstring
        inst_list = self.linker.unlink(prog_list, labels, meta_infos)

        stream = InstructionStream(inst_list)
        root = parse_root(stream)

        if stream.peek() is not None:
            raise ValueError("Unparsed instructions remaining in stream")

        return root

    def unbuild(
        self, ir: RootNode, *, pmem_size: int | None = None
    ) -> tuple[list[dict], dict[str, str], list[dict[str, Any]], IRCursor]:
        inst_list: list[Instruction] = []
        ir.emit(inst_list, pmem_size=pmem_size)
        return self.linker.link(inst_list)
