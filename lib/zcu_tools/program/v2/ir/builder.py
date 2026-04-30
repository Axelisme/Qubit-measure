from __future__ import annotations

from .node import IRNode
from .instructions import Instruction


class IRBuilder:
    def build(self, prog_list: list[dict], labels: dict[str, str]) -> IRNode:
        insts = [Instruction.from_dict(d) for d in prog_list]
        return IRNode(insts, labels)

    def unbuild(self, ir: IRNode) -> tuple[list[dict], dict[str, str]]:
        prog_list = [inst.to_dict() for inst in ir.insts]
        return prog_list, ir.labels
