from __future__ import annotations

from qick.asm_v2 import AsmInst

from .node import IRNode


class IRBuilder:
    def build(self, prog_list: list[AsmInst], labels: dict[str, str]) -> IRNode:
        return IRNode(prog_list, labels)

    def unbuild(self, ir: IRNode) -> tuple[list[AsmInst], dict[str, str]]:
        return ir.insts, ir.labels
