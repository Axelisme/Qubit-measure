from __future__ import annotations


from .node import IRNode


class IRBuilder:
    def build(self, prog_list: list[dict], labels: dict[str, str]) -> IRNode:
        return IRNode(prog_list, labels)

    def unbuild(self, ir: IRNode) -> tuple[list[dict], dict[str, str]]:
        return ir.insts, ir.labels
