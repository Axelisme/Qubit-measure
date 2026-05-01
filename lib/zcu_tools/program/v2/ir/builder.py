from __future__ import annotations

from .factory import BuildContext, build_from_instruction
from .instructions import Instruction
from .node import IRNode, RootNode


class IRBuilder:
    def build(self, prog_list: list[dict], labels: dict[str, str]) -> RootNode:
        root = RootNode(labels=labels)
        ctx = BuildContext(root=root)

        for d in prog_list:
            inst = Instruction.from_dict(d)
            build_from_instruction(inst, ctx)

        if len(ctx.block_stack) != 1:
            raise ValueError("Unclosed blocks in IRBuilder")
        if len(ctx.struct_stack) != 0:
            raise ValueError("Unclosed structures in IRBuilder")
        if len(ctx.case_stack) != 0:
            raise ValueError("Unclosed branch cases in IRBuilder")

        return root

    def unbuild(self, ir: IRNode) -> tuple[list[dict], dict[str, str]]:
        if not isinstance(ir, RootNode):
            raise ValueError("IR node passed to unbuild must be a RootNode")

        prog_list: list[dict] = []
        ir.emit(prog_list)
        return prog_list, ir.labels
