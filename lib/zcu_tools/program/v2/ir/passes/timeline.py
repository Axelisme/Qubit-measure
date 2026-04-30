from __future__ import annotations

from ..instructions import GenericInst, Instruction
from ..node import BlockNode, IRLoop, IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext


class ZeroDelayDCEPass(AbsPipeLinePass):
    """Remove lower-level zero reference-time increments."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        self._rewrite_node(ir)
        return ir

    def _rewrite_node(self, node: IRNode) -> None:
        if isinstance(node, IRLoop):
            self._rewrite_node(node.initial)
            self._rewrite_node(node.stop_check)
            self._rewrite_node(node.body)
            self._rewrite_node(node.update)
            return

        if not isinstance(node, BlockNode):
            return

        rewritten = []
        for item in node.insts:
            if isinstance(item, Instruction):
                if not _is_zero_delay_inst(item):
                    rewritten.append(item)
            elif isinstance(item, IRNode):
                self._rewrite_node(item)
                rewritten.append(item)
            else:
                rewritten.append(item)
        node.insts = rewritten


def _is_zero_delay_inst(inst: Instruction) -> bool:
    if not isinstance(inst, GenericInst):
        return False
    if inst.cmd != "TIME":
        return False
    if set(inst.args) != {"C_OP", "LIT"}:
        return False
    if inst.args.get("C_OP") != "inc_ref":
        return False

    lit = inst.args.get("LIT")
    if not isinstance(lit, str) or not lit.startswith("#"):
        return False

    try:
        return int(lit[1:]) == 0
    except ValueError:
        return False
