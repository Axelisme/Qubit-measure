from __future__ import annotations

from typing_extensions import Optional

from ..instructions import GenericInst, Instruction
from ..node import BlockNode, IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class ZeroDelayDCEPass(AbsPipeLinePass, IRTransformer):
    """Remove lower-level zero reference-time increments."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return res or ir

    def visit_GenericInst(self, node: GenericInst) -> Optional[IRNode]:
        if _is_zero_delay_inst(node):
            return None
        return node


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


class TimedInstructionMergePass(AbsPipeLinePass, IRTransformer):
    """Merge adjacent reference-time increments with identical semantics."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return res or ir

    def visit_BlockNode(self, node: BlockNode) -> Optional[IRNode]:
        # Use generic_visit to handle recursion into structural children (like in IRLoop or IRBranch)
        # but we also need to handle BlockNode.insts ourselves if we want to merge.
        # Actually, generic_visit(node) will visit each item in node.insts.

        # Correct approach: let generic_visit handle recursion first
        self.generic_visit(node)

        # Perform merging on node.insts
        rewritten: list[IRNode] = []
        pending_inst: Optional[GenericInst] = None
        pending_value = 0
        pending_count = 0

        def flush_pending() -> None:
            nonlocal pending_inst, pending_value, pending_count
            if pending_inst is None:
                return
            if pending_count == 1:
                rewritten.append(pending_inst)
            else:
                rewritten.append(
                    GenericInst(
                        cmd="TIME",
                        args={"C_OP": "inc_ref", "LIT": f"#{pending_value}"},
                        line=pending_inst.line,
                        p_addr=pending_inst.p_addr,
                    )
                )
            pending_inst = None
            pending_value = 0
            pending_count = 0

        for item in node.insts:
            merge_value = (
                _positive_time_increment(item)
                if isinstance(item, Instruction)
                else None
            )

            if merge_value is None:
                flush_pending()
                rewritten.append(item)
            elif not isinstance(item, GenericInst):
                flush_pending()
                rewritten.append(item)
            else:
                if pending_inst is None:
                    pending_inst = item
                    pending_value = merge_value
                    pending_count = 1
                else:
                    pending_value += merge_value
                    pending_count += 1

        flush_pending()
        node.insts = rewritten
        return node


def _positive_time_increment(inst: Instruction) -> Optional[int]:
    if not isinstance(inst, GenericInst):
        return None
    if inst.cmd != "TIME":
        return None
    if set(inst.args) != {"C_OP", "LIT"}:
        return None
    if inst.args.get("C_OP") != "inc_ref":
        return None

    lit = inst.args.get("LIT")
    if not isinstance(lit, str) or not lit.startswith("#"):
        return None

    try:
        value = int(lit[1:])
    except ValueError:
        return None

    if value <= 0:
        return None
    return value
