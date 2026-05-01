from __future__ import annotations

from typing import cast
from typing_extensions import Optional

from ..instructions import Instruction, TimeInst
from ..node import BlockNode, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class ZeroDelayDCEPass(AbsPipeLinePass, IRTransformer):
    """Remove lower-level zero reference-time increments."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return cast(RootNode, res or ir)

    def visit_TimeInst(self, node: TimeInst) -> Optional[IRNode]:
        if _is_zero_delay_inst(node):
            return None
        return node


def _is_zero_delay_inst(inst: Instruction) -> bool:
    if not isinstance(inst, TimeInst):
        return False
    if inst.annotations:
        return False
    if inst.c_op != "inc_ref":
        return False
    if inst.r1 is not None:
        return False
    if inst.lit is None:
        return False

    lit = inst.lit
    if not lit.startswith("#"):
        return False

    try:
        return int(lit[1:]) == 0
    except ValueError:
        return False


class TimedInstructionMergePass(AbsPipeLinePass, IRTransformer):
    """Merge adjacent reference-time increments with identical semantics."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return cast(RootNode, res or ir)

    def visit_BlockNode(self, node: BlockNode) -> Optional[IRNode]:
        # Use generic_visit to handle recursion first
        self.generic_visit(node)

        # Perform merging on node.insts
        rewritten: list[IRNode] = []
        pending_inst: Optional[TimeInst] = None
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
                    TimeInst(
                        c_op="inc_ref",
                        lit=f"#{pending_value}",
                        line=pending_inst.line,
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
            elif not isinstance(item, TimeInst):
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
        
    def visit_RootNode(self, node: BlockNode) -> Optional[IRNode]:
        return self.visit_BlockNode(node)

    def visit_IRBranchCase(self, node: BlockNode) -> Optional[IRNode]:
        return self.visit_BlockNode(node)


def _positive_time_increment(inst: Instruction) -> Optional[int]:
    if not isinstance(inst, TimeInst):
        return None
    if inst.annotations:
        return None
    if inst.c_op != "inc_ref":
        return None
    if inst.r1 is not None:
        return None
    if inst.lit is None:
        return None

    lit = inst.lit
    if not lit.startswith("#"):
        return None

    try:
        value = int(lit[1:])
    except ValueError:
        return None

    if value <= 0:
        return None
    return value
