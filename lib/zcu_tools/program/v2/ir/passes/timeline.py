from __future__ import annotations

from typing import cast

from typing_extensions import Optional

from ..instructions import Instruction, TimeInst
from ..node import BlockNode, InstNode, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class TimelinePassBase(AbsPipeLinePass, IRTransformer):
    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        return cast(RootNode, res or ir)


def _is_zero_ref_increment(inst: Instruction) -> bool:
    if not isinstance(inst, TimeInst):
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


class ZeroDelayDCEPass(TimelinePassBase):
    """Remove lower-level zero reference-time increments."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        if not ctx.config.enable_zero_delay_dce:
            return ir
        return super().process(ir, ctx)

    def visit_TimeInst(self, inst: TimeInst) -> Optional[Instruction]:
        if _is_zero_ref_increment(inst):
            return None
        return inst


def _is_mergeable_time_increment(inst: Instruction) -> bool:
    if not isinstance(inst, TimeInst):
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
        value = int(lit[1:])
    except ValueError:
        return False

    return value > 0


def _merged_time_run(run: list[TimeInst]) -> InstNode:
    if len(run) == 1:
        return InstNode(run[0])

    total = sum(int(inst.lit[1:]) for inst in run if inst.lit is not None)
    return InstNode(
        TimeInst(
            c_op="inc_ref",
            lit=f"#{total}",
        )
    )


class TimedMergePass(TimelinePassBase):
    """Merge adjacent reference-time increments with identical semantics."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        if not ctx.config.enable_timed_instruction_merge:
            return ir
        return super().process(ir, ctx)

    def visit_BlockNode(self, node: BlockNode) -> Optional[IRNode]:
        self.generic_visit(node)

        rewritten: list[IRNode] = []
        pending_run: list[TimeInst] = []

        def flush_pending() -> None:
            if not pending_run:
                return
            rewritten.append(_merged_time_run(pending_run))
            pending_run.clear()

        for item in node.insts:
            if not isinstance(item, InstNode):
                flush_pending()
                rewritten.append(item)
                continue

            if not isinstance(item.inst, TimeInst):
                flush_pending()
                rewritten.append(item)
                continue

            if not _is_mergeable_time_increment(item.inst):
                flush_pending()
                rewritten.append(item)
            else:
                pending_run.append(item.inst)

        flush_pending()
        node.insts = rewritten
        return node

    def visit_RootNode(self, node: RootNode) -> Optional[IRNode]:
        return self.visit_BlockNode(node)

    def visit_IRBranchCase(self, node: BlockNode) -> Optional[IRNode]:
        return self.visit_BlockNode(node)
