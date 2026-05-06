from __future__ import annotations

from typing import Optional, cast

from ..instructions import Instruction, TimeInst
from ..node import BasicBlockNode, BlockNode, InstNode, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer
from .base import AbsLinearPass, LinearPassAdapter


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


# ---------------------------------------------------------------------------
# AbsLinearPass implementations
# ---------------------------------------------------------------------------

class ZeroDelayDCELinear(AbsLinearPass):
    """Remove TIME inc_ref #0 instructions from a flat instruction list."""

    def process_linear(self, insts: list[Instruction]) -> list[Instruction]:
        return [inst for inst in insts if not _is_zero_ref_increment(inst)]


class TimedMergeLinear(AbsLinearPass):
    """Merge adjacent TIME inc_ref #N instructions in a flat instruction list."""

    def process_linear(self, insts: list[Instruction]) -> list[Instruction]:
        result: list[Instruction] = []
        pending_run: list[TimeInst] = []

        def flush() -> None:
            if not pending_run:
                return
            if len(pending_run) == 1:
                result.append(pending_run[0])
            else:
                total = sum(int(t.lit[1:]) for t in pending_run if t.lit is not None)
                result.append(TimeInst(c_op="inc_ref", lit=f"#{total}"))
            pending_run.clear()

        for inst in insts:
            if _is_mergeable_time_increment(inst):
                pending_run.append(cast(TimeInst, inst))
            else:
                flush()
                result.append(inst)

        flush()
        return result


# ---------------------------------------------------------------------------
# Pipeline pass wrappers (check enable flag, then delegate to LinearPassAdapter)
# ---------------------------------------------------------------------------

class ZeroDelayDCEPass(AbsPipeLinePass):
    """Pipeline pass: remove zero-delay TIME inc_ref instructions.

    Operates on BasicBlockNode.insts via LinearPassAdapter (skips fix_inst_num
    blocks).  Falls back to the legacy IRTransformer path for any InstNode /
    BlockNode content not yet migrated.
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        if not ctx.config.enable_zero_delay_dce:
            return ir
        LinearPassAdapter(ZeroDelayDCELinear()).process(ir, ctx)
        # Legacy: also strip from old InstNode-based BlockNodes.
        _ZeroDelayDCELegacy().process(ir, ctx)
        return ir


class TimedMergePass(AbsPipeLinePass):
    """Pipeline pass: merge adjacent TIME inc_ref instructions.

    Operates on BasicBlockNode.insts via LinearPassAdapter, then applies the
    legacy BlockNode-level merge for any remaining InstNode content.
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        if not ctx.config.enable_timed_instruction_merge:
            return ir
        LinearPassAdapter(TimedMergeLinear()).process(ir, ctx)
        # Legacy: merge across old InstNode-based BlockNodes.
        _TimedMergeLegacy().process(ir, ctx)
        return ir


# ---------------------------------------------------------------------------
# Legacy IRTransformer paths (for InstNode / BlockNode content)
# These will be removed once all IR is BasicBlockNode-based.
# ---------------------------------------------------------------------------

class _LegacyTimelineBase(AbsPipeLinePass, IRTransformer):
    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:  # noqa: ARG002
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        return cast(RootNode, res or ir)

    def visit_BasicBlockNode(self, node: BasicBlockNode) -> Optional[IRNode]:
        return node  # already handled by LinearPassAdapter


class _ZeroDelayDCELegacy(_LegacyTimelineBase):
    def visit_TimeInst(self, inst: TimeInst) -> Optional[Instruction]:
        if _is_zero_ref_increment(inst):
            return None
        return inst


class _TimedMergeLegacy(_LegacyTimelineBase):
    def visit_BlockNode(self, node: BlockNode) -> Optional[IRNode]:
        self.generic_visit(node)

        rewritten: list[IRNode] = []
        pending_run: list[TimeInst] = []

        def flush_pending() -> None:
            if not pending_run:
                return
            if len(pending_run) == 1:
                rewritten.append(InstNode(pending_run[0]))
            else:
                total = sum(int(t.lit[1:]) for t in pending_run if t.lit is not None)
                rewritten.append(InstNode(TimeInst(c_op="inc_ref", lit=f"#{total}")))
            pending_run.clear()

        for item in node.insts:
            if not isinstance(item, InstNode):
                flush_pending()
                rewritten.append(item)
                continue
            if not _is_mergeable_time_increment(item.inst):
                flush_pending()
                rewritten.append(item)
            else:
                pending_run.append(cast(TimeInst, item.inst))

        flush_pending()
        node.insts = rewritten
        return node

    def visit_RootNode(self, node: RootNode) -> Optional[IRNode]:
        return self.visit_BlockNode(node)

    def visit_IRBranchCase(self, node: BlockNode) -> Optional[IRNode]:
        return self.visit_BlockNode(node)
