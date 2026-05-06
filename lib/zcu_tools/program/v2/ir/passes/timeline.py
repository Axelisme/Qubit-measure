from __future__ import annotations

from typing import Optional, cast

from ..instructions import Instruction, NopInst, TimeInst
from ..node import BasicBlockNode, BlockNode, InstNode, IRNode, RootNode
from ..pipeline import AbsIRPass, AbsLinearPass, PipeLineContext
from ..traversal import IRTransformer


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
    """Remove TIME inc_ref #0 instructions from a BasicBlockNode.

    fix_inst_num=False: removes zero-delay TIME instructions.
    fix_inst_num=True:  replaces them with NopInst to preserve stride.
    """

    def process_block(self, block: BasicBlockNode) -> None:
        if block.fix_inst_num:
            block.insts = [
                NopInst() if _is_zero_ref_increment(inst) else inst
                for inst in block.insts
            ]
        else:
            block.insts = [
                inst for inst in block.insts if not _is_zero_ref_increment(inst)
            ]


class TimedMergeLinear(AbsLinearPass):
    """Merge adjacent TIME inc_ref #N instructions in a BasicBlockNode.

    fix_inst_num=False: merges adjacent runs into a single instruction.
    fix_inst_num=True:  merges the value into the first instruction of each
                        run, then replaces the remaining instructions with
                        NopInst to preserve stride.
    """

    def process_block(self, block: BasicBlockNode) -> None:
        if block.fix_inst_num:
            self._merge_fixed(block)
        else:
            self._merge_free(block)

    def _merge_free(self, block: BasicBlockNode) -> None:
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

        for inst in block.insts:
            if _is_mergeable_time_increment(inst):
                pending_run.append(cast(TimeInst, inst))
            else:
                flush()
                result.append(inst)

        flush()
        block.insts = result

    def _merge_fixed(self, block: BasicBlockNode) -> None:
        # Merge run values into the first slot; fill the rest with NOP.
        result: list[Instruction] = list(block.insts)
        i = 0
        while i < len(result):
            if not _is_mergeable_time_increment(result[i]):
                i += 1
                continue
            # Start of a run — find its extent.
            j = i + 1
            while j < len(result) and _is_mergeable_time_increment(result[j]):
                j += 1
            if j == i + 1:
                i += 1
                continue
            # Run from i to j-1: sum values into slot i, NOP out i+1..j-1.
            total = sum(
                int(cast(TimeInst, result[k]).lit[1:])  # type: ignore[union-attr]
                for k in range(i, j)
            )
            result[i] = TimeInst(c_op="inc_ref", lit=f"#{total}")
            for k in range(i + 1, j):
                result[k] = NopInst()
            i = j
        block.insts = result


# ---------------------------------------------------------------------------
# Legacy IRTransformer passes for InstNode / BlockNode content
# ---------------------------------------------------------------------------

class _LegacyBase(AbsIRPass, IRTransformer):
    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:  # noqa: ARG002
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        return cast(RootNode, res or ir)

    def visit_BasicBlockNode(self, node: BasicBlockNode) -> Optional[IRNode]:
        return node  # BasicBlockNode path handled by LinearPipeline


class ZeroDelayDCELegacyPass(_LegacyBase):
    """Remove TIME inc_ref #0 from legacy InstNode-based BlockNodes."""

    def visit_TimeInst(self, inst: TimeInst) -> Optional[Instruction]:
        if _is_zero_ref_increment(inst):
            return None
        return inst


class TimedMergeLegacyPass(_LegacyBase):
    """Merge adjacent TIME inc_ref in legacy InstNode-based BlockNodes."""

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


# ---------------------------------------------------------------------------
# Pipeline pass wrappers (backwards-compatible: check enable flag, run both
# BasicBlockNode via LinearPipeline and legacy InstNode via Legacy pass).
# These are kept so existing code that imports ZeroDelayDCEPass / TimedMergePass
# still works; new code should use ZeroDelayDCELinear / TimedMergeLinear
# directly inside a LinearPipeline.
# ---------------------------------------------------------------------------

class ZeroDelayDCEPass(AbsIRPass):
    """Compatibility wrapper: run ZeroDelayDCELinear + ZeroDelayDCELegacyPass."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        if not ctx.config.enable_zero_delay_dce:
            return ir
        from ..pipeline import LinearPipeline
        LinearPipeline(ZeroDelayDCELinear()).process(ir)
        ZeroDelayDCELegacyPass().process(ir, ctx)
        return ir


class TimedMergePass(AbsIRPass):
    """Compatibility wrapper: run TimedMergeLinear + TimedMergeLegacyPass."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        if not ctx.config.enable_timed_instruction_merge:
            return ir
        from ..pipeline import LinearPipeline
        LinearPipeline(TimedMergeLinear()).process(ir)
        TimedMergeLegacyPass().process(ir, ctx)
        return ir
