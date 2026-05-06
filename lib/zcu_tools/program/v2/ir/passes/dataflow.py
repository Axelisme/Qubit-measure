from __future__ import annotations

from typing import Optional

from ..analysis import instruction_reads, instruction_writes
from ..instructions import Instruction, NopInst
from ..node import BasicBlockNode, BlockNode, InstNode, IRNode, RootNode
from ..pipeline import AbsIRPass, AbsLinearPass, LinearPipeline, PipeLineContext
from ..traversal import IRTransformer
from .base import OptimizationPassBase, is_safe_linear_inst


class DeadWriteEliminationLinear(AbsLinearPass):
    """Remove overwritten register writes in a BasicBlockNode.

    fix_inst_num=False: removes dead-write instructions from the list.
    fix_inst_num=True:  replaces dead-write instructions with NopInst to
                        preserve jump-table stride.
    """

    def process_block(self, block: BasicBlockNode) -> None:
        insts = block.insts
        dead: set[int] = self._find_dead_indices(insts)
        if not dead:
            return
        if block.fix_inst_num:
            block.insts = [
                NopInst() if i in dead else inst
                for i, inst in enumerate(insts)
            ]
        else:
            block.insts = [
                inst for i, inst in enumerate(insts) if i not in dead
            ]

    def _find_dead_indices(self, insts: list[Instruction]) -> set[int]:
        pending: dict[str, int] = {}  # reg -> index of last pending write
        dead: set[int] = set()

        for idx, inst in enumerate(insts):
            if not is_safe_linear_inst(inst):
                pending.clear()
                continue

            reads = instruction_reads(inst)
            writes = list(instruction_writes(inst))

            if len(writes) > 1:
                pending.clear()
                continue

            for reg in reads:
                pending.pop(reg, None)

            if len(writes) == 1:
                dst = writes[0]
                prev_idx = pending.get(dst)
                if prev_idx is not None:
                    dead.add(prev_idx)
                pending[dst] = idx

        return dead


class DeadWriteEliminationLegacyPass(AbsIRPass, IRTransformer):
    """Remove overwritten writes in legacy InstNode-based BlockNodes.

    The BasicBlockNode path is handled by LinearPipeline(DeadWriteEliminationLinear).
    This pass only touches InstNode content inside BlockNode / RootNode.
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        if not ctx.config.enable_dead_write:
            return ir
        self.ctx = ctx
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        return res or ir  # type: ignore[return-value]

    def visit_BasicBlockNode(self, node: BasicBlockNode) -> Optional[IRNode]:
        return node  # handled by LinearPipeline

    def visit_BlockNode(self, node: BlockNode) -> Optional[IRNode | list[IRNode]]:
        rewritten: list[IRNode | None] = []
        pending_writes: dict[str, int] = {}

        def flush_pending() -> None:
            pending_writes.clear()

        def append_item(item: IRNode | list[IRNode] | None) -> None:
            if item is None:
                return
            if isinstance(item, list):
                for child in item:
                    append_item(child)
                return
            rewritten.append(item)

        for item in node.insts:
            if not isinstance(item, InstNode):
                flush_pending()
                visited = self.visit(item)
                if visited is None:
                    continue
                append_item(visited)
                flush_pending()
                continue

            inst = item.inst
            if not is_safe_linear_inst(inst):
                flush_pending()
                rewritten.append(item)
                continue

            reads = instruction_reads(inst)
            writes = list(instruction_writes(inst))

            if len(writes) > 1:
                flush_pending()
                rewritten.append(item)
                continue

            for reg in reads:
                pending_writes.pop(reg, None)

            if len(writes) == 1:
                dst = writes[0]
                prev_idx = pending_writes.get(dst)
                if prev_idx is not None and 0 <= prev_idx < len(rewritten):
                    if rewritten[prev_idx] is not None:
                        rewritten[prev_idx] = None
                pending_writes[dst] = len(rewritten)

            rewritten.append(item)

        node.insts = [item for item in rewritten if item is not None]
        return node

    visit_RootNode = visit_BlockNode
    visit_IRBranchCase = visit_BlockNode


class DeadWriteEliminationPass(AbsIRPass):
    """Compatibility wrapper: run DeadWriteEliminationLinear + Legacy pass.

    Kept so existing code that imports DeadWriteEliminationPass still works.
    New code should use DeadWriteEliminationLinear inside a LinearPipeline
    and DeadWriteEliminationLegacyPass as an AbsIRPass.
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        if not ctx.config.enable_dead_write:
            return ir
        LinearPipeline(DeadWriteEliminationLinear()).process(ir)
        DeadWriteEliminationLegacyPass().process(ir, ctx)
        return ir
