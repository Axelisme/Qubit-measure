from __future__ import annotations

from typing import Optional

from ..analysis import instruction_reads, instruction_writes
from ..instructions import Instruction
from ..node import BasicBlockNode, BlockNode, InstNode, IRNode
from .base import AbsLinearPass, LinearPassAdapter, OptimizationPassBase, is_safe_linear_inst


class DeadWriteEliminationLinear(AbsLinearPass):
    """Remove overwritten register writes in a flat instruction list.

    Only operates on instructions that pass `is_safe_linear_inst` — control
    flow and memory-mapped instructions are left in place and flush the
    pending-write tracking.
    """

    def process_linear(self, insts: list[Instruction]) -> list[Instruction]:
        result: list[Instruction | None] = []
        pending: dict[str, int] = {}  # reg -> index in result

        for inst in insts:
            if not is_safe_linear_inst(inst):
                pending.clear()
                result.append(inst)
                continue

            reads = instruction_reads(inst)
            writes = list(instruction_writes(inst))

            if len(writes) > 1:
                pending.clear()
                result.append(inst)
                continue

            for reg in reads:
                pending.pop(reg, None)

            if len(writes) == 1:
                dst = writes[0]
                prev_idx = pending.get(dst)
                if prev_idx is not None and result[prev_idx] is not None:
                    result[prev_idx] = None
                pending[dst] = len(result)

            result.append(inst)

        return [inst for inst in result if inst is not None]


class DeadWriteEliminationPass(OptimizationPassBase):
    """Pipeline pass: remove overwritten writes in straight-line blocks.

    Applies DeadWriteEliminationLinear to every BasicBlockNode.insts via
    LinearPassAdapter (skips fix_inst_num blocks), then falls back to the
    legacy IRTransformer path for any remaining InstNode / BlockNode content.
    """

    def process(self, ir, ctx):
        if not ctx.config.enable_dead_write:
            return ir
        self.ctx = ctx
        LinearPassAdapter(DeadWriteEliminationLinear()).process(ir, ctx)
        # Legacy: also clean up old InstNode-based BlockNodes.
        return super().process(ir, ctx)

    def visit_BasicBlockNode(self, node: BasicBlockNode) -> Optional[IRNode]:
        return node  # already handled by LinearPassAdapter

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
