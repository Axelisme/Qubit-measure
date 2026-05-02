from __future__ import annotations

from typing import Optional

from ..analysis import instruction_reads, instruction_writes
from ..node import BlockNode, InstNode, IRNode
from .base import OptimizationPassBase, is_safe_linear_inst


class DeadWriteEliminationPass(OptimizationPassBase):
    """Remove overwritten writes in straight-line blocks."""

    def visit_BlockNode(self, node: BlockNode) -> Optional[IRNode | list[IRNode]]:
        if not self.ctx.config.enable_dead_write:
            return self.generic_visit(node)

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
                        self._bump_stat("dead_write.removed")
                pending_writes[dst] = len(rewritten)

            rewritten.append(item)

        node.insts = [item for item in rewritten if item is not None]
        return node

    visit_RootNode = visit_BlockNode
    visit_IRBranchCase = visit_BlockNode
