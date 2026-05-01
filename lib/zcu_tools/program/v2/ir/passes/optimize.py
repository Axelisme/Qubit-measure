from __future__ import annotations

from typing_extensions import Optional, Set, Tuple, Union

from ..analysis import (
    instruction_reads,
    instruction_writes,
    is_marked_hoistable,
    strip_internal_annotations,
)
from ..instructions import Instruction
from ..node import BlockNode, IRLoop, IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class LoopInvariantHoistPass(AbsPipeLinePass, IRTransformer):
    """Hoist explicitly marked invariant instructions into the loop initial block."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return res or ir

    def visit_IRLoop(self, node: IRLoop) -> Optional[IRNode]:
        self.generic_visit(node)
        self._hoist_loop(node)
        return node

    def _hoist_loop(self, loop: IRLoop) -> None:
        loop_control_writes = _block_reads_writes(loop.initial)[1].union(
            _block_reads_writes(loop.update)[1]
        )
        loop_control_reads = _block_reads_writes(loop.stop_check)[0].union(
            _block_reads_writes(loop.update)[0]
        )
        blocked_regs = loop_control_writes.union(loop_control_reads)

        hoisted: list[Instruction] = []
        remaining = []
        for item in loop.body.insts:
            if (
                isinstance(item, Instruction)
                and is_marked_hoistable(item)
                and instruction_reads(item).isdisjoint(blocked_regs)
                and instruction_writes(item).isdisjoint(blocked_regs)
            ):
                hoisted.append(strip_internal_annotations(item))
            else:
                remaining.append(item)

        if hoisted:
            loop.initial.insts.extend(hoisted)
            loop.body.insts = remaining


class PeepholePass(AbsPipeLinePass, IRTransformer):
    """Apply local cleanups that do not remove executable no-op instructions."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return res or ir

    def visit_Instruction(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)


def _block_reads_writes(block: BlockNode) -> Tuple[Set[str], Set[str]]:
    reads: set[str] = set()
    writes: set[str] = set()
    for item in block.insts:
        if isinstance(item, Instruction):
            reads.update(instruction_reads(item))
            writes.update(instruction_writes(item))
    return reads, writes
