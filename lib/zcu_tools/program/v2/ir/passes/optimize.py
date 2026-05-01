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

    def visit_IRLoop(self, node: IRLoop) -> Union[IRNode, list[IRNode], None]:
        self.generic_visit(node)
        hoisted = self._hoist_loop(node)
        if hoisted:
            return hoisted + [node]
        return node

    def _hoist_loop(self, loop: IRLoop) -> list[IRNode]:
        from ..utils import regs_from_value
        # The loop control register is written and read. The limit n might be a register.
        blocked_regs = {loop.counter_reg}
        if isinstance(loop.n, str):
            blocked_regs.update(regs_from_value(loop.n))

        hoisted: list[IRNode] = []
        remaining: list[IRNode] = []
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
            loop.body.insts = remaining
            
        return hoisted


class PeepholePass(AbsPipeLinePass, IRTransformer):
    """Apply local cleanups that do not remove executable no-op instructions."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return res or ir

    def visit_GenericInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)

    def visit_RegWriteInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)
        
    def visit_PortWriteInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)

    def visit_TimeInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)

    def visit_LabelInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)

    def visit_TestInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)
        
    def visit_JumpInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)
        
    def visit_NopInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)
        
    def visit_DmemReadInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)
        
    def visit_DmemWriteInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)
        
    def visit_DportWriteInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)
        
    def visit_WaitInst(self, node: Instruction) -> Union[IRNode, list[IRNode], None]:
        return strip_internal_annotations(node)


def _block_reads_writes(block: BlockNode) -> Tuple[Set[str], Set[str]]:
    reads: set[str] = set()
    writes: set[str] = set()
    for item in block.insts:
        if isinstance(item, Instruction):
            reads.update(instruction_reads(item))
            writes.update(instruction_writes(item))
    return reads, writes
