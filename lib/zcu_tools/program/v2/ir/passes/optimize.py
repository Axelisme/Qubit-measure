from __future__ import annotations

from typing import cast

from typing_extensions import Optional, Set, Tuple, Union

from ..analysis import (
    instruction_reads,
    instruction_writes,
    is_marked_hoistable,
    strip_internal_annotations,
)
from ..instructions import Instruction
from ..node import BlockNode, InstNode, IRLoop, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class LoopInvariantHoistPass(AbsPipeLinePass, IRTransformer):
    """Hoist explicitly marked invariant instructions into the loop initial block."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return cast(RootNode, res or ir)

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
            if isinstance(item, InstNode):
                inst = item.inst
                if (
                    is_marked_hoistable(inst)
                    and instruction_reads(inst).isdisjoint(blocked_regs)
                    and instruction_writes(inst).isdisjoint(blocked_regs)
                ):
                    hoisted.append(InstNode(strip_internal_annotations(inst)))
                else:
                    remaining.append(item)
            else:
                remaining.append(item)

        if hoisted:
            loop.body.insts = remaining
            
        return hoisted


class PeepholePass(AbsPipeLinePass, IRTransformer):
    """Apply local cleanups that do not remove executable no-op instructions."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return cast(RootNode, res or ir)

    def visit_GenericInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)

    def visit_RegWriteInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)
        
    def visit_PortWriteInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)

    def visit_TimeInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)

    def visit_LabelInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)

    def visit_TestInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)
        
    def visit_JumpInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)
        
    def visit_NopInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)
        
    def visit_DmemReadInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)
        
    def visit_DmemWriteInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)
        
    def visit_DportWriteInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)
        
    def visit_WaitInst(self, inst: Instruction) -> Optional[Instruction]:
        return strip_internal_annotations(inst)


def _block_reads_writes(block: BlockNode) -> Tuple[Set[str], Set[str]]:
    reads: set[str] = set()
    writes: set[str] = set()
    for item in block.insts:
        if isinstance(item, InstNode):
            reads.update(instruction_reads(item.inst))
            writes.update(instruction_writes(item.inst))
    return reads, writes
