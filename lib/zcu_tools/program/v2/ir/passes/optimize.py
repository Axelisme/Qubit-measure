from __future__ import annotations

from ..analysis import (
    instruction_reads,
    instruction_writes,
    is_marked_hoistable,
    strip_internal_annotations,
)
from ..instructions import Instruction
from ..node import BlockNode, IRLoop, IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext


class LoopInvariantHoistPass(AbsPipeLinePass):
    """Hoist explicitly marked invariant instructions into the loop initial block."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        self._rewrite_node(ir)
        return ir

    def _rewrite_node(self, node: IRNode) -> None:
        if isinstance(node, IRLoop):
            self._rewrite_node(node.initial)
            self._rewrite_node(node.stop_check)
            self._rewrite_node(node.body)
            self._rewrite_node(node.update)
            self._rewrite_node(node.jump_back)
            self._hoist_loop(node)
            return

        if isinstance(node, BlockNode):
            for item in node.insts:
                if isinstance(item, IRNode):
                    self._rewrite_node(item)

    def _hoist_loop(self, loop: IRLoop) -> None:
        loop_control_writes = _block_writes(loop.initial) | _block_writes(loop.update)
        loop_control_reads = _block_reads(loop.stop_check) | _block_reads(loop.update)
        blocked_regs = loop_control_writes | loop_control_reads

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


class PeepholePass(AbsPipeLinePass):
    """Apply local cleanups that do not remove executable no-op instructions."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        self._rewrite_node(ir)
        return ir

    def _rewrite_node(self, node: IRNode) -> None:
        if isinstance(node, IRLoop):
            self._rewrite_node(node.initial)
            self._rewrite_node(node.stop_check)
            self._rewrite_node(node.body)
            self._rewrite_node(node.update)
            self._rewrite_node(node.jump_back)
            return

        if not isinstance(node, BlockNode):
            return

        rewritten = []
        for item in node.insts:
            if isinstance(item, Instruction):
                rewritten.append(strip_internal_annotations(item))
            elif isinstance(item, IRNode):
                self._rewrite_node(item)
                rewritten.append(item)
            else:
                rewritten.append(item)
        node.insts = rewritten


def _block_reads(block: BlockNode) -> set[str]:
    reads: set[str] = set()
    for item in block.insts:
        if isinstance(item, Instruction):
            reads.update(instruction_reads(item))
    return reads


def _block_writes(block: BlockNode) -> set[str]:
    writes: set[str] = set()
    for item in block.insts:
        if isinstance(item, Instruction):
            writes.update(instruction_writes(item))
    return writes
