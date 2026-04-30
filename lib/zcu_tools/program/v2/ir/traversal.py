from __future__ import annotations

from collections.abc import Iterator

from .instructions import Instruction
from .node import BlockNode, IRLoop, IRNode


def iter_child_nodes(node: IRNode) -> Iterator[IRNode]:
    """Yield structural child nodes without duplicating branch case blocks."""
    if isinstance(node, IRLoop):
        yield node.initial
        yield node.stop_check
        yield node.body
        yield node.update
        yield node.jump_back
        return

    if isinstance(node, BlockNode):
        for item in node.insts:
            if isinstance(item, IRNode):
                yield item


def walk_nodes(node: IRNode) -> Iterator[IRNode]:
    seen: set[int] = set()

    def _walk(current: IRNode) -> Iterator[IRNode]:
        ident = id(current)
        if ident in seen:
            return
        seen.add(ident)
        yield current
        for child in iter_child_nodes(current):
            yield from _walk(child)

    yield from _walk(node)


def walk_instructions(node: IRNode) -> Iterator[Instruction]:
    for current in walk_nodes(node):
        if isinstance(current, BlockNode):
            for item in current.insts:
                if isinstance(item, Instruction):
                    yield item
