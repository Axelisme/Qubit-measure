from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import List, Union

from .instructions import Instruction
from .node import BasicBlockNode, BlockNode, IRNode


class IRTransformer:
    """Base class for IR transformations with automatic recursion."""

    def visit(self, node: IRNode) -> Union[IRNode, List[IRNode], None]:
        """Visit a node, returning a new node, a list of nodes, the same node, or None to delete."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def visit_BasicBlockNode(
        self, node: BasicBlockNode
    ) -> Union[IRNode, List[IRNode], None]:
        """Default visitor for BasicBlockNode: return unchanged (no child IRNodes)."""
        return node

    def generic_visit(self, node: IRNode) -> Union[IRNode, List[IRNode], None]:
        """Default visitor that automatically recurses into child nodes using dataclass fields."""
        if not dataclasses.is_dataclass(node):
            return node

        for field in dataclasses.fields(node):
            old_value = getattr(node, field.name)

            # 1. Handle lists of IRNodes (e.g., BlockNode.insts, IRBranch.cases)
            if isinstance(old_value, list):
                new_values = []
                for item in old_value:
                    if isinstance(item, IRNode):
                        res = self.visit(item)
                        if res is None:
                            continue
                        elif isinstance(res, list):
                            new_values.extend(res)
                        else:
                            new_values.append(res)
                    else:
                        new_values.append(item)
                setattr(node, field.name, new_values)

            # 2. Handle single IRNode fields (e.g., IRLoop.body, IRBranch.dispatch)
            elif isinstance(old_value, IRNode):
                res = self.visit(old_value)

                # Protection: Structural fields expecting a BlockNode shouldn't be destroyed
                if res is None:
                    setattr(node, field.name, BlockNode())
                elif isinstance(res, list):
                    setattr(node, field.name, BlockNode(insts=res))
                else:
                    setattr(node, field.name, res)

        return node


def walk_nodes(node: IRNode) -> Iterator[IRNode]:
    seen: set[int] = set()

    def _walk(current: IRNode) -> Iterator[IRNode]:
        ident = id(current)
        if ident in seen:
            return
        seen.add(ident)
        yield current
        for child in current.children():
            yield from _walk(child)

    yield from _walk(node)


def walk_basic_blocks(node: IRNode) -> Iterator[BasicBlockNode]:
    """Yield every BasicBlockNode reachable from node (depth-first)."""
    for current in walk_nodes(node):
        if isinstance(current, BasicBlockNode):
            yield current


def walk_instructions(node: IRNode) -> Iterator[Instruction]:
    """Yield every Instruction reachable from node (labels, insts, branch)."""
    for current in walk_nodes(node):
        if isinstance(current, BasicBlockNode):
            yield from current.labels
            yield from current.insts
            if current.branch is not None:
                yield current.branch
