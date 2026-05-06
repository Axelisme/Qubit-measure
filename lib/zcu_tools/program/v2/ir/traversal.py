from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import List, Union

from .instructions import Instruction
from .node import BlockNode, InstNode, IRNode


class IRTransformer:
    """Base class for IR transformations with automatic recursion."""

    @staticmethod
    def _normalize_inst_visit_result(
        res: Union[Instruction, IRNode, List[Union[Instruction, IRNode]], None],
    ) -> Union[IRNode, List[IRNode], None]:
        if res is None:
            return None
        if isinstance(res, list):
            return [
                InstNode(item) if isinstance(item, Instruction) else item
                for item in res
            ]
        return InstNode(res) if isinstance(res, Instruction) else res

    def visit(self, node: IRNode) -> Union[IRNode, List[IRNode], None]:
        """Visit a node, returning a new node, a list of nodes, the same node, or None to delete."""
        # Dynamic dispatch based on class name
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def visit_InstNode(self, node: InstNode) -> Union[IRNode, List[IRNode], None]:
        """Dispatch to visitor for the wrapped instruction."""
        inst = node.inst
        method_name = f"visit_{inst.__class__.__name__}"
        visitor = getattr(self, method_name, None)
        if visitor:
            return self._normalize_inst_visit_result(visitor(inst))

        return self.generic_visit(node)

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


def walk_instructions(node: IRNode) -> Iterator[Instruction]:
    for current in walk_nodes(node):
        if isinstance(current, InstNode):
            yield current.inst
