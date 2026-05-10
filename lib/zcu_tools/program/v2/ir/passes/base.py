from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from typing_extensions import List, Union, cast

from ..instructions import Instruction
from ..node import BasicBlockNode, BlockNode, IRNode, RootNode
from ..pipeline import AbsIRPass, PipeLineContext


class IRTransformer:
    """Base class for IR transformations with automatic recursion.

    Subclasses (or callers wrapping ``visit``) should reset ``_changed`` to
    ``False`` before a top-level traversal and read it afterwards to learn
    whether any in-place mutation occurred.  ``generic_visit`` sets the flag
    whenever it inserts, removes, or replaces a child node by identity.
    """

    _changed: bool = False

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
                new_values: list = []
                list_changed = False
                for item in old_value:
                    if isinstance(item, IRNode):
                        res = self.visit(item)
                        if res is None:
                            list_changed = True
                            continue
                        elif isinstance(res, list):
                            list_changed = True
                            new_values.extend(res)
                        else:
                            if res is not item:
                                list_changed = True
                            new_values.append(res)
                    else:
                        new_values.append(item)
                if list_changed or len(new_values) != len(old_value):
                    self._changed = True
                setattr(node, field.name, new_values)

            # 2. Handle single IRNode fields (e.g., IRLoop.body, IRBranch.dispatch)
            elif isinstance(old_value, IRNode):
                res = self.visit(old_value)

                # Protection: Structural fields expecting a BlockNode shouldn't be destroyed
                if res is None:
                    self._changed = True
                    setattr(node, field.name, BlockNode())
                elif isinstance(res, list):
                    self._changed = True
                    setattr(node, field.name, BlockNode(insts=res))
                else:
                    if res is not old_value:
                        self._changed = True
                    setattr(node, field.name, res)

        return node


def _walk_nodes(node: IRNode) -> Iterator[IRNode]:
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
    for current in _walk_nodes(node):
        if isinstance(current, BasicBlockNode):
            yield current


def walk_instructions(node: IRNode) -> Iterator[Instruction]:
    """Yield every Instruction reachable from node (labels, insts, branch)."""
    for current in _walk_nodes(node):
        if isinstance(current, BasicBlockNode):
            yield from current.labels
            yield from current.insts
            if current.branch is not None:
                yield current.branch


class OptimizationPassBase(AbsIRPass, IRTransformer):
    """Base class for IR-level optimization passes with recursive IR traversal."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> tuple[RootNode, bool]:
        self.ctx = ctx
        self._changed = False
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Unexpected list returned from visit")
        out = cast(RootNode, res or ir)
        changed = self._changed or out is not ir
        return out, changed
