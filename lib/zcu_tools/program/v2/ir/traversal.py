from __future__ import annotations

from collections.abc import Iterator
from typing import List, Optional, Union

from .instructions import Instruction
from .node import BlockNode, IRBranch, IRBranchCase, IRLoop, IRNode, RootNode


class IRTransformer:
    """Base class for IR transformations with automatic recursion."""

    def visit(self, node: IRNode) -> Union[IRNode, List[IRNode], None]:
        """Visit a node, returning a new node, a list of nodes, the same node, or None to delete."""
        if node is None:
            return None

        # Dispatch based on type
        if isinstance(node, RootNode):
            return self.visit_RootNode(node)
        if isinstance(node, IRBranchCase):
            return self.visit_IRBranchCase(node)
        if isinstance(node, IRBranch):
            return self.visit_IRBranch(node)
        if isinstance(node, BlockNode):
            return self.visit_BlockNode(node)
        if isinstance(node, IRLoop):
            return self.visit_IRLoop(node)
        if isinstance(node, Instruction):
            return self.visit_Instruction(node)

        return self.generic_visit(node)

    def visit_RootNode(self, node: RootNode) -> Union[IRNode, List[IRNode], None]:
        return self.visit_BlockNode(node)

    def visit_IRBranchCase(self, node: IRBranchCase) -> Union[IRNode, List[IRNode], None]:
        return self.visit_BlockNode(node)

    def visit_IRBranch(self, node: IRBranch) -> Union[IRNode, List[IRNode], None]:
        return self.generic_visit(node)

    def visit_BlockNode(self, node: BlockNode) -> Union[IRNode, List[IRNode], None]:
        return self.generic_visit(node)

    def visit_IRLoop(self, node: IRLoop) -> Union[IRNode, List[IRNode], None]:
        return self.generic_visit(node)

    def visit_Instruction(self, node: Instruction) -> Union[IRNode, List[IRNode], None]:
        # Instruction subclasses can be handled here or in generic_visit
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: IRNode) -> Union[IRNode, List[IRNode], None]:
        """Default visitor that recurses into child nodes using structural definition."""
        
        # We use node.children() to identify what to visit, 
        # but for reconstruction we need to know WHICH attribute to update.
        # For simplicity and Python 3.9, we handle known structural types explicitly.
        
        if isinstance(node, IRLoop):
            node.initial = self._visit_block(node.initial)
            node.stop_check = self._visit_block(node.stop_check)
            node.body = self._visit_block(node.body)
            node.update = self._visit_block(node.update)
            node.jump_back = self._visit_block(node.jump_back)
        elif isinstance(node, IRBranch):
            node.dispatch = self._visit_block(node.dispatch)
            new_cases = []
            for c in node.cases:
                v = self.visit(c)
                if v is None:
                    continue
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, IRBranchCase):
                            new_cases.append(item)
                elif isinstance(v, IRBranchCase):
                    new_cases.append(v)
            node.cases = new_cases
        elif isinstance(node, BlockNode):
            new_insts = []
            for i in node.insts:
                v = self.visit(i)
                if v is None:
                    continue
                if isinstance(v, list):
                    new_insts.extend(v)
                else:
                    new_insts.append(v)
            node.insts = new_insts

        return node

    def _visit_block(self, node: BlockNode) -> BlockNode:
        """Helper to visit a block and ensure it remains a BlockNode."""
        visited = self.visit(node)
        if visited is None:
            return BlockNode()
        if isinstance(visited, list):
            return BlockNode(insts=visited)
        if not isinstance(visited, BlockNode):
            raise TypeError(f"Expected BlockNode, got {type(visited)}")
        return visited


def iter_child_nodes(node: IRNode) -> Iterator[IRNode]:
    """Yield structural child nodes."""
    return node.children()


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
        if isinstance(current, Instruction):
            yield current
