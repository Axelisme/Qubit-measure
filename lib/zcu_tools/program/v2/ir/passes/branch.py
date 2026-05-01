from __future__ import annotations

from typing import Optional, Tuple, Union

from ..node import IRBranch, IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class BranchCaseNormalizePass(AbsPipeLinePass, IRTransformer):
    """Normalize branch case metadata order without changing emitted instructions."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return res or ir

    def visit_IRBranch(self, node: IRBranch) -> Optional[IRNode]:
        self.generic_visit(node)
        node.cases.sort(key=lambda case: _case_sort_key(case.name))
        return node


def _case_sort_key(name: str) -> Tuple[int, Union[int, str]]:
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)
