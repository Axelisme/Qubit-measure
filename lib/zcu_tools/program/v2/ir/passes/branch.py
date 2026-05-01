from __future__ import annotations

from typing import Optional, Tuple, Union, cast

from ..node import IRBranch, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import IRTransformer


class BranchCaseNormalizePass(AbsPipeLinePass, IRTransformer):
    """Normalize branch case metadata order without changing emitted instructions."""

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        res = self.visit(ir)
        if isinstance(res, list):
            raise ValueError("Root node cannot be unrolled into a list")
        return cast(RootNode, res or ir)

    def visit_IRBranch(self, node: IRBranch) -> Optional[IRNode]:
        self.generic_visit(node)
        node.cases.sort(key=lambda case: _case_sort_key(case.name))
        return node


def _case_sort_key(name: str) -> Tuple[int, Union[int, str]]:
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)
