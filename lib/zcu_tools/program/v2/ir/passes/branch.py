from __future__ import annotations

from ..node import IRBranch, IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import walk_nodes


class BranchCaseNormalizePass(AbsPipeLinePass):
    """Normalize branch case metadata order without changing emitted instructions."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        for node in walk_nodes(ir):
            if isinstance(node, IRBranch):
                node.cases.sort(key=lambda case: _case_sort_key(case.name))
        return ir


def _case_sort_key(name: str) -> tuple[int, int | str]:
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)
