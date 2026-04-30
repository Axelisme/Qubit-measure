from __future__ import annotations

from ..node import IRNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..instructions import LabelInst


class LabelDCEPass(AbsPipeLinePass):
    """
    Dead Code Elimination for labels.
    Removes any label definitions from IRNode.labels that are not
    referenced by any instruction (LabelInst) in the program.
    """

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        used_labels: set[str] = set()

        for inst in ir.insts:
            if isinstance(inst, LabelInst):
                used_labels.add(inst.name)

        # Keep only labels that are actually used
        # Note: QICK also supports reserved labels like 'PREV', 'HERE', 'NEXT', 'SKIP'
        # which are not stored in ir.labels, so we only filter what's in ir.labels.
        new_labels = {
            name: addr for name, addr in ir.labels.items() if name in used_labels
        }

        # Return a new IRNode with the optimized labels
        return IRNode(ir.insts, new_labels)
