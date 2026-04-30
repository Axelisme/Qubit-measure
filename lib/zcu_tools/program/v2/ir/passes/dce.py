from __future__ import annotations

from ..instructions import Instruction, LabelInst
from ..node import BlockNode, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext


class LabelDCEPass(AbsPipeLinePass):
    """
    Dead Code Elimination for labels.
    Removes any label definitions from IRNode.labels that are not
    referenced by any instruction (LabelInst) in the program.
    """

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        if not isinstance(ir, RootNode):
            return ir  # Pass if it's not a RootNode

        used_labels: set[str] = set()

        def _find_labels(node: IRNode) -> None:
            if isinstance(node, BlockNode):
                for inst in node.insts:
                    if isinstance(inst, LabelInst):
                        used_labels.add(inst.name)
                    elif isinstance(inst, IRNode):
                        _find_labels(inst)

        _find_labels(ir)

        # Keep only labels that are actually used
        new_labels = {
            name: addr for name, addr in ir.labels.items() if name in used_labels
        }

        # Return a new RootNode with the optimized labels, preserving insts
        # Since we might not want to mutate `ir.labels` directly if treating nodes as mostly immutable.
        new_ir = RootNode(insts=ir.insts, labels=new_labels)
        return new_ir
