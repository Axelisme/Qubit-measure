from __future__ import annotations

from ..labels import iter_label_references
from ..node import InstNode, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import walk_instructions


class LabelDCEPass(AbsPipeLinePass):
    """
    Dead Code Elimination for labels.
    Removes labels that are not referenced by jump-like instructions.
    """

    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode:
        from ..instructions import LabelInst
        from ..node import IRLoop
        from ..traversal import walk_nodes

        used_labels: set[str] = set()
        for inst in walk_instructions(ir):
            used_labels.update(iter_label_references(inst))

        # Also retain labels implicitly used by structural nodes like IRLoop
        for node in walk_nodes(ir):
            if isinstance(node, IRLoop):
                used_labels.add(node.start_label or f"{node.name}_start")
                used_labels.add(node.end_label or f"{node.name}_end")

        # Keep only labels that are actually used (we now check all LabelInsts)
        new_insts = []
        for node in ir.insts:
            if isinstance(node, InstNode) and isinstance(node.inst, LabelInst):
                if node.inst.name not in used_labels:
                    continue
            new_insts.append(node)

        new_ir = RootNode(insts=new_insts)
        return new_ir
