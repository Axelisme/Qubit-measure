from __future__ import annotations

from ..labels import iter_label_references
from ..node import IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import walk_instructions


class LabelDCEPass(AbsPipeLinePass):
    """
    Dead Code Elimination for labels.
    Removes labels that are not referenced by jump-like instructions.
    """

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        if not isinstance(ir, RootNode):
            return ir  # Pass if it's not a RootNode

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

        # Keep only labels that are actually used
        new_labels = {
            name: addr for name, addr in ir.labels.items() if name in used_labels
        }

        new_ir = RootNode(insts=ir.insts, labels=new_labels)
        return new_ir
