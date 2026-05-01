from __future__ import annotations

from ..instructions import JumpInst, MetaInst, TestInst
from ..labels import iter_label_references
from ..node import BlockNode, IRBranch, IRBranchCase, IRLoop, IRNode, RootNode
from ..pipeline import AbsPipeLinePass, PipeLineContext
from ..traversal import walk_instructions, walk_nodes


class IRStructureValidationPass(AbsPipeLinePass):
    """Validate structural invariants expected by later IR passes."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        for node in walk_nodes(ir):
            if isinstance(node, IRLoop):
                self._validate_loop(node)
            elif isinstance(node, IRBranch):
                self._validate_branch(node)

            if isinstance(node, BlockNode):
                for item in node.insts:
                    if isinstance(item, MetaInst):
                        raise ValueError("MetaInst should not remain inside IR nodes")

        return ir

    def _validate_loop(self, loop: IRLoop) -> None:
        if loop.name == "":
            raise ValueError("IRLoop requires a non-empty name")
        if loop.counter_reg == "":
            raise ValueError(f"IRLoop '{loop.name}' requires a counter_reg")

        if not isinstance(loop.body, BlockNode):
            raise ValueError(f"IRLoop '{loop.name}' body must be a BlockNode")

    def _validate_branch(self, branch: IRBranch) -> None:
        if branch.name == "":
            raise ValueError("IRBranch requires a non-empty name")
        if len(branch.cases) == 0:
            raise ValueError(f"IRBranch '{branch.name}' requires at least one case")

        for case in branch.cases:
            if not isinstance(case, IRBranchCase):
                raise ValueError(
                    f"IRBranch '{branch.name}' case must be an IRBranchCase"
                )
            if case.name == "":
                raise ValueError(
                    f"IRBranch '{branch.name}' case requires a non-empty name"
                )
            # Logic: In structural IR, cases are logically part of the branch.
            # We don't need to check if they are in 'insts' because IRBranch 
            # no longer has an 'insts' list (except inside its dispatch block).


class LabelReferenceValidationPass(AbsPipeLinePass):
    """Ensure instruction label references point to known labels."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        if not isinstance(ir, RootNode):
            return ir

        # defined_labels from ir.labels
        defined_labels = set(ir.labels)
        
        # Add labels from structural node attributes
        for node in walk_nodes(ir):
            if isinstance(node, IRLoop):
                defined_labels.add(node.start_label or f"{node.name}_start")
                defined_labels.add(node.end_label or f"{node.name}_end")

        missing: set[str] = set()
        for inst in walk_instructions(ir):
            for label in iter_label_references(inst):
                if label not in defined_labels:
                    # Ignore pseudo-labels like HERE
                    if label != "HERE":
                        missing.add(label)

        if missing:
            labels = ", ".join(sorted(missing))
            raise ValueError(f"Undefined label reference(s): {labels}")

        return ir
