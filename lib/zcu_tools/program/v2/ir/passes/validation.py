from __future__ import annotations

from ..instructions import MetaInst
from ..labels import iter_label_references
from ..node import BlockNode, IRBranch, IRLoop, IRNode, RootNode
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

        sections = (
            ("initial", loop.initial),
            ("stop_check", loop.stop_check),
            ("body", loop.body),
            ("update", loop.update),
        )
        for section_name, section in sections:
            if not isinstance(section, BlockNode):
                raise ValueError(f"IRLoop.{section_name} must be a BlockNode")

    def _validate_branch(self, branch: IRBranch) -> None:
        if branch.name == "":
            raise ValueError("IRBranch requires a non-empty name")
        if len(branch.cases) == 0:
            raise ValueError(f"IRBranch '{branch.name}' requires at least one case")

        for case in branch.cases:
            if not isinstance(case, BlockNode):
                raise ValueError(f"IRBranch '{branch.name}' case must be a BlockNode")
            if case not in branch.insts:
                raise ValueError(
                    f"IRBranch '{branch.name}' case is not present in branch body"
                )


class LabelReferenceValidationPass(AbsPipeLinePass):
    """Ensure instruction label references point to known labels."""

    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode:
        if not isinstance(ir, RootNode):
            return ir

        defined_labels = set(ir.labels)
        missing: set[str] = set()
        for inst in walk_instructions(ir):
            for label in iter_label_references(inst):
                if label not in defined_labels:
                    missing.add(label)

        if missing:
            labels = ", ".join(sorted(missing))
            raise ValueError(f"Undefined label reference(s): {labels}")

        return ir
