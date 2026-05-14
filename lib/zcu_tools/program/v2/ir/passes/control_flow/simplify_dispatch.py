from __future__ import annotations

from ...instructions import JumpInst, LabelInst
from ...labels import Label
from ...node import BasicBlockNode, BlockNode, IRBranch, IRNode
from ...operands import AluExpr, AluOp, Immediate
from ...pipeline import AbsIRPass, PipeLineContext
from ..base import IRTransformer


class SimplifyDispatchPass(IRTransformer, AbsIRPass):
    """IR-layer pass: replace 2-case IRBranch with a single conditional jump.

    When an IRBranch has exactly 2 cases, the dispatch table is unnecessary.
    This pass rewrites:

        IRBranch(compare_reg, cases=[case0, case1])

    into a BlockNode containing:

        BasicBlockNode(branch=JUMP else_label -if(NZ) -op(compare_reg - #0))
        case0           (IRNode, inlined as-is)
        BasicBlockNode(branch=JUMP end_label)   <- BranchEliminationPass may remove
        BasicBlockNode(labels=[else_label], ...) <- first block of case1 gets label
        case1           (IRNode, inlined as-is)
        BasicBlockNode(labels=[end_label])

    compare_reg == 0  -> Z flag set, NZ not taken -> fallthrough to case 0
    compare_reg != 0  -> NZ taken -> jump to case 1 (else)

    The produced BlockNode is NOT marked disable_opt so that BranchEliminationPass
    can later remove the redundant end-jump when end_label immediately follows.
    """

    def process(self, ir: BlockNode, ctx: PipeLineContext) -> tuple[BlockNode, bool]:  # noqa: ARG002
        self._changed = False
        result = self.visit(ir)
        return result, self._changed  # type: ignore[return-value]

    def visit_IRBranch(self, node: IRBranch) -> IRNode:
        if len(node.cases) != 2:
            return node

        else_label = Label.make_new(f"{node.name}_simp_else")
        end_label = Label.make_new(f"{node.name}_simp_end")

        guard = BasicBlockNode(
            branch=JumpInst(
                label=else_label,
                if_cond="NZ",
                op=AluExpr(node.compare_reg, AluOp.SUB, Immediate(0)),
            )
        )
        end_jump = BasicBlockNode(branch=JumpInst(label=end_label))
        else_entry = BasicBlockNode(
            labels=[LabelInst(name=else_label, can_remove=True)]
        )
        end_pad = BasicBlockNode(labels=[LabelInst(name=end_label, can_remove=True)])

        self._changed = True
        return BlockNode(
            insts=[guard, node.cases[0], end_jump, else_entry, node.cases[1], end_pad]
        )
