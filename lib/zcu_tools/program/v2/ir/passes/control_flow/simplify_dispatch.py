from __future__ import annotations

from ...instructions import JumpInst, LabelInst
from ...labels import Label
from ...node import BasicBlockNode, IRBranch, IRNode
from ...operands import AluExpr, AluOp, Immediate
from ...pipeline import AbsIRTreePass, PipeLineContext


class SimplifyDispatchPass(AbsIRTreePass):
    """IR-layer pass: replace 2-case IRBranch with a single conditional jump.

    When an IRBranch has exactly 2 cases, the dispatch table is unnecessary.
    This pass receives the already-lowered child chunks and assembles:

        guard  = BasicBlockNode(branch=JUMP else_label -if(NZ) -op(compare_reg - #0))
        case0_chunks  (already ChunkPass-optimized)
        end_jump = BasicBlockNode(branch=JUMP end_label)
        else_entry = BasicBlockNode(labels=[else_label])
        case1_chunks  (already ChunkPass-optimized)
        end_pad = BasicBlockNode(labels=[end_label])

    compare_reg == 0  -> Z flag set, NZ not taken -> fallthrough to case 0
    compare_reg != 0  -> NZ taken -> jump to case 1 (else)

    BranchEliminationPass will later remove end_jump when end_label immediately
    follows, and BlockMergePass will collapse adjacent label-only blocks.
    """

    def transform(
        self,
        node: IRNode,
        child_chunks: list[list[BasicBlockNode]],
        ctx: PipeLineContext,  # noqa: ARG002
    ) -> IRNode | list[BasicBlockNode]:
        if not isinstance(node, IRBranch) or len(node.cases) != 2:
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

        return (
            [guard]
            + child_chunks[0]
            + [end_jump, else_entry]
            + child_chunks[1]
            + [end_pad]
        )
