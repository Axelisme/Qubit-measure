from __future__ import annotations

from typing_extensions import Optional

from ...hw_semantics import needs_big_jump
from ...instructions import JumpInst, RegWriteInst
from ...labels import LabelRef
from ...node import BasicBlockNode, BlockNode, IRDispatch, IRNode
from ...operands import AluExpr, AluOp, Immediate, Register, SrcKeyword
from ...pipeline import AbsIRTreePass, PipeLineContext


class SimplifyDispatchPass(AbsIRTreePass):
    """IR-layer pass: replace a 2-target IRDispatch with two explicit jumps.

    When an IRDispatch has exactly 2 target labels, the dispatch table island
    is unnecessary.  This pass emits a BlockNode with two BasicBlockNodes:

        block 0 (conditional): NZ → jump to target_labels[1]
        block 1 (fallthrough): unconditional jump to target_labels[0]

    BranchEliminationPass will remove the unconditional jump in block 1 when
    target_labels[0] immediately follows (small-PMEM, direct jump).  For
    big-PMEM both blocks always use the indirect REG_WR+JUMP idiom.

    small-PMEM layout::

        JUMP target_labels[1] -if(NZ) -op(value_reg - #0)
        JUMP target_labels[0]

    big-PMEM layout::

        REG_WR s15 label target_labels[1]
        JUMP [s15] -if(NZ) -op(value_reg - #0)
        REG_WR s15 label target_labels[0]
        JUMP [s15]
    """

    def transform(
        self,
        node: IRNode,
        ctx: PipeLineContext,
    ) -> Optional[IRNode]:
        if not isinstance(node, IRDispatch) or len(node.target_labels) != 2:
            return None

        pmem_size = ctx.config.pmem_capacity
        target0 = node.target_labels[0]
        target1 = node.target_labels[1]
        op = AluExpr(node.value_reg, AluOp.SUB, Immediate(0))

        if needs_big_jump(pmem_size):
            cond_bb = BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("s15"),
                        src=SrcKeyword.LABEL,
                        label=LabelRef(target1),
                    )
                ],
                branch=JumpInst(addr=Register("s15"), if_cond="NZ", op=op, uf=True),
            )
            fallthrough_bb = BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("s15"),
                        src=SrcKeyword.LABEL,
                        label=LabelRef(target0),
                    )
                ],
                branch=JumpInst(addr=Register("s15")),
            )
        else:
            cond_bb = BasicBlockNode(
                branch=JumpInst(label=LabelRef(target1), if_cond="NZ", op=op, uf=True)
            )
            fallthrough_bb = BasicBlockNode(branch=JumpInst(label=LabelRef(target0)))

        return BlockNode(insts=[cond_bb, fallthrough_bb])
