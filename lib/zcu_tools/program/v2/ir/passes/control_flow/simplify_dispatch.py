from __future__ import annotations

from ...dispatch import needs_big_jump
from ...instructions import BaseInst, JumpInst, RegWriteInst
from ...labels import LabelRef
from ...node import BasicBlockNode, IRDispatch, IRNode
from ...operands import AluExpr, AluOp, Immediate, Register, SrcKeyword
from ...pipeline import AbsIRTreePass, PipeLineContext


class SimplifyDispatchPass(AbsIRTreePass):
    """IR-layer pass: replace a 2-target IRDispatch with a single conditional jump.

    When an IRDispatch has exactly 2 target labels, the dispatch table island
    is unnecessary.  This pass emits a single conditional jump:

        value_reg == 0  (Z set)  -> fall through to target_labels[0]
        value_reg != 0  (NZ set) -> jump to target_labels[1]

    big-PMEM:
        REG_WR s15 label target_labels[1]
        JUMP [s15] -if(NZ) -op(value_reg - #0)

    small-PMEM:
        JUMP target_labels[1] -if(NZ) -op(value_reg - #0)

    The body copies that follow as siblings in the parent BlockNode are
    unaffected; the pipeline naturally falls through to target_labels[0].
    """

    def transform(
        self,
        node: IRNode,
        ctx: PipeLineContext,
    ) -> IRNode:
        if not isinstance(node, IRDispatch) or len(node.target_labels) != 2:
            return node

        pmem_size = ctx.config.pmem_capacity
        target1 = node.target_labels[1]
        op = AluExpr(node.value_reg, AluOp.SUB, Immediate(0))

        if needs_big_jump(pmem_size):
            return BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("s15"),
                        src=SrcKeyword.LABEL,
                        label=LabelRef(target1),
                    )
                ],
                branch=JumpInst(
                    addr=Register("s15"),
                    if_cond="NZ",
                    op=op,
                ),
            )
        else:
            return BasicBlockNode(
                branch=JumpInst(
                    label=LabelRef(target1),
                    if_cond="NZ",
                    op=op,
                )
            )
