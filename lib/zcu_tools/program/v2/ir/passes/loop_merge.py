from __future__ import annotations

from ..instructions import JumpInst, RegWriteInst, TestInst
from ..node import BasicBlockNode
from ..operands import AluExpr, AluOp, Immediate, SideWrite, SrcKeyword
from ..pipeline import AbsChunkPass, ChunkList, PipeLineContext


def _is_pure_regwrite_op(inst: RegWriteInst) -> bool:
    """True only for a plain REG_WR dst op <expr> with no extra semantics."""
    return (
        inst.src == SrcKeyword.OP
        and inst.op is not None
        and inst.lit is None
        and inst.if_cond is None
        and not inst.uf
        and inst.wr is None
        and inst.label is None
        and inst.addr is None
    )


def _make_merged_branch(branch: JumpInst, inst: RegWriteInst) -> JumpInst:
    return JumpInst(
        label=branch.label,
        if_cond=branch.if_cond,
        addr=branch.addr,
        wr=SideWrite(inst.dst, "op"),
        op=inst.op,
        uf=branch.uf,
    )


class LoopConditionMergePass(AbsChunkPass):
    """Chunk pass to merge register increments and conditional jumps.

    Pattern 1: 1-Word Compression (Zero-based Comparison)
    - Before:
        REG_WR r1 op r1 - #1
        JUMP label -if(NZ) -op(r1 - #0)
    - After:
        JUMP label -if(NZ) -wr(r1 op) -op(r1 - #1)

    Pattern 2: Generic Side-Data Injection
    - Before:
        TEST op(...)
        REG_WR r1 op r1 + #1
        JUMP label -if(COND)  (no internal -op)
    - After:
        TEST op(...)
        JUMP label -if(COND) -wr(r1 op) -op(r1 + #1)
    """

    def process(
        self, chunks: ChunkList, ctx: PipeLineContext
    ) -> tuple[ChunkList, bool]:
        _ = ctx
        changed = False
        for chunk in chunks:
            if not isinstance(chunk, BasicBlockNode):
                continue
            changed |= self._process_block(chunk)
        return chunks, changed

    def _process_block(self, block: BasicBlockNode) -> bool:
        if block.fix_addr_size:
            return False
        if block.branch is None or not block.insts:
            return False

        # Pattern 1: Merge Dec+JumpZ
        changed = self._merge_zero_comparison(block)

        # Pattern 2: Merge Inc into empty Jump
        changed |= self._merge_side_data_injection(block)
        return changed

    def _merge_zero_comparison(self, block: BasicBlockNode) -> bool:
        """Pattern 1: REG_WR r1 op r1-1 + JUMP op(r1-0) -> JUMP wr(r1) op(r1-1)."""
        if not block.insts:
            return False

        last_idx = len(block.insts) - 1
        last_inst = block.insts[last_idx]
        branch = block.branch

        if (
            isinstance(last_inst, RegWriteInst)
            and _is_pure_regwrite_op(last_inst)
            and branch is not None
            and branch.if_cond is not None
            and branch.op is not None
            and branch.wr is None
        ):
            # Target pattern: branch.op is "reg - #0" and last_inst.dst is "reg"
            op = branch.op
            if (
                isinstance(op, AluExpr)
                and op.lhs == last_inst.dst
                and op.op == AluOp.SUB
                and op.rhs == Immediate(0)
            ):
                block.branch = _make_merged_branch(branch, last_inst)
                block.insts.pop(last_idx)
                return True
        return False

    def _merge_side_data_injection(self, block: BasicBlockNode) -> bool:
        """Pattern 2: TEST + REG_WR + JUMP -> TEST + JUMP wr."""
        if len(block.insts) < 2:
            return False

        # Need branch to have NO op/wr/uf already
        branch = block.branch
        if (
            branch is None
            or branch.if_cond is None
            or branch.op is not None
            or branch.wr is not None
            or branch.uf
        ):
            return False

        last_idx = len(block.insts) - 1
        last_inst = block.insts[last_idx]
        prev_inst = block.insts[last_idx - 1]

        # We look for a REG_WR preceded by something (usually a TEST, but could be anything
        # that doesn't use the ALU in a way that conflicts, though hardware flags are safe).
        # To be conservative and match the plan: TEST + REG_WR + JUMP.
        if (
            isinstance(last_inst, RegWriteInst)
            and _is_pure_regwrite_op(last_inst)
            and isinstance(prev_inst, TestInst)
        ):
            block.branch = _make_merged_branch(branch, last_inst)
            block.insts.pop(last_idx)
            return True
        return False
