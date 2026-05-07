from __future__ import annotations

from ..instructions import JumpInst, NopInst, RegWriteInst, TestInst
from ..node import BasicBlockNode
from ..pipeline import AbsLinearPass


class LoopConditionMergeLinear(AbsLinearPass):
    """Linear pass to merge register increments and conditional jumps.

    Pattern 1: 1-Word Compression (Zero-based Comparison)
    - Before:
        REG_WR r1 op r1 - #1
        JUMP label -if(NZ) -op(r1 - #0)
    - After:
        NOP (if fix_addr_size else removed)
        JUMP label -if(NZ) -wr(r1 op) -op(r1 - #1)

    Pattern 2: Generic Side-Data Injection
    - Before:
        TEST op(...)
        REG_WR r1 op r1 + #1
        JUMP label -if(COND)  (no internal -op)
    - After:
        TEST op(...)
        NOP (if fix_addr_size else removed)
        JUMP label -if(COND) -wr(r1 op) -op(r1 + #1)
    """

    def process_block(self, block: BasicBlockNode) -> None:
        if block.branch is None or not block.insts:
            return

        # Pattern 1: Merge Dec+JumpZ
        self._merge_zero_comparison(block)

        # Pattern 2: Merge Inc into empty Jump
        self._merge_side_data_injection(block)

    def _merge_zero_comparison(self, block: BasicBlockNode) -> None:
        """Pattern 1: REG_WR r1 op r1-1 + JUMP op(r1-0) -> JUMP wr(r1) op(r1-1)."""
        if not block.insts:
            return

        last_idx = len(block.insts) - 1
        last_inst = block.insts[last_idx]
        branch = block.branch

        if (
            isinstance(last_inst, RegWriteInst)
            and last_inst.src == "op"
            and last_inst.op is not None
            and branch is not None
            and branch.op is not None
        ):
            # Target pattern: branch.op is "reg - #0" and last_inst.dst is "reg"
            reg = last_inst.dst
            if branch.op == f"{reg} - #0":
                # Merge: move last_inst's op into branch.op and use side-data wr
                new_branch = JumpInst(
                    label=branch.label,
                    if_cond=branch.if_cond,
                    addr=branch.addr,
                    wr=f"{reg} op",
                    op=last_inst.op,
                    uf=branch.uf,
                    extra_args=branch.extra_args,
                )
                block.branch = new_branch

                if block.fix_addr_size:
                    block.insts[last_idx] = NopInst()
                else:
                    block.insts.pop(last_idx)

    def _merge_side_data_injection(self, block: BasicBlockNode) -> None:
        """Pattern 2: TEST + REG_WR + JUMP -> TEST + JUMP wr."""
        if len(block.insts) < 2:
            return

        # Need branch to have NO op/wr/uf already
        branch = block.branch
        if branch is None or branch.op is not None or branch.wr is not None:
            return

        last_idx = len(block.insts) - 1
        last_inst = block.insts[last_idx]
        prev_inst = block.insts[last_idx - 1]

        # We look for a REG_WR preceded by something (usually a TEST, but could be anything
        # that doesn't use the ALU in a way that conflicts, though hardware flags are safe).
        # To be conservative and match the plan: TEST + REG_WR + JUMP.
        if isinstance(last_inst, RegWriteInst) and last_inst.src == "op" and isinstance(prev_inst, TestInst):
            reg = last_inst.dst
            new_branch = JumpInst(
                label=branch.label,
                if_cond=branch.if_cond,
                addr=branch.addr,
                wr=f"{reg} op",
                op=last_inst.op,
                uf=branch.uf,
                extra_args=branch.extra_args,
            )
            block.branch = new_branch

            if block.fix_addr_size:
                block.insts[last_idx] = NopInst()
            else:
                block.insts.pop(last_idx)
