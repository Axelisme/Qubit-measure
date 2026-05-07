from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    NopInst,
    RegWriteInst,
    TestInst,
)
from zcu_tools.program.v2.ir.node import BasicBlockNode
from zcu_tools.program.v2.ir.passes.loop_merge import LoopConditionMergeLinear


def test_pattern_1_merge_zero_comparison():
    # Before:
    # REG_WR r1 op r1 - #1
    # JUMP label -if(NZ) -op(r1 - #0)
    block = BasicBlockNode(
        insts=[RegWriteInst(dst="r1", src="op", op="r1 - #1")],
        branch=JumpInst(label="loop", if_cond="NZ", op="r1 - #0"),
    )
    pass_ = LoopConditionMergeLinear()
    pass_.process_block(block)

    # After:
    # JUMP label -if(NZ) -wr(r1 op) -op(r1 - #1)
    assert len(block.insts) == 0
    assert block.branch.wr == "r1 op"
    assert block.branch.op == "r1 - #1"
    assert block.branch.if_cond == "NZ"


def test_pattern_1_merge_zero_comparison_fix_addr():
    block = BasicBlockNode(
        insts=[RegWriteInst(dst="r1", src="op", op="r1 - #1")],
        branch=JumpInst(label="loop", if_cond="NZ", op="r1 - #0"),
        fix_addr_size=True,
    )
    pass_ = LoopConditionMergeLinear()
    pass_.process_block(block)

    assert len(block.insts) == 1
    assert isinstance(block.insts[0], NopInst)
    assert block.branch.wr == "r1 op"


def test_pattern_2_merge_side_data_injection():
    # Before:
    # TEST op(r1 - #10)
    # REG_WR r2 op r2 + #1
    # JUMP label -if(S)
    block = BasicBlockNode(
        insts=[
            TestInst(op="r1 - #10"),
            RegWriteInst(dst="r2", src="op", op="r2 + #1"),
        ],
        branch=JumpInst(label="loop", if_cond="S"),
    )
    pass_ = LoopConditionMergeLinear()
    pass_.process_block(block)

    # After:
    # TEST op(r1 - #10)
    # JUMP label -if(S) -wr(r2 op) -op(r2 + #1)
    assert len(block.insts) == 1
    assert isinstance(block.insts[0], TestInst)
    assert block.branch.wr == "r2 op"
    assert block.branch.op == "r2 + #1"
    assert block.branch.if_cond == "S"


def test_pattern_2_merge_side_data_injection_fix_addr():
    block = BasicBlockNode(
        insts=[
            TestInst(op="r1 - #10"),
            RegWriteInst(dst="r2", src="op", op="r2 + #1"),
        ],
        branch=JumpInst(label="loop", if_cond="S"),
        fix_addr_size=True,
    )
    pass_ = LoopConditionMergeLinear()
    pass_.process_block(block)

    assert len(block.insts) == 2
    assert isinstance(block.insts[0], TestInst)
    assert isinstance(block.insts[1], NopInst)
    assert block.branch.wr == "r2 op"
