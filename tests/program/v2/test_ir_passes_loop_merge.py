from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    NopInst,
    RegWriteInst,
    TestInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import BasicBlockNode
from zcu_tools.program.v2.ir.operands import AluExpr, Literal, Register, SideWrite
from zcu_tools.program.v2.ir.passes.loop_merge import LoopConditionMergeLinear


def test_pattern_1_merge_zero_comparison():
    block = BasicBlockNode(
        insts=[RegWriteInst(dst=Register("r1"), src="op", op=AluExpr(Register("r1"), "-", Literal("#1")))],
        branch=JumpInst(label=Label("loop"), if_cond="NZ", op=AluExpr(Register("r1"), "-", Literal("#0"))),
    )
    pass_ = LoopConditionMergeLinear()
    pass_.process_block(block)

    assert len(block.insts) == 0
    assert block.branch is not None
    assert block.branch.wr == SideWrite(Register("r1"), "op")
    assert block.branch.op.op == "-"
    assert block.branch.op.rhs.value == "#1"
    assert block.branch.if_cond == "NZ"

def test_pattern_1_merge_zero_comparison_fix_addr():
    block = BasicBlockNode(
        insts=[RegWriteInst(dst=Register("r1"), src="op", op=AluExpr(Register("r1"), "-", Literal("#1")))],
        branch=JumpInst(label=Label("loop"), if_cond="NZ", op=AluExpr(Register("r1"), "-", Literal("#0"))),
        fix_addr_size=True,
    )
    pass_ = LoopConditionMergeLinear()
    pass_.process_block(block)

    assert len(block.insts) == 1
    assert isinstance(block.insts[0], NopInst)
    assert block.branch is not None
    assert block.branch.wr == SideWrite(Register("r1"), "op")

def test_pattern_2_merge_side_data_injection():
    block = BasicBlockNode(
        insts=[
            TestInst(op=AluExpr(Register("r1"), "-", Literal("#10"))),
            RegWriteInst(dst=Register("r2"), src="op", op=AluExpr(Register("r2"), "+", Literal("#1"))),
        ],
        branch=JumpInst(label=Label("loop"), if_cond="S"),
    )
    pass_ = LoopConditionMergeLinear()
    pass_.process_block(block)

    assert len(block.insts) == 1
    assert isinstance(block.insts[0], TestInst)
    assert block.branch is not None
    assert block.branch.wr == SideWrite(Register("r2"), "op")
    assert block.branch.op.rhs.value == "#1"
    assert block.branch.if_cond == "S"

def test_pattern_2_merge_side_data_injection_fix_addr():
    block = BasicBlockNode(
        insts=[
            TestInst(op=AluExpr(Register("r1"), "-", Literal("#10"))),
            RegWriteInst(dst=Register("r2"), src="op", op=AluExpr(Register("r2"), "+", Literal("#1"))),
        ],
        branch=JumpInst(label=Label("loop"), if_cond="S"),
        fix_addr_size=True,
    )
    pass_ = LoopConditionMergeLinear()
    pass_.process_block(block)

    assert len(block.insts) == 2
    assert isinstance(block.insts[0], TestInst)
    assert isinstance(block.insts[1], NopInst)
    assert block.branch is not None
    assert block.branch.wr == SideWrite(Register("r2"), "op")

def test_pattern_1_rejects_regwrite_with_uf():
    block = BasicBlockNode(
        insts=[RegWriteInst(dst=Register("r1"), src="op", op=AluExpr(Register("r1"), "-", Literal("#1")), uf="1")],
        branch=JumpInst(label=Label("loop"), if_cond="NZ", op=AluExpr(Register("r1"), "-", Literal("#0"))),
    )

    LoopConditionMergeLinear().process_block(block)

    assert len(block.insts) == 1
    assert isinstance(block.insts[0], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr is None
    assert block.branch.op.rhs.value == "#0"

def test_pattern_1_rejects_branch_with_existing_wr():
    block = BasicBlockNode(
        insts=[RegWriteInst(dst=Register("r1"), src="op", op=AluExpr(Register("r1"), "-", Literal("#1")))],
        branch=JumpInst(label=Label("loop"), if_cond="NZ", op=AluExpr(Register("r1"), "-", Literal("#0")), wr=SideWrite(Register("r7"), "op")),
    )

    LoopConditionMergeLinear().process_block(block)

    assert len(block.insts) == 1
    assert isinstance(block.insts[0], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr == SideWrite(Register("r7"), "op")
    assert block.branch.op.rhs.value == "#0"

def test_pattern_2_rejects_conditional_regwrite():
    block = BasicBlockNode(
        insts=[
            TestInst(op=AluExpr(Register("r1"), "-", Literal("#10"))),
            RegWriteInst(dst=Register("r2"), src="op", op=AluExpr(Register("r2"), "+", Literal("#1")), if_cond="NZ"),
        ],
        branch=JumpInst(label=Label("loop"), if_cond="S"),
    )

    LoopConditionMergeLinear().process_block(block)

    assert len(block.insts) == 2
    assert isinstance(block.insts[0], TestInst)
    assert isinstance(block.insts[1], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr is None
    assert block.branch.op is None

def test_pattern_2_rejects_branch_with_uf():
    block = BasicBlockNode(
        insts=[
            TestInst(op=AluExpr(Register("r1"), "-", Literal("#10"))),
            RegWriteInst(dst=Register("r2"), src="op", op=AluExpr(Register("r2"), "+", Literal("#1"))),
        ],
        branch=JumpInst(label=Label("loop"), if_cond="S", uf="1"),
    )

    LoopConditionMergeLinear().process_block(block)

    assert len(block.insts) == 2
    assert isinstance(block.insts[0], TestInst)
    assert isinstance(block.insts[1], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr is None
    assert block.branch.op is None
