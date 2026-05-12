from zcu_tools.program.v2.ir.instructions import JumpInst, RegWriteInst, TestInst
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import BasicBlockNode
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    Register,
    SideWrite,
    SrcKeyword,
)
from zcu_tools.program.v2.ir.passes.loop import LoopConditionMergePass
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext


def _run_pass(block: BasicBlockNode) -> None:
    LoopConditionMergePass().process(
        [block], PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    )


def test_pattern_1_merge_zero_comparison():
    block = BasicBlockNode(
        insts=[
            RegWriteInst(
                dst=Register("r1"),
                src=SrcKeyword.OP,
                op=AluExpr(Register("r1"), AluOp.SUB, Immediate(1)),
            )
        ],
        branch=JumpInst(
            label=Label("loop"),
            if_cond="NZ",
            op=AluExpr(Register("r1"), AluOp.SUB, Immediate(0)),        ),
    )
    _run_pass(block)

    assert len(block.insts) == 0
    assert block.branch is not None
    assert block.branch.wr == SideWrite(Register("r1"), "op")
    assert block.branch.op.op == AluOp.SUB
    assert str(block.branch.op.rhs) == "#1"
    assert block.branch.if_cond == "NZ"


def test_pattern_1_merge_zero_comparison_fix_addr_is_skipped():
    block = BasicBlockNode(
        insts=[
            RegWriteInst(
                dst=Register("r1"),
                src=SrcKeyword.OP,
                op=AluExpr(Register("r1"), AluOp.SUB, Immediate(1)),
            )
        ],
        branch=JumpInst(
            label=Label("loop"),
            if_cond="NZ",
            op=AluExpr(Register("r1"), AluOp.SUB, Immediate(0)),        ),
        fix_addr_size=True,
    )
    _run_pass(block)

    assert len(block.insts) == 1
    assert isinstance(block.insts[0], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr is None


def test_pattern_2_merge_side_data_injection():
    block = BasicBlockNode(
        insts=[
            TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10))),
            RegWriteInst(
                dst=Register("r2"),
                src=SrcKeyword.OP,
                op=AluExpr(Register("r2"), AluOp.ADD, Immediate(1)),
            ),
        ],
        branch=JumpInst(label=Label("loop"), if_cond="S"),
    )
    _run_pass(block)

    assert len(block.insts) == 1
    assert isinstance(block.insts[0], TestInst)
    assert block.branch is not None
    assert block.branch.wr == SideWrite(Register("r2"), "op")
    assert str(block.branch.op.rhs) == "#1"
    assert block.branch.if_cond == "S"


def test_pattern_2_merge_side_data_injection_fix_addr_is_skipped():
    block = BasicBlockNode(
        insts=[
            TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10))),
            RegWriteInst(
                dst=Register("r2"),
                src=SrcKeyword.OP,
                op=AluExpr(Register("r2"), AluOp.ADD, Immediate(1)),
            ),
        ],
        branch=JumpInst(label=Label("loop"), if_cond="S"),
        fix_addr_size=True,
    )
    _run_pass(block)

    assert len(block.insts) == 2
    assert isinstance(block.insts[0], TestInst)
    assert isinstance(block.insts[1], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr is None


def test_pattern_1_rejects_regwrite_with_uf():
    block = BasicBlockNode(
        insts=[
            RegWriteInst(
                dst=Register("r1"),
                src=SrcKeyword.OP,
                op=AluExpr(Register("r1"), AluOp.SUB, Immediate(1)),
                uf="1",
            )
        ],
        branch=JumpInst(
            label=Label("loop"),
            if_cond="NZ",
            op=AluExpr(Register("r1"), AluOp.SUB, Immediate(0)),        ),
    )

    _run_pass(block)

    assert len(block.insts) == 1
    assert isinstance(block.insts[0], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr is None
    assert str(block.branch.op.rhs) == "#0"


def test_pattern_1_rejects_branch_with_existing_wr():
    block = BasicBlockNode(
        insts=[
            RegWriteInst(
                dst=Register("r1"),
                src=SrcKeyword.OP,
                op=AluExpr(Register("r1"), AluOp.SUB, Immediate(1)),
            )
        ],
        branch=JumpInst(
            label=Label("loop"),
            if_cond="NZ",
            op=AluExpr(Register("r1"), AluOp.SUB, Immediate(0)),            wr=SideWrite(Register("r7"), "op"),
        ),
    )

    _run_pass(block)

    assert len(block.insts) == 1
    assert isinstance(block.insts[0], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr == SideWrite(Register("r7"), "op")
    assert str(block.branch.op.rhs) == "#0"


def test_pattern_2_rejects_conditional_regwrite():
    block = BasicBlockNode(
        insts=[
            TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10))),
            RegWriteInst(
                dst=Register("r2"),
                src=SrcKeyword.OP,
                op=AluExpr(Register("r2"), AluOp.ADD, Immediate(1)),
                if_cond="NZ",
            ),
        ],
        branch=JumpInst(label=Label("loop"), if_cond="S"),
    )

    _run_pass(block)

    assert len(block.insts) == 2
    assert isinstance(block.insts[0], TestInst)
    assert isinstance(block.insts[1], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr is None
    assert block.branch.op is None


def test_pattern_2_rejects_branch_with_uf():
    block = BasicBlockNode(
        insts=[
            TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10))),
            RegWriteInst(
                dst=Register("r2"),
                src=SrcKeyword.OP,
                op=AluExpr(Register("r2"), AluOp.ADD, Immediate(1)),
            ),
        ],
        branch=JumpInst(label=Label("loop"), if_cond="S", uf="1"),
    )

    _run_pass(block)

    assert len(block.insts) == 2
    assert isinstance(block.insts[0], TestInst)
    assert isinstance(block.insts[1], RegWriteInst)
    assert block.branch is not None
    assert block.branch.wr is None
    assert block.branch.op is None
