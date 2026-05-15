from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TimeInst,
    WmemWriteInst,
)
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    SrcKeyword,
    TimeOffset,
)
from zcu_tools.program.v2.ir.passes.dataflow import IncRegMergePass
from zcu_tools.program.v2.ir.passes.dataflow.inc_reg_merge import INC_REG_IMM_MAX
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext


def _run_chunk_pass(root: BlockNode) -> BlockNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    chunks, _ = IncRegMergePass().process(
        chunks, PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    )
    return parser.parse(chunks)


def test_inc_reg_merge_free_basic():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(2)),
                    ),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(3)),
                    ),
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r2"), AluOp.ADD, Immediate(5)),
                    ),
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r2"), AluOp.SUB, Immediate(1)),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    block = root.insts[0]
    assert isinstance(block, BasicBlockNode)
    insts = block.insts

    # NopInst is a barrier: r1+=2 is flushed before it, r1+=3 and r2+=4 after.
    assert len(insts) == 4
    assert isinstance(insts[0], RegWriteInst) and insts[0].dst.name == "r1"
    assert insts[0].op is not None and str(insts[0].op.rhs) == "#2"
    assert isinstance(insts[1], NopInst)
    assert isinstance(insts[2], RegWriteInst) and insts[2].dst.name == "r1"
    assert insts[2].op is not None and str(insts[2].op.rhs) == "#3"
    assert isinstance(insts[3], RegWriteInst) and insts[3].dst.name == "r2"
    assert insts[3].op is not None and str(insts[3].op.rhs) == "#4"


def test_inc_reg_merge_free_flush_on_read():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(2)),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r1")),
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(3)),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    _bb = root.insts[0]

    assert isinstance(_bb, BasicBlockNode)

    insts = _bb.insts

    assert len(insts) == 3
    assert isinstance(insts[0], RegWriteInst) and insts[0].op is not None
    assert str(insts[0].op.rhs) == "#2"
    assert isinstance(insts[1], TimeInst)
    assert isinstance(insts[2], RegWriteInst) and insts[2].op is not None
    assert str(insts[2].op.rhs) == "#3"


def test_inc_reg_merge_free_can_cross_port_write():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r0"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r0"), AluOp.ADD, Immediate(1)),
                    ),
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(1),
                        time=TimeOffset(0),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r0"), AluOp.ADD, Immediate(1)),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    _bb = root.insts[0]

    assert isinstance(_bb, BasicBlockNode)

    insts = _bb.insts

    # r0+=1 sinks past PORT and register-driven TIME (both transparent), merges at end.
    assert len(insts) == 3
    assert isinstance(insts[0], PortWriteInst)
    assert isinstance(insts[1], TimeInst)
    assert isinstance(insts[2], RegWriteInst) and insts[2].dst.name == "r0"
    assert insts[2].op is not None and str(insts[2].op.rhs) == "#2"


def test_inc_reg_merge_free_cpmg_like_unrolled_body():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(1),
                        time=TimeOffset(0),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r0"), AluOp.ADD, Immediate(1)),
                    ),
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(1),
                        time=TimeOffset(0),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r0"), AluOp.ADD, Immediate(1)),
                    ),
                    PortWriteInst(
                        dst=ImmValue(2),
                        src=SrcKeyword.WMEM,
                        addr=MemAddr(1),
                        time=TimeOffset(0),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r0"), AluOp.ADD, Immediate(1)),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    _bb = root.insts[0]

    assert isinstance(_bb, BasicBlockNode)

    insts = _bb.insts

    # Register-driven TIME is transparent: all 3 r0+=1 sink past PORT and TIME,
    # merge at end. Result: PORT, TIME, PORT, TIME, PORT, TIME, r0+=3
    assert len(insts) == 7
    for i in range(3):
        assert isinstance(insts[i * 2], PortWriteInst)
        assert isinstance(insts[i * 2 + 1], TimeInst)
    assert isinstance(insts[6], RegWriteInst) and insts[6].dst.name == "r0"
    assert isinstance(insts[6], RegWriteInst) and insts[6].op is not None
    assert str(insts[6].op.rhs) == "#3"


def test_inc_reg_merge_fixed_basic_is_skipped():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(2)),
                    ),
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(3)),
                    ),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r2"), AluOp.ADD, Immediate(5)),
                    ),
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r2"), AluOp.SUB, Immediate(5)),
                    ),
                ],
                disable_opt=True,
            )
        ]
    )

    root = _run_chunk_pass(root)
    block = root.insts[0]
    assert isinstance(block, BasicBlockNode)
    insts = block.insts

    assert len(insts) == 5
    assert isinstance(insts[0], RegWriteInst)
    assert insts[0].op is not None and str(insts[0].op.rhs) == "#2"
    assert isinstance(insts[1], RegWriteInst)
    assert isinstance(insts[2], NopInst)
    assert isinstance(insts[3], RegWriteInst)
    assert isinstance(insts[4], RegWriteInst)


def test_inc_reg_merge_fixed_barrier():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(2)),
                    ),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(3)),
                    ),
                ],
                disable_opt=True,
            )
        ]
    )

    root = _run_chunk_pass(root)
    _bb = root.insts[0]

    assert isinstance(_bb, BasicBlockNode)

    insts = _bb.insts
    # In fixed blocks, non-adjacent increments are not merged
    assert len(insts) == 3
    assert isinstance(insts[0], RegWriteInst) and insts[0].op is not None
    assert str(insts[0].op.rhs) == "#2"
    assert isinstance(insts[2], RegWriteInst) and insts[2].op is not None
    assert str(insts[2].op.rhs) == "#3"


def test_inc_reg_merge_does_not_cross_wmem_write_via_alias():
    # Regression: REG_WR w_freq op (w_freq + #5) ; WMEM_WR ; REG_WR w_freq op
    # (w_freq + #3).  WMEM_WR reads {w0..w5, r_wave, s14}; w_freq is an alias
    # of w0.  Without alias-aware read tracking, the pending +5 increment on
    # w_freq would be sunk past WMEM_WR and merged with +3, so the WMEM_WR
    # would observe the un-incremented value.
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("w_freq"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("w_freq"), AluOp.ADD, Immediate(5)),
                    ),
                    WmemWriteInst(addr=MemAddr(0)),
                    RegWriteInst(
                        dst=Register("w_freq"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("w_freq"), AluOp.ADD, Immediate(3)),
                    ),
                ]
            )
        ]
    )

    out = _run_chunk_pass(root)
    _bb = out.insts[0]

    assert isinstance(_bb, BasicBlockNode)

    insts = _bb.insts
    # The first +5 must remain before WMEM_WR; only the +3 may sit afterward.
    assert isinstance(insts[0], RegWriteInst)
    assert insts[0].op is not None and str(insts[0].op.rhs) == "#5"
    assert isinstance(insts[1], WmemWriteInst)
    # Whether the trailing +3 stays adjacent or merges into a #3 inc is an
    # implementation detail; what matters is that #5 never crosses WMEM_WR.
    assert any(
        isinstance(inst, RegWriteInst)
        and inst.op is not None
        and str(inst.op.rhs) == "#3"
        for inst in insts[2:]
    )


def test_inc_reg_merge_overflow_flushes_before_accumulating():
    # Two increments whose sum exceeds INC_REG_IMM_MAX must NOT be merged.
    # The first must be flushed, the second starts a new accumulation.
    big = INC_REG_IMM_MAX  # exactly at the limit — still OK to merge with 0
    step = 1  # big + step = INC_REG_IMM_MAX + 1 → overflow

    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(big)),
                    ),
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(step)),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    block = root.insts[0]
    assert isinstance(block, BasicBlockNode)
    insts = block.insts

    # Must be two separate increments; merged value would overflow.
    assert len(insts) == 2
    assert isinstance(insts[0], RegWriteInst) and insts[0].op is not None
    assert insts[0].op.rhs == Immediate(big)
    assert isinstance(insts[1], RegWriteInst) and insts[1].op is not None
    assert insts[1].op.rhs == Immediate(step)


def test_inc_reg_merge_single_oversized_step_emitted_as_is():
    # A single increment already exceeding INC_REG_IMM_MAX must pass through
    # unchanged (no crash, no silent truncation).
    huge = INC_REG_IMM_MAX + 1

    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(huge)),
                    ),
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r1"), AluOp.ADD, Immediate(1)),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    block = root.insts[0]
    assert isinstance(block, BasicBlockNode)
    insts = block.insts

    # The oversized step must be emitted as-is; the trailing +1 may merge or not.
    assert len(insts) >= 2
    assert isinstance(insts[0], RegWriteInst) and insts[0].op is not None
    assert insts[0].op.rhs == Immediate(huge)
