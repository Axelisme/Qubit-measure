from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TimeInst,
    WmemWriteInst,
)
from zcu_tools.program.v2.ir.node import BasicBlockNode, RootNode
from zcu_tools.program.v2.ir.operands import AluExpr, ImmValue, Register, AluOp
from zcu_tools.program.v2.ir.passes.dataflow import IncRegMergePass
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext


def _run_chunk_pass(root: RootNode) -> RootNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    chunks, _ = IncRegMergePass().process(
        chunks, PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    )
    return parser.parse(chunks)


def test_inc_reg_merge_free_basic():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), AluOp.ADD, ImmValue(2, prefix="#")),
                    ),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), AluOp.ADD, ImmValue(3, prefix="#")),
                    ),
                    RegWriteInst(
                        dst=Register("r2"),
                        src="op",
                        op=AluExpr(Register("r2"), AluOp.ADD, ImmValue(5, prefix="#")),
                    ),
                    RegWriteInst(
                        dst=Register("r2"),
                        src="op",
                        op=AluExpr(Register("r2"), AluOp.SUB, ImmValue(1, prefix="#")),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    block = root.insts[0]
    insts = block.insts

    assert len(insts) == 3
    assert isinstance(insts[0], NopInst)
    assert str(insts[1].op.rhs) == "#5"  # r1 + #5
    assert insts[1].dst.name == "r1"
    assert str(insts[2].op.rhs) == "#4"  # r2 + #4
    assert insts[2].dst.name == "r2"


def test_inc_reg_merge_free_flush_on_read():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), AluOp.ADD, ImmValue(2, prefix="#")),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r1")),
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), AluOp.ADD, ImmValue(3, prefix="#")),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    insts = root.insts[0].insts

    assert len(insts) == 3
    assert str(insts[0].op.rhs) == "#2"
    assert isinstance(insts[1], TimeInst)
    assert str(insts[2].op.rhs) == "#3"


def test_inc_reg_merge_free_can_cross_port_write():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), AluOp.ADD, ImmValue(1, prefix="#")),
                    ),
                    PortWriteInst(
                        dst=ImmValue(2, prefix=""),
                        src="wmem",
                        addr=ImmValue(1, prefix="&"),
                        time=ImmValue(0, prefix="@"),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), AluOp.ADD, ImmValue(1, prefix="#")),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    insts = root.insts[0].insts

    assert len(insts) == 3
    assert isinstance(insts[0], PortWriteInst)
    assert isinstance(insts[1], TimeInst)
    assert isinstance(insts[2], RegWriteInst)
    assert str(insts[2].op.rhs) == "#2"
    assert insts[2].dst.name == "r0"


def test_inc_reg_merge_free_cpmg_like_unrolled_body():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    PortWriteInst(
                        dst=ImmValue(2, prefix=""),
                        src="wmem",
                        addr=ImmValue(1, prefix="&"),
                        time=ImmValue(0, prefix="@"),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), AluOp.ADD, ImmValue(1, prefix="#")),
                    ),
                    PortWriteInst(
                        dst=ImmValue(2, prefix=""),
                        src="wmem",
                        addr=ImmValue(1, prefix="&"),
                        time=ImmValue(0, prefix="@"),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), AluOp.ADD, ImmValue(1, prefix="#")),
                    ),
                    PortWriteInst(
                        dst=ImmValue(2, prefix=""),
                        src="wmem",
                        addr=ImmValue(1, prefix="&"),
                        time=ImmValue(0, prefix="@"),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), AluOp.ADD, ImmValue(1, prefix="#")),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    insts = root.insts[0].insts

    # 3 port writes + 3 time insts + 1 final merged reg write = 7
    assert len(insts) == 7
    assert isinstance(insts[-1], RegWriteInst)
    assert str(insts[-1].op.rhs) == "#3"


def test_inc_reg_merge_fixed_basic_is_skipped():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), AluOp.ADD, ImmValue(2, prefix="#")),
                    ),
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), AluOp.ADD, ImmValue(3, prefix="#")),
                    ),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r2"),
                        src="op",
                        op=AluExpr(Register("r2"), AluOp.ADD, ImmValue(5, prefix="#")),
                    ),
                    RegWriteInst(
                        dst=Register("r2"),
                        src="op",
                        op=AluExpr(Register("r2"), AluOp.SUB, ImmValue(5, prefix="#")),
                    ),
                ],
                fix_addr_size=True,
            )
        ]
    )

    root = _run_chunk_pass(root)
    block = root.insts[0]
    insts = block.insts

    assert len(insts) == 5
    assert isinstance(insts[0], RegWriteInst)
    assert str(insts[0].op.rhs) == "#2"
    assert isinstance(insts[1], RegWriteInst)
    assert isinstance(insts[2], NopInst)
    assert isinstance(insts[3], RegWriteInst)
    assert isinstance(insts[4], RegWriteInst)


def test_inc_reg_merge_fixed_barrier():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), AluOp.ADD, ImmValue(2, prefix="#")),
                    ),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), AluOp.ADD, ImmValue(3, prefix="#")),
                    ),
                ],
                fix_addr_size=True,
            )
        ]
    )

    root = _run_chunk_pass(root)
    insts = root.insts[0].insts
    # In fixed blocks, non-adjacent increments are not merged
    assert len(insts) == 3
    assert str(insts[0].op.rhs) == "#2"
    assert str(insts[2].op.rhs) == "#3"


def test_inc_reg_merge_does_not_cross_wmem_write_via_alias():
    # Regression: REG_WR w_freq op (w_freq + #5) ; WMEM_WR ; REG_WR w_freq op
    # (w_freq + #3).  WMEM_WR reads {w0..w5, r_wave, s14}; w_freq is an alias
    # of w0.  Without alias-aware read tracking, the pending +5 increment on
    # w_freq would be sunk past WMEM_WR and merged with +3, so the WMEM_WR
    # would observe the un-incremented value.
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("w_freq"),
                        src="op",
                        op=AluExpr(Register("w_freq"), AluOp.ADD, ImmValue(5, prefix="#")),
                    ),
                    WmemWriteInst(addr=ImmValue(0, prefix="&")),
                    RegWriteInst(
                        dst=Register("w_freq"),
                        src="op",
                        op=AluExpr(Register("w_freq"), AluOp.ADD, ImmValue(3, prefix="#")),
                    ),
                ]
            )
        ]
    )

    out = _run_chunk_pass(root)
    insts = out.insts[0].insts
    # The first +5 must remain before WMEM_WR; only the +3 may sit afterward.
    assert isinstance(insts[0], RegWriteInst)
    assert str(insts[0].op.rhs) == "#5"
    assert isinstance(insts[1], WmemWriteInst)
    # Whether the trailing +3 stays adjacent or merges into a #3 inc is an
    # implementation detail; what matters is that #5 never crosses WMEM_WR.
    assert any(
        isinstance(inst, RegWriteInst) and str(inst.op.rhs) == "#3"
        for inst in insts[2:]
    )
