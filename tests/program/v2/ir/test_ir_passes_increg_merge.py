from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.node import BasicBlockNode, RootNode
from zcu_tools.program.v2.ir.operands import AluExpr, Literal, Register
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
                        op=AluExpr(Register("r1"), "+", Literal("#2")),
                    ),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), "+", Literal("#3")),
                    ),
                    RegWriteInst(
                        dst=Register("r2"),
                        src="op",
                        op=AluExpr(Register("r2"), "+", Literal("#5")),
                    ),
                    RegWriteInst(
                        dst=Register("r2"),
                        src="op",
                        op=AluExpr(Register("r2"), "-", Literal("#1")),
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
    assert insts[1].op.rhs.value == "#5"  # r1 + #5
    assert insts[1].dst.name == "r1"
    assert insts[2].op.rhs.value == "#4"  # r2 + #4
    assert insts[2].dst.name == "r2"


def test_inc_reg_merge_free_flush_on_read():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), "+", Literal("#2")),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r1")),
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), "+", Literal("#3")),
                    ),
                ]
            )
        ]
    )

    root = _run_chunk_pass(root)
    insts = root.insts[0].insts

    assert len(insts) == 3
    assert insts[0].op.rhs.value == "#2"
    assert isinstance(insts[1], TimeInst)
    assert insts[2].op.rhs.value == "#3"


def test_inc_reg_merge_free_can_cross_port_write():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), "+", Literal("#1")),
                    ),
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&1"),
                        time=Literal("@0"),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), "+", Literal("#1")),
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
    assert insts[2].op.rhs.value == "#2"
    assert insts[2].dst.name == "r0"


def test_inc_reg_merge_free_cpmg_like_unrolled_body():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&1"),
                        time=Literal("@0"),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), "+", Literal("#1")),
                    ),
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&1"),
                        time=Literal("@0"),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), "+", Literal("#1")),
                    ),
                    PortWriteInst(
                        dst=Literal("2"),
                        src="wmem",
                        addr=Literal("&1"),
                        time=Literal("@0"),
                    ),
                    TimeInst(c_op="inc_ref", r1=Register("r4")),
                    RegWriteInst(
                        dst=Register("r0"),
                        src="op",
                        op=AluExpr(Register("r0"), "+", Literal("#1")),
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
    assert insts[-1].op.rhs.value == "#3"


def test_inc_reg_merge_fixed_basic_is_skipped():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), "+", Literal("#2")),
                    ),
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), "+", Literal("#3")),
                    ),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r2"),
                        src="op",
                        op=AluExpr(Register("r2"), "+", Literal("#5")),
                    ),
                    RegWriteInst(
                        dst=Register("r2"),
                        src="op",
                        op=AluExpr(Register("r2"), "-", Literal("#5")),
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
    assert insts[0].op.rhs.value == "#2"
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
                        op=AluExpr(Register("r1"), "+", Literal("#2")),
                    ),
                    NopInst(),
                    RegWriteInst(
                        dst=Register("r1"),
                        src="op",
                        op=AluExpr(Register("r1"), "+", Literal("#3")),
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
    assert insts[0].op.rhs.value == "#2"
    assert insts[2].op.rhs.value == "#3"
