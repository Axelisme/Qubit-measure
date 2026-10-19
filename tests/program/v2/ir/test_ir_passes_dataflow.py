"""Dead-write and dead-test elimination contracts at the chunk-pass seam."""

from __future__ import annotations

from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    CallInst,
    DmemWriteInst,
    JumpInst,
    NopInst,
    RegWriteInst,
    TestInst,
)
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    MemAddr,
    Register,
    SideWrite,
    SrcKeyword,
)
from zcu_tools.program.v2.ir.passes.dataflow import (
    DeadTestEliminationPass,
    DeadWriteEliminationPass,
)
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext

from .pass_test_support import run_chunk_passes_on_root, walk_basic_blocks

# ---------------------------------------------------------------------------
# DeadWriteEliminationPass
# ---------------------------------------------------------------------------


def test_dead_write_elimination_removes_overwritten_write():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)
                    ),
                    NopInst(),
                ]
            ),
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    inst = bb.insts[0]
    assert isinstance(inst, RegWriteInst)
    assert str(inst.lit) == "#2"


def test_dead_write_elimination_keeps_write_before_read():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(1))),
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)
                    ),
                ]
            ),
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 3
    assert [str(item.lit) for item in bb.insts if isinstance(item, RegWriteInst)] == [
        "#1",
        "#2",
    ]


def test_dead_write_elimination_removes_overwritten_write_in_basic_block():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)
                    ),
                ]
            ),
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    inst = bb.insts[0]
    assert isinstance(inst, RegWriteInst)
    assert str(inst.lit) == "#2"


def test_dead_write_elimination_keeps_reg_write_when_side_write_is_read():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.IMM,
                        lit=Immediate(0),
                        wr=SideWrite(Register("r1"), "op"),
                    ),
                    RegWriteInst(
                        dst=Register("r2"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(0))),
                ]
            )
        ]
    )

    out = run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])
    bb = out.insts[0]

    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 3
    first = bb.insts[0]
    assert isinstance(first, RegWriteInst)
    assert first.wr == SideWrite(Register("r1"), "op")


def test_dead_write_elimination_never_deletes_dmem_write_with_side_write():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    DmemWriteInst(
                        dst=MemAddr(0),
                        wr=SideWrite(Register("r1"), "op"),
                    ),
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                ]
            )
        ]
    )

    out = run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])
    bb = out.insts[0]

    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], DmemWriteInst)


def test_dead_write_elimination_does_not_cross_call_boundary():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    CallInst(label=LabelRef(Label("sub"))),
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)
                    ),
                ]
            )
        ]
    )

    out = run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])
    bb = out.insts[0]

    assert isinstance(bb, BasicBlockNode)
    writes = [inst for inst in bb.insts if isinstance(inst, RegWriteInst)]
    assert [inst.lit for inst in writes] == [Immediate(1), Immediate(2)]


# ---------------------------------------------------------------------------
# DeadTestEliminationPass
# ---------------------------------------------------------------------------


def test_dead_test_elimination_removes_unused_test():
    exit_label = Label("exit")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10))),
                    NopInst(),
                ],
                branch=JumpInst(label=LabelRef(exit_label)),
            ),
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], NopInst)


def test_dead_test_elimination_keeps_pending_test_before_fallthrough_exit():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10))),
                    NopInst(),
                ]
            ),
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.IMM,
                        lit=Immediate(1),
                        if_cond="NZ",
                    )
                ]
            ),
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert isinstance(bb.insts[0], TestInst)
    assert isinstance(bb.insts[1], NopInst)


def test_dead_test_elimination_keeps_used_test():
    lbl = Label("loop")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10)))],
                branch=JumpInst(label=LabelRef(lbl), if_cond="NZ"),
            ),
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], TestInst)


def test_dead_test_elimination_uf_instruction_overwrites_flag():
    # A REG_WR -uf overwrites the ALU flags, so a preceding TEST whose flag is
    # never read before it becomes dead. The -uf REG_WR itself is kept.
    lbl = Label("loop")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10))),
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r2"), AluOp.ADD, Immediate(1)),
                        uf=True,
                    ),
                ],
                branch=JumpInst(label=LabelRef(lbl), if_cond="NZ"),
            ),
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], RegWriteInst)
    assert bb.insts[0].uf is True


def test_dead_test_elimination_keeps_test_when_uf_instruction_follows_consumption():
    # TEST consumed by the conditional branch is live even if a -uf REG_WR
    # would otherwise overwrite flags — the -uf write is after the branch
    # point only conceptually; here the branch is the block terminal so the
    # TEST before a non-uf inst stays. Guards against over-eager removal.
    lbl = Label("loop")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r2"), AluOp.ADD, Immediate(1)),
                        uf=True,
                    ),
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10))),
                ],
                branch=JumpInst(label=LabelRef(lbl), if_cond="NZ"),
            ),
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], RegWriteInst)
    assert isinstance(bb.insts[1], TestInst)


def test_dead_test_elimination_reg_write_if_cond_consumes_flag():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(0))),
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.IMM,
                        lit=Immediate(1),
                        if_cond="Z",
                    ),
                ]
            )
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadTestEliminationPass()])
    bb = root.insts[0]

    assert isinstance(bb, BasicBlockNode)
    assert isinstance(bb.insts[0], TestInst)
    assert isinstance(bb.insts[1], RegWriteInst)


def test_dead_test_elimination_if_cond_uf_consumes_old_flag_before_overwrite():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(0))),
                    RegWriteInst(
                        dst=Register("r2"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r2"), AluOp.ADD, Immediate(1)),
                        if_cond="Z",
                        uf=True,
                    ),
                ]
            )
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadTestEliminationPass()])
    bb = root.insts[0]

    assert isinstance(bb, BasicBlockNode)
    assert isinstance(bb.insts[0], TestInst)
    assert isinstance(bb.insts[1], RegWriteInst)
    assert bb.insts[1].if_cond == "Z"
    assert bb.insts[1].uf is True


def test_dead_test_elimination_does_not_delete_test_before_call_boundary():
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(0))),
                    CallInst(label=LabelRef(Label("sub"))),
                ]
            )
        ]
    )

    root = run_chunk_passes_on_root(root, [DeadTestEliminationPass()])
    bb = root.insts[0]

    assert isinstance(bb, BasicBlockNode)
    assert isinstance(bb.insts[0], TestInst)
    assert isinstance(bb.insts[1], CallInst)


# ---------------------------------------------------------------------------
# DeadTestEliminationPass — uncovered branches (8.4)
# ---------------------------------------------------------------------------


def _run_dead_test(root: BlockNode) -> BlockNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    chunks, _ = DeadTestEliminationPass().process(chunks, ctx)
    return parser.parse(chunks)


def _test_inst() -> TestInst:
    return TestInst(op=AluExpr(Register("r0"), AluOp.SUB, Immediate(0)))


def test_dead_test_consecutive_tests_first_is_dead():
    """Two consecutive TestInsts: the first is dead (overwritten before consumed)."""
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    _test_inst(),  # dead: overwritten by next TEST
                    _test_inst(),  # alive: consumed by branch
                ],
                branch=JumpInst(
                    label=LabelRef(Label("end")),
                    if_cond="Z",
                    op=AluExpr(Register("r0"), AluOp.SUB, Immediate(0)),
                ),
            )
        ]
    )
    out = _run_dead_test(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    test_insts = [i for i in bb.insts if isinstance(i, TestInst)]
    assert len(test_insts) == 1  # first dead TEST removed


def test_dead_test_uf_inst_kills_pending_test():
    """-uf instruction overwrites ALU flags → preceding pending TEST is dead."""
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    _test_inst(),  # dead: flags overwritten by -uf below
                    RegWriteInst(
                        dst=Register("r1"),
                        src=SrcKeyword.IMM,
                        lit=Immediate(0),
                        uf=True,  # -uf side-effect overwrites ALU flags
                    ),
                ],
                branch=None,  # no branch consuming flags
            )
        ]
    )
    out = _run_dead_test(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    # TEST must be removed; the -uf REG_WR must remain
    test_insts = [i for i in bb.insts if isinstance(i, TestInst)]
    assert len(test_insts) == 0
    reg_insts = [i for i in bb.insts if isinstance(i, RegWriteInst)]
    assert len(reg_insts) == 1


def test_dead_test_conditional_branch_and_fallthrough_keep_flags():
    first = _test_inst()
    fallthrough = _test_inst()
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[first],
                branch=JumpInst(label=LabelRef(Label("skip")), if_cond="Z"),
            ),
            BasicBlockNode(insts=[fallthrough]),
        ]
    )
    out = _run_dead_test(root)
    blocks = list(walk_basic_blocks(out))
    assert len(blocks) == 2
    assert blocks[0].insts == [first]
    assert blocks[0].branch == JumpInst(label=LabelRef(Label("skip")), if_cond="Z")
    assert blocks[1].insts == [fallthrough]


# DeadWriteEliminationPass — disable_opt guard (line 65-66)
# ---------------------------------------------------------------------------


def test_dead_write_disable_opt_block_not_modified():
    """disable_opt=True block: DeadWriteEliminationPass must skip it (line 65-66)."""
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)
                    ),
                ],
                disable_opt=True,
            )
        ]
    )
    out = run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    # Both writes must survive (disable_opt blocks are left untouched)
    assert len([i for i in bb.insts if isinstance(i, RegWriteInst)]) == 2


def test_dead_test_disable_opt_block_not_modified():
    """disable_opt=True block: DeadTestEliminationPass must not modify it (line 54)."""
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    _test_inst(),  # would normally be dead (no branch consumer)
                ],
                branch=None,
                disable_opt=True,
            )
        ]
    )
    out = _run_dead_test(root)
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    # TEST must NOT be removed (block is disable_opt)
    assert len([i for i in bb.insts if isinstance(i, TestInst)]) == 1
