from __future__ import annotations

from typing import cast

from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    NopInst,
    RegWriteInst,
    TestInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRLoop,
    RootNode,
)
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    Register,
    SrcKeyword,
)
from zcu_tools.program.v2.ir.passes import walk_basic_blocks
from zcu_tools.program.v2.ir.passes.dataflow import (
    DeadTestEliminationPass,
    DeadWriteEliminationPass,
)
from zcu_tools.program.v2.ir.passes.loop import UnrollLoopPass
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    make_default_pipeline,
)


def _config(**kwargs) -> PipeLineConfig:
    return PipeLineConfig(**kwargs)


def _run_chunk_passes_on_root(root: RootNode, passes: list) -> RootNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    for pass_ in passes:
        chunks, _ = pass_.process(chunks, ctx)
    return parser.parse(chunks)


def _flatten_root(root: RootNode) -> list[Instruction]:
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser

    parser = IRParser()
    lexer = IRLexer()
    return lexer.flatten(parser.unparse(root))


def _count_fixed_blocks(root: RootNode) -> int:
    return sum(1 for bb in walk_basic_blocks(root) if bb.fix_addr_size)


def _counter_update(reg: str) -> BasicBlockNode:
    """Standard counter increment for tests."""
    return BasicBlockNode(
        insts=[
            RegWriteInst(
                dst=Register(reg),
                src=SrcKeyword.OP,
                op=AluExpr(Register(reg), AluOp.ADD, Immediate(1)),
            )
        ]
    )


# ---------------------------------------------------------------------------
# DeadWriteEliminationPass
# ---------------------------------------------------------------------------


def test_dead_write_elimination_removes_overwritten_write():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)),
                    RegWriteInst(dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)),
                    NopInst(),
                ]
            ),
        ]
    )

    root = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], RegWriteInst)
    assert str(cast(RegWriteInst, bb.insts[0]).lit) == "#2"


def test_dead_write_elimination_keeps_write_before_read():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)),
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(1))),
                    RegWriteInst(dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)),
                ]
            ),
        ]
    )

    root = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 3
    assert [
        str(cast(RegWriteInst, item).lit)
        for item in bb.insts
        if isinstance(item, RegWriteInst)
    ] == [
        "#1",
        "#2",
    ]


def test_dead_write_elimination_removes_overwritten_write_in_basic_block():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)),
                    RegWriteInst(dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)),
                ]
            ),
        ]
    )

    root = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert str(cast(RegWriteInst, bb.insts[0]).lit) == "#2"


# ---------------------------------------------------------------------------
# DeadTestEliminationPass
# ---------------------------------------------------------------------------


def test_dead_test_elimination_removes_unused_test():
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10))),
                    NopInst(),
                ]
            ),
        ]
    )

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], NopInst)


def test_dead_test_elimination_keeps_used_test():
    lbl = Label("loop")
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10)))],
                branch=JumpInst(label=lbl, if_cond="NZ"),
            ),
        ]
    )

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], TestInst)


# ---------------------------------------------------------------------------
# UnrollLoopPass + DeadWriteElimination
# ---------------------------------------------------------------------------


def test_unroll_full_expansion_removes_overwritten_writes_in_body():
    """body: REG_WR r1 imm #1. n=2. Full expansion = 2 copies.
    Only the last write to r1 should survive.
    """
    root = RootNode(
        insts=[
            IRLoop(
                name="L",
                counter_reg="r_cnt",
                n=2,
                body=BlockNode(
                    insts=[
                        BasicBlockNode(
                            insts=[
                                RegWriteInst(
                                    dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)
                                )
                            ]
                        ),
                    ]
                ),
            )
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=512)
    ctx = PipeLineContext(config=pipeline.config, pmem_budget=512)

    # 1. Unroll
    out, _ = UnrollLoopPass().process(root, ctx)

    # 2. Merge blocks so the linear pass sees the writes together in one block
    from zcu_tools.program.v2.ir.passes.control_flow import BlockMergePass

    out, _ = BlockMergePass().process(out, ctx)

    # 3. Chunk passes (inc DeadWriteElimination)
    out = _run_chunk_passes_on_root(out, pipeline.chunk_passes)

    bbs = _collect_all_basic_blocks(out)
    # Should find only one write to r1 across all expanded copies.
    r1_writes = [
        i
        for bb in bbs
        for i in bb.insts
        if isinstance(i, RegWriteInst) and i.dst.name == "r1"
    ]
    assert len(r1_writes) == 1


def test_unroll_full_expansion_keeps_counter_init_for_counter_dependent_body():
    """When body reads/increments loop counter, full expansion must still emit
    the counter init (`REG_WR counter imm #0`) before the first body copy."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r0",
                n=1,
                body=BlockNode(
                    insts=[
                        BasicBlockNode(
                            insts=[
                                RegWriteInst(
                                    dst=Register("r1"),
                                    src=SrcKeyword.OP,
                                    op=AluExpr(Register("r0"), AluOp.NONE),
                                )
                            ]
                        ),
                        _counter_update("r0"),
                    ]
                ),
            )
        ]
    )

    out, _ = UnrollLoopPass().process(
        root, PipeLineContext(config=_config(), pmem_budget=512)
    )

    # Full expansion: counter init BB + body BB + increment BB = 3 BasicBlockNodes
    assert len(out.insts) == 3
    init_bb = out.insts[0]
    assert isinstance(init_bb, BasicBlockNode)
    assert len(init_bb.insts) > 0
    assert isinstance(init_bb.insts[0], RegWriteInst)
    init = cast(RegWriteInst, init_bb.insts[0])
    assert init.dst.name == "r0"


# ---------------------------------------------------------------------------
# Global Pipeline Integration
# ---------------------------------------------------------------------------


def test_default_pipeline_can_disable_all_optimization_passes():
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser

    """Disabling all pass flags should keep the IR layout unchanged."""
    Label.reset()
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)),
                    RegWriteInst(dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)),
                ]
            ),
            BasicBlockNode(
                labels=[LabelInst(name=Label.make_new("dead"))],
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(0)),
                    TimeInst(c_op="inc_ref", lit=Immediate(1)),
                    TimeInst(c_op="inc_ref", lit=Immediate(2)),
                ],
            ),
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=8192)
    pipeline.config.disable_all_opt = True

    lexer = IRLexer()
    parser = IRParser(pmem_size=8192)
    insts = lexer.flatten(parser.unparse(root))

    out_insts, _ctx = pipeline(insts)
    out = parser.parse(lexer.lex(out_insts))
    assert len(out.insts) == 2
    bb0 = cast(BasicBlockNode, out.insts[0])
    assert isinstance(bb0.insts[0], RegWriteInst)
    assert isinstance(bb0.insts[1], RegWriteInst)
    bb1 = cast(BasicBlockNode, out.insts[1])
    assert len(bb1.labels) == 1
    assert isinstance(bb1.labels[0], LabelInst)
    assert isinstance(bb1.insts[0], TimeInst)
    assert str(cast(TimeInst, bb1.insts[0]).lit) == "#0"  # type: ignore


def _collect_all_basic_blocks(root: RootNode) -> list[BasicBlockNode]:
    return list(walk_basic_blocks(root))


def test_unroll_register_driven_jump_table_structure():
    """Verify the dispatch-table loop has the expected fixed/free split."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(
                    insts=[
                        BasicBlockNode(
                            insts=[TimeInst(c_op="inc_ref", lit=Immediate(1))]
                        )
                    ]
                ),
            )
        ]
    )

    config = _config(max_unroll_factor=2)
    out, _ = UnrollLoopPass().process(
        root, PipeLineContext(config=config, pmem_budget=512)
    )

    # Only the dispatch-table stubs remain fixed.
    assert _count_fixed_blocks(out) == 2

    emit = _flatten_root(out)
    cond_jumps = [
        inst for inst in emit if isinstance(inst, JumpInst) and inst.if_cond is not None
    ]
    # n==0 guard: JUMP -if(Z) -op(n - #0)
    assert any(j.if_cond == "Z" and str(j.op) == "r_n - #0" for j in cond_jumps)
