from __future__ import annotations

from typing import cast

from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TestInst,
    TimeInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRLoop,
    RootNode,
)
from zcu_tools.program.v2.ir.operands import AluExpr, Literal, Register
from zcu_tools.program.v2.ir.passes.dataflow import (
    DeadTestEliminationLinear,
    DeadWriteEliminationLinear,
)
from zcu_tools.program.v2.ir.passes.loop import UnrollLoopPass
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    _run_linear_passes,
    make_default_pipeline,
)


def _config(**kwargs) -> PipeLineConfig:
    return PipeLineConfig(**kwargs)


def _flatten_root(root: RootNode) -> list[Instruction]:
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser
    parser = IRParser()
    lexer = IRLexer()
    return lexer.flatten(parser.unparse(root))


def _count_fixed_blocks(root: RootNode) -> int:
    from zcu_tools.program.v2.ir.traversal import walk_basic_blocks
    return sum(1 for bb in walk_basic_blocks(root) if bb.fix_addr_size)


def _counter_update(reg: str) -> BasicBlockNode:
    """Standard counter increment for tests."""
    return BasicBlockNode(insts=[
        RegWriteInst(dst=Register(reg), src="op", op=AluExpr(Register(reg), "+", Literal("#1")))
    ])


# ---------------------------------------------------------------------------
# DeadWriteEliminationLinear
# ---------------------------------------------------------------------------

def test_dead_write_elimination_removes_overwritten_write():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                RegWriteInst(dst=Register("s1"), src="imm", lit=Literal("#1")),
                RegWriteInst(dst=Register("s1"), src="imm", lit=Literal("#2")),
                NopInst(),
            ]),
        ]
    )

    _run_linear_passes([DeadWriteEliminationLinear()], root)

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], RegWriteInst)
    assert cast(RegWriteInst, bb.insts[0]).lit.value == "#2"


def test_dead_write_elimination_keeps_write_before_read():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                RegWriteInst(dst=Register("s1"), src="imm", lit=Literal("#1")),
                TestInst(op=AluExpr(Register("s1"), "-", Literal("#1"))),
                RegWriteInst(dst=Register("s1"), src="imm", lit=Literal("#2")),
            ]),
        ]
    )

    _run_linear_passes([DeadWriteEliminationLinear()], root)

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 3
    assert [
        cast(RegWriteInst, item).lit.value
        for item in bb.insts
        if isinstance(item, RegWriteInst)
    ] == [
        "#1",
        "#2",
    ]


def test_dead_write_elimination_removes_overwritten_write_in_basic_block():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                RegWriteInst(dst=Register("s1"), src="imm", lit=Literal("#1")),
                RegWriteInst(dst=Register("s1"), src="imm", lit=Literal("#2")),
            ]),
        ]
    )

    _run_linear_passes([DeadWriteEliminationLinear()], root)

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert bb.insts[0].lit.value == "#2"


# ---------------------------------------------------------------------------
# DeadTestEliminationLinear
# ---------------------------------------------------------------------------

def test_dead_test_elimination_removes_unused_test():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                TestInst(op=AluExpr(Register("r1"), "-", Literal("#10"))),
                NopInst(),
            ]),
        ]
    )

    _run_linear_passes([DeadTestEliminationLinear()], root)

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert isinstance(bb.insts[0], NopInst)


def test_dead_test_elimination_keeps_used_test():
    lbl = Label("loop")
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[TestInst(op=AluExpr(Register("r1"), "-", Literal("#10")))],
                branch=JumpInst(label=lbl, if_cond="NZ"),
            ),
        ]
    )

    _run_linear_passes([DeadTestEliminationLinear()], root)

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
                body=BlockNode(insts=[
                    BasicBlockNode(insts=[RegWriteInst(dst=Register("r1"), src="imm", lit=Literal("#1"))]),
                ]),
            )
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=512)
    ctx = PipeLineContext(config=pipeline.config, pmem_size=512)

    # 1. Unroll
    out = UnrollLoopPass().process(root, ctx)
    
    # 2. Merge blocks so the linear pass sees the writes together in one block
    from zcu_tools.program.v2.ir.passes.control_flow import BlockMergePass
    out = BlockMergePass().process(out, ctx)
    
    # 3. Linear (inc DeadWriteElimination)
    _run_linear_passes(pipeline.linear_passes, out)

    bbs = _collect_all_basic_blocks(out)
    # Should find only one write to r1 across all expanded copies.
    r1_writes = [
        i for bb in bbs for i in bb.insts
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
                        BasicBlockNode(insts=[RegWriteInst(dst=Register("r1"), src="op", op=AluExpr(Register("r0"), ""))]),
                        _counter_update("r0"),
                    ]
                ),
            )
        ]
    )

    out = UnrollLoopPass().process(root, PipeLineContext(config=_config()))

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
            BasicBlockNode(insts=[
                RegWriteInst(dst=Register("s1"), src="imm", lit=Literal("#1")),
                RegWriteInst(dst=Register("s1"), src="imm", lit=Literal("#2")),
            ]),
            BasicBlockNode(labels=[LabelInst(name=Label.make_new("dead"))], insts=[
                TimeInst(c_op="inc_ref", lit=Literal("#0")),
                TimeInst(c_op="inc_ref", lit=Literal("#1")),
                TimeInst(c_op="inc_ref", lit=Literal("#2")),
            ]),
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
    assert cast(TimeInst, bb1.insts[0]).lit.value == "#0"


def _collect_all_basic_blocks(root: RootNode) -> list[BasicBlockNode]:
    from zcu_tools.program.v2.ir.traversal import walk_basic_blocks
    return list(walk_basic_blocks(root))


def test_unroll_register_driven_jump_table_structure():
    """Verify the jump-table blocks have correct structure and conditional jumps."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Literal("#1"))])]),
            )
        ]
    )

    config = _config(max_unroll_factor=2)
    out = UnrollLoopPass().process(root, PipeLineContext(config=config))

    # k=2: 2 entry blocks + 2 back-edge blocks = 4 fixed blocks.
    assert _count_fixed_blocks(out) == 4

    emit = _flatten_root(out)
    cond_jumps = [
        inst for inst in emit if isinstance(inst, JumpInst) and inst.if_cond is not None
    ]
    # n==0 guard: JUMP -if(Z) -op(n - #0)
    assert any(j.if_cond == "Z" and str(j.op) == "r_n - #0" for j in cond_jumps)
