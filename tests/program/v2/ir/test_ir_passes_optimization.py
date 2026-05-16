from __future__ import annotations

from typing import Iterator, cast

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
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRLoop,
    IRNode,
)
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    Register,
    SrcKeyword,
)
from zcu_tools.program.v2.ir.passes.dataflow import (
    DeadTestEliminationPass,
    DeadWriteEliminationPass,
)
from zcu_tools.program.v2.ir.passes.loop import UnrollLoopPass
from zcu_tools.program.v2.ir.pipeline import (
    AbsIRTreePass,
    PipeLineConfig,
    PipeLineContext,
    make_default_pipeline,
)


def _walk_basic_blocks(node: IRNode) -> Iterator[BasicBlockNode]:
    if isinstance(node, BasicBlockNode):
        yield node
    elif isinstance(node, BlockNode):
        for child in node.insts:
            yield from _walk_basic_blocks(child)
    elif isinstance(node, IRLoop):
        yield from _walk_basic_blocks(node.body)
    elif isinstance(node, IRBranch):
        for case in node.cases:
            yield from _walk_basic_blocks(case)


def _config(**kwargs) -> PipeLineConfig:
    return PipeLineConfig(**kwargs)


def _apply_tree_pass(
    node: IRNode, pass_: AbsIRTreePass, ctx: PipeLineContext
) -> IRNode:
    """Apply an AbsIRTreePass to a single IRNode (post-order, no ChunkPass between layers)."""
    from zcu_tools.program.v2.ir.node import IRBranch, IRLoop

    # Recurse into children first (post-order).
    if isinstance(node, IRLoop):
        new_body = _apply_tree_pass(node.body, pass_, ctx)
        if new_body is not node.body:
            node.body = new_body
    elif isinstance(node, IRBranch):
        for i, case in enumerate(node.cases):
            new_case = _apply_tree_pass(case, pass_, ctx)
            if new_case is not case:
                node.cases[i] = new_case
    elif isinstance(node, BlockNode):
        for i, child in enumerate(node.insts):
            new_child = _apply_tree_pass(child, pass_, ctx)
            if new_child is not child:
                node.insts[i] = new_child
        return node
    else:
        return node

    return pass_.transform(node, ctx)


def _apply_tree_pass_to_root(
    root: BlockNode, pass_: AbsIRTreePass, ctx: PipeLineContext
) -> tuple[BlockNode, bool]:
    """Apply an AbsIRTreePass to all nodes in a BlockNode tree, return (result, changed)."""
    before_id = id(root)
    result = _apply_tree_pass(root, pass_, ctx)
    changed = id(result) != before_id or any(
        id(child) != id(orig)
        for child, orig in zip(
            (result.insts if isinstance(result, BlockNode) else []),
            root.insts,
        )
    )
    if isinstance(result, BlockNode):
        return result, changed
    return BlockNode(insts=[result]), changed


def _run_chunk_passes_on_root(root: BlockNode, passes: list) -> BlockNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    for pass_ in passes:
        chunks, _ = pass_.process(chunks, ctx)
    return parser.parse(chunks)


def _flatten_root(root: BlockNode) -> list[Instruction]:
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser

    parser = IRParser()
    lexer = IRLexer()
    return lexer.flatten(parser.unparse(root))


def _count_fixed_blocks(root: BlockNode) -> int:
    return sum(1 for bb in _walk_basic_blocks(root) if bb.disable_opt)


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

    root = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], RegWriteInst)
    assert str(cast(RegWriteInst, bb.insts[0]).lit) == "#2"


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

    root = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert str(cast(RegWriteInst, bb.insts[0]).lit) == "#2"


# ---------------------------------------------------------------------------
# DeadTestEliminationPass
# ---------------------------------------------------------------------------


def test_dead_test_elimination_removes_unused_test():
    root = BlockNode(
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
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Immediate(10)))],
                branch=JumpInst(label=LabelRef(lbl), if_cond="NZ"),
            ),
        ]
    )

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

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

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

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

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], RegWriteInst)
    assert isinstance(bb.insts[1], TestInst)


# ---------------------------------------------------------------------------
# UnrollLoopPass + DeadWriteElimination
# ---------------------------------------------------------------------------


def test_unroll_full_expansion_removes_overwritten_writes_in_body():
    """body: REG_WR r1 imm #1. n=2. Full expansion = 2 copies.
    Only the last write to r1 should survive.
    """
    root = BlockNode(
        insts=[
            IRLoop(
                name="L",
                counter_reg=Register("r0"),
                n=2,
                body=BlockNode(
                    insts=[
                        BasicBlockNode(
                            insts=[
                                RegWriteInst(
                                    dst=Register("r1"),
                                    src=SrcKeyword.IMM,
                                    lit=Immediate(1),
                                )
                            ]
                        ),
                    ]
                ),
            )
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=512)
    ctx = PipeLineContext(
        config=pipeline.config, pmem_budget=512, available_regs={"r14"}
    )

    # 1. Unroll
    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)

    # 2. Merge blocks so the linear pass sees the writes together in one block
    from zcu_tools.program.v2.ir.passes.control_flow import BlockMergePass

    out = _run_chunk_passes_on_root(out, [BlockMergePass()])

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
    root = BlockNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg=Register("r0"),
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

    out, _ = _apply_tree_pass_to_root(
        root, UnrollLoopPass(), PipeLineContext(config=_config(), pmem_budget=512)
    )

    # Full expansion: the IRLoop is replaced by a BlockNode(insts=[init_bb, ...])
    assert len(out.insts) == 1
    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    assert len(expanded.insts) == 3
    init_bb = expanded.insts[0]
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
            BasicBlockNode(
                labels=[LabelInst(name=Label("dead"))],
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


def _collect_all_basic_blocks(root: BlockNode) -> list[BasicBlockNode]:
    return list(_walk_basic_blocks(root))


def test_unroll_register_driven_jump_table_structure():
    """UnrollLoopPass emits an IRDispatch node (not pre-lowered stubs) for the
    dispatch table; stubs are produced later by the pipeline fallback."""
    from zcu_tools.program.v2.ir.node import IRDispatch

    root = BlockNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg=Register("r0"),
                n=Register("r1"),
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
    out, _ = _apply_tree_pass_to_root(
        root,
        UnrollLoopPass(),
        PipeLineContext(config=config, pmem_budget=512, available_regs={"r14"}),
    )

    # UnrollLoopPass output should contain an IRDispatch node (not yet lowered).
    def _collect_dispatch(node: IRNode) -> list[IRDispatch]:
        if isinstance(node, IRDispatch):
            return [node]
        if isinstance(node, BlockNode):
            result = []
            for child in node.insts:
                result.extend(_collect_dispatch(child))
            return result
        return []

    dispatch_nodes = _collect_dispatch(out)
    assert len(dispatch_nodes) == 1
    assert len(dispatch_nodes[0].target_labels) == 2

    emit = _flatten_root(out)
    cond_jumps = [
        inst for inst in emit if isinstance(inst, JumpInst) and inst.if_cond is not None
    ]
    # n==0 guard: JUMP -if(Z) -op(n - #0)
    assert any(j.if_cond == "Z" and str(j.op) == "r1 - #0" for j in cond_jumps)


def test_clone_body_remaps_internal_label_refs():
    """_clone_body must remap LabelRef targets inside the cloned body.

    A body containing a conditional back-jump to its own internal label must
    have that LabelRef updated to point to the cloned (renamed) label, not the
    original label in the un-cloned body.
    """
    from zcu_tools.program.v2.ir.passes.loop.unroll import _clone_body

    inner_label = Label("inner")
    body: list = [
        BasicBlockNode(
            labels=[LabelInst(name=inner_label)],
            insts=[TimeInst(c_op="inc_ref", lit=Immediate(1))],
            branch=JumpInst(
                label=LabelRef(inner_label),
                if_cond="S",
                op=AluExpr(Register("r0"), AluOp.SUB, Immediate(1)),
            ),
        )
    ]

    allocated: set[str] = {"inner"}
    cloned = _clone_body(body, allocated)

    assert len(cloned) == 1
    bb = cloned[0]
    assert isinstance(bb, BasicBlockNode)

    # The cloned label must have a different name (suffix added).
    cloned_label_name = bb.labels[0].name
    assert cloned_label_name != inner_label, "cloned LabelInst must be renamed"

    # The branch LabelRef must point to the same renamed label.
    assert bb.branch is not None
    assert isinstance(bb.branch, JumpInst)
    assert bb.branch.label is not None
    assert not bb.branch.label.is_pseudo()
    assert bb.branch.label.as_label() == cloned_label_name, (
        "cloned branch LabelRef must point to the renamed label"
    )
