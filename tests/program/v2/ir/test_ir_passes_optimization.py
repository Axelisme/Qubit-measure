from __future__ import annotations

from collections.abc import Iterator

import pytest
from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    CallInst,
    DmemWriteInst,
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
    MemAddr,
    Register,
    SideWrite,
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


def _apply_tree_pass_to_root(
    root: BlockNode, pass_: AbsIRTreePass, ctx: PipeLineContext
) -> tuple[BlockNode, bool]:
    """Run an AbsIRTreePass over a tree via the real pipeline post-order driver.

    Returns (result, changed). `changed` is best-effort: True when the root
    object identity changed.
    """
    from zcu_tools.program.v2.ir.pipeline import _optimize_tree

    result = _optimize_tree(root, [pass_], ctx)
    if isinstance(result, BlockNode):
        return result, result is not root
    return BlockNode(insts=[result]), True


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

    root = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

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

    root = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])

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

    out = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])
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

    out = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])
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

    out = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])
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

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

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

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])

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

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])
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

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])
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

    root = _run_chunk_passes_on_root(root, [DeadTestEliminationPass()])
    bb = root.insts[0]

    assert isinstance(bb, BasicBlockNode)
    assert isinstance(bb.insts[0], TestInst)
    assert isinstance(bb.insts[1], CallInst)


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
    init = init_bb.insts[0]
    assert isinstance(init, RegWriteInst)
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
    bb0 = out.insts[0]
    assert isinstance(bb0, BasicBlockNode)
    assert isinstance(bb0.insts[0], RegWriteInst)
    assert isinstance(bb0.insts[1], RegWriteInst)
    bb1 = out.insts[1]
    assert isinstance(bb1, BasicBlockNode)
    assert len(bb1.labels) == 1
    assert isinstance(bb1.labels[0], LabelInst)
    time_inst = bb1.insts[0]
    assert isinstance(time_inst, TimeInst)
    assert str(time_inst.lit) == "#0"


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
    assert any(j.if_cond == "Z" for j in cond_jumps)


def test_clone_renamed_remaps_internal_label_refs():
    """clone_renamed must remap LabelRef targets inside the cloned body.

    A body containing a conditional back-jump to its own internal label must
    have that LabelRef updated to point to the cloned (renamed) label, not the
    original label in the un-cloned body.
    """
    from zcu_tools.program.v2.ir.node import clone_renamed

    inner_label = Label("inner")
    body = BlockNode(
        insts=[
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
    )

    allocated: set[str] = {"inner"}
    cloned = clone_renamed(body, allocated)

    assert isinstance(cloned, BlockNode)
    assert len(cloned.insts) == 1
    bb = cloned.insts[0]
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


def test_clone_renamed_uniquifies_nested_structure_names():
    """clone_renamed must give a nested IRLoop a fresh name so the labels its
    later unparse() synthesises (``{name}_start`` etc.) do not collide."""
    from zcu_tools.program.v2.ir.node import clone_renamed

    inner = IRLoop(
        name="inner",
        counter_reg=Register("r2"),
        n=2,
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )
    body = BlockNode(insts=[inner])

    allocated: set[str] = {"inner"}
    cloned = clone_renamed(body, allocated)

    assert isinstance(cloned, BlockNode)
    cloned_loop = cloned.insts[0]
    assert isinstance(cloned_loop, IRLoop)
    assert cloned_loop.name != "inner", "nested loop name must be uniquified"


def test_clone_renamed_remaps_dmem_addr_table_labels():
    """clone_renamed must remap the entry labels inside a DmemAddr.

    A dmem dispatch instruction (REG_WR s15 op (idx + DmemAddr)) inside a
    cloned loop body must have its DmemAddr.table_labels remapped to the
    cloned case-entry labels — otherwise every unrolled copy's dispatch would
    point at the first copy's case entries.
    """
    from zcu_tools.program.v2.ir.node import clone_renamed
    from zcu_tools.program.v2.ir.operands import AluExpr, AluOp, DmemAddr

    entry = Label("case_entry_0")
    # A block that both *defines* `entry` and references it via a DmemAddr.
    body = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("s15"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r0"), AluOp.ADD, DmemAddr((entry,))),
                    )
                ],
            ),
            BasicBlockNode(labels=[LabelInst(name=entry)], insts=[NopInst()]),
        ]
    )

    cloned = clone_renamed(body, {"case_entry_0"})
    assert isinstance(cloned, BlockNode)

    # The defining LabelInst was renamed.
    entry_block = cloned.insts[1]
    assert isinstance(entry_block, BasicBlockNode)
    cloned_entry = entry_block.labels[0].name
    assert cloned_entry != entry

    # The DmemAddr.table_labels must point at the *cloned* entry label.
    disp_block = cloned.insts[0]
    assert isinstance(disp_block, BasicBlockNode)
    inst = disp_block.insts[0]
    assert isinstance(inst, RegWriteInst)
    assert isinstance(inst.op, AluExpr) and isinstance(inst.op.rhs, DmemAddr)
    assert inst.op.rhs.table_labels == (cloned_entry,), (
        "DmemAddr.table_labels must be remapped to the cloned entry label"
    )


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


def test_dead_test_conditional_jump_in_insts_consumes_flag():
    """JumpInst with if_cond inside the insts list (not branch) consumes pending TEST.

    BasicBlockNode.__post_init__ forbids JumpInst in .insts, so we test
    _find_dead_indices directly — the method is designed to handle arbitrary
    instruction lists, including ones that may arrive from non-IR paths.
    """
    from zcu_tools.program.v2.ir.passes.dataflow.dead_test import (
        DeadTestEliminationPass,
    )

    pass_ = DeadTestEliminationPass()
    lbl = Label("skip")
    insts = [
        _test_inst(),  # alive: consumed by cond jump below
        JumpInst(
            label=LabelRef(lbl),
            if_cond="Z",
            op=AluExpr(Register("r0"), AluOp.SUB, Immediate(0)),
        ),
        _test_inst(),  # kept: branch=None is a possible fall-through boundary
    ]
    dead = pass_._find_dead_indices(insts, branch=None)  # type: ignore[attr-defined]
    # The first TEST (index 0) should NOT be dead (consumed by cond jump at index 1)
    # The second TEST (index 2) is also kept conservatively because branch=None
    # means a following block may consume the flags.
    assert 0 not in dead
    assert 2 not in dead


# ---------------------------------------------------------------------------
# Pipeline — non-convergence warning and disable_opt violation (8.7)
# ---------------------------------------------------------------------------


def test_pipeline_non_convergence_emits_warning(caplog):
    """_run_chunklist_opt logs a warning when passes oscillate past max_opt_iterations."""
    import logging

    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkListPass,
        ChunkList,
        _run_chunklist_opt,
    )

    class OscillatingPass(AbsChunkListPass):
        """Always reports changed=True — simulates an oscillating pass."""

        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            return chunks, True

    ctx = PipeLineContext(config=PipeLineConfig(max_opt_iterations=2), pmem_budget=1024)
    chunks = IRParser().unparse(BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]))

    with caplog.at_level(logging.WARNING, logger="zcu_tools.program.v2.ir.pipeline"):
        _run_chunklist_opt([], [OscillatingPass()], chunks, ctx)

    assert any("did not converge" in r.message for r in caplog.records)
    assert any("OscillatingPass" in r.message for r in caplog.records)


def test_pipeline_non_convergence_diagnostic_does_not_mutate_live_chunks(caplog):
    import logging

    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkListPass,
        ChunkList,
        _run_chunklist_opt,
    )

    class DiagnosticMutatingPass(AbsChunkListPass):
        def __init__(self) -> None:
            self.calls = 0

        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            self.calls += 1
            if self.calls > ctx.config.max_opt_iterations:
                block = chunks[0]
                assert isinstance(block, BasicBlockNode)
                block.insts.append(NopInst())
            return chunks, True

    ctx = PipeLineContext(config=PipeLineConfig(max_opt_iterations=2), pmem_budget=1024)
    block = BasicBlockNode(insts=[NopInst()])
    chunks: ChunkList = [block]

    with caplog.at_level(logging.WARNING, logger="zcu_tools.program.v2.ir.pipeline"):
        out = _run_chunklist_opt([], [DiagnosticMutatingPass()], chunks, ctx)

    assert out is chunks
    assert block.insts == [NopInst()]
    assert any("DiagnosticMutatingPass" in r.message for r in caplog.records)


def test_pipeline_disable_opt_violation_raises():
    """A ChunkPass that changes word count of a disable_opt block must raise ValueError."""
    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkPass,
        ChunkList,
        _run_passes,
    )

    class WordCountChangingPass(AbsChunkPass):
        """Appends a NopInst to any disable_opt block — illegal word-count change."""

        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            changed = False
            for chunk in chunks:
                if isinstance(chunk, BasicBlockNode) and chunk.disable_opt:
                    chunk.insts.append(NopInst())
                    changed = True
            return chunks, changed

    import pytest

    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    fixed_bb = BasicBlockNode(insts=[NopInst()], disable_opt=True)
    chunks: ChunkList = [fixed_bb]

    with pytest.raises(ValueError, match="disable_opt"):
        _run_passes([WordCountChangingPass()], chunks, ctx)


def test_pipeline_disable_opt_deletion_raises():
    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkPass,
        ChunkList,
        _run_passes,
    )

    class DeletingPass(AbsChunkPass):
        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            return [
                chunk
                for chunk in chunks
                if not (isinstance(chunk, BasicBlockNode) and chunk.disable_opt)
            ], True

    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    fixed_bb = BasicBlockNode(insts=[NopInst()], disable_opt=True)

    with pytest.raises(ValueError, match="disable_opt"):
        _run_passes([DeletingPass()], [fixed_bb], ctx)


def test_pipeline_disable_opt_replacement_raises():
    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkPass,
        ChunkList,
        _run_passes,
    )

    class ReplacingPass(AbsChunkPass):
        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            return [
                BasicBlockNode(insts=list(chunk.insts), disable_opt=True)
                if isinstance(chunk, BasicBlockNode) and chunk.disable_opt
                else chunk
                for chunk in chunks
            ], True

    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    fixed_bb = BasicBlockNode(insts=[NopInst()], disable_opt=True)

    with pytest.raises(ValueError, match="disable_opt"):
        _run_passes([ReplacingPass()], [fixed_bb], ctx)


def test_pipeline_tree_pass_missing_disable_opt_block_raises():
    from zcu_tools.program.v2.ir.pipeline import _optimize_tree

    class DroppingTreePass(AbsIRTreePass):
        def transform(self, node: IRNode, ctx: PipeLineContext):
            if isinstance(node, BlockNode):
                return BlockNode()
            return None

    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    root = BlockNode(insts=[BasicBlockNode(insts=[NopInst()], disable_opt=True)])

    with pytest.raises(ValueError, match="disable_opt"):
        _optimize_tree(root, [DroppingTreePass()], ctx)


# ---------------------------------------------------------------------------
# UnrollLoopPass — _unroll_partial big-jump paths (8.2)
# ---------------------------------------------------------------------------


def _nop_loop(name: str, n: int, counter: str = "r0") -> IRLoop:
    """IRLoop with a single NopInst body (size=1, cost=1, scheduled_ticks=0)."""
    return IRLoop(
        name=name,
        counter_reg=Register(counter),
        n=n,
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )


def _ctx_partial(pmem_capacity: int, pmem_budget: int = 10) -> PipeLineContext:
    """Context that forces k_final=pmem_budget (body_size=1) via pmem_budget control.
    slack=-1 < 0 → k_timing=max_unroll_factor=32; k_budget=pmem_budget//1=pmem_budget.
    k_final = min(32, pmem_budget).
    """
    return PipeLineContext(
        config=PipeLineConfig(pmem_capacity=pmem_capacity),
        pmem_budget=pmem_budget,
    )


def _collect_reg_write_s15(root: BlockNode) -> list:
    """Collect all RegWriteInst with dst=s15 from the flat basic-block tree."""
    result = []
    for bb in _walk_basic_blocks(root):
        for inst in bb.insts:
            if isinstance(inst, RegWriteInst) and inst.dst == Register("s15"):
                result.append(inst)
    return result


def test_partial_unroll_suffixes_start_label_when_name_is_already_allocated():
    loop = _nop_loop("loop", n=100)
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=512, pmem_budget=10)
    ctx.allocated_names.add("loop_unrolled_start")

    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)
    label_names = [
        str(label.name) for bb in _walk_basic_blocks(out) for label in bb.labels
    ]

    assert "loop_unrolled_start_0" in label_names
    assert "loop_unrolled_start" not in label_names


def test_partial_unroll_small_pmem_no_remainder_label_jump():
    """Partial unroll (no remainder), small pmem: back-edge uses label-mode JUMP."""
    # n=100, pmem_budget=10 → k=10, remainder=0, small pmem
    loop = _nop_loop("lp", n=100)
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=512, pmem_budget=10)
    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)

    s15_writes = _collect_reg_write_s15(out)
    assert len(s15_writes) == 0  # no s15 in small-pmem mode

    # All back-edge jump BBs have label-mode JumpInst
    back_jumps = [
        bb.branch
        for bb in _walk_basic_blocks(out)
        if isinstance(getattr(bb, "branch", None), JumpInst)
        and bb.branch is not None
        and bb.branch.if_cond == "S"
    ]
    assert len(back_jumps) >= 1
    assert all(j.label is not None for j in back_jumps)


def test_partial_unroll_big_pmem_no_remainder_s15_jump():
    """Partial unroll (no remainder), big pmem: back-edge uses s15 indirect jump."""
    loop = _nop_loop("lp", n=100)
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=4096, pmem_budget=10)
    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)

    s15_writes = _collect_reg_write_s15(out)
    # back-edge writes s15 in big-pmem mode
    assert len(s15_writes) >= 1


def test_partial_unroll_with_remainder_small_pmem():
    """Partial unroll with remainder (n=103, k=10), small pmem: init_bb uses label JUMP."""
    loop = _nop_loop("lp", n=103)  # remainder = 3
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=512, pmem_budget=10)
    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)

    s15_writes = _collect_reg_write_s15(out)
    assert len(s15_writes) == 0  # no s15 in small-pmem mode


def test_partial_unroll_with_remainder_big_pmem():
    """Partial unroll with remainder (n=103, k=10), big pmem: init_bb uses s15 jump."""
    loop = _nop_loop("lp", n=103)  # remainder = 3
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=4096, pmem_budget=10)
    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)

    s15_writes = _collect_reg_write_s15(out)
    # init_bb (for remainder skip) + back-edge both write s15
    assert len(s15_writes) >= 2


# ---------------------------------------------------------------------------
# UnrollLoopPass — _maybe_build_jump_table early-return paths (8.2)
# ---------------------------------------------------------------------------


def _reg_loop(name: str, counter: str = "r0", n_reg: str = "n_reg") -> IRLoop:
    """Register-driven IRLoop (n=Register) with a NopInst body."""
    return IRLoop(
        name=name,
        counter_reg=Register(counter),
        n=Register(n_reg),
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )


def _reg_loop_empty(name: str) -> IRLoop:
    """Register-driven IRLoop with an empty body (body_size=0 → skip jump table)."""
    return IRLoop(
        name=name,
        counter_reg=Register("r0"),
        n=Register("n_reg"),
        body=BlockNode(insts=[]),
    )


def test_jump_table_body_size_zero_returns_none():
    """IRLoop with empty body → body_size=0 → _maybe_build_jump_table returns None (no transform)."""
    loop = _reg_loop_empty("lp")
    root = BlockNode(insts=[loop])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1024,
        available_regs={"r14"},
    )
    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)
    # No IRDispatch should appear — loop was not transformed
    from zcu_tools.program.v2.ir.node import IRDispatch

    dispatches = [n for n in _walk_all_nodes(out) if isinstance(n, IRDispatch)]
    assert len(dispatches) == 0


def test_jump_table_no_available_regs_returns_none():
    """No available_regs → _maybe_build_jump_table returns None (cannot allocate scratch)."""
    loop = _reg_loop("lp")
    root = BlockNode(insts=[loop])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1024,
        available_regs=set(),  # empty
    )
    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)
    from zcu_tools.program.v2.ir.node import IRDispatch

    dispatches = [n for n in _walk_all_nodes(out) if isinstance(n, IRDispatch)]
    assert len(dispatches) == 0


def test_jump_table_k_raw_le_1_returns_none():
    """k_raw <= 1 → skip jump table.  Force by using pmem_budget=1 (k_budget=1//1=1)."""
    loop = _reg_loop("lp")
    root = BlockNode(insts=[loop])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1,  # k_budget = 1//1 = 1 → k_raw=min(max_unroll,1)=1 → skip
        available_regs={"r14"},
    )
    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)
    from zcu_tools.program.v2.ir.node import IRDispatch

    dispatches = [n for n in _walk_all_nodes(out) if isinstance(n, IRDispatch)]
    assert len(dispatches) == 0


def test_jump_table_normal_path_produces_dispatch():
    """Normal register-driven path: k≥2 → BlockNode with IRDispatch inside."""
    loop = _reg_loop("lp")
    root = BlockNode(insts=[loop])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1024,
        available_regs={"r14"},
    )
    out, _ = _apply_tree_pass_to_root(root, UnrollLoopPass(), ctx)
    from zcu_tools.program.v2.ir.node import IRDispatch

    dispatches = [n for n in _walk_all_nodes(out) if isinstance(n, IRDispatch)]
    assert len(dispatches) >= 1


def _walk_all_nodes(node):
    """Walk all IRNodes recursively."""
    yield node
    if isinstance(node, BlockNode):
        for child in node.insts:
            yield from _walk_all_nodes(child)
    elif isinstance(node, IRLoop):
        yield from _walk_all_nodes(node.body)
    elif isinstance(node, IRBranch):
        for case in node.cases:
            yield from _walk_all_nodes(case)


# ---------------------------------------------------------------------------
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
    out = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    # Both writes must survive (disable_opt blocks are left untouched)
    assert len([i for i in bb.insts if isinstance(i, RegWriteInst)]) == 2


# ---------------------------------------------------------------------------
# DmemDispatchPass — small pmem (line 97-103) and big pmem (line 84-95) paths
# ---------------------------------------------------------------------------


def test_dmem_dispatch_k1_returns_none():
    """k == 1: DmemDispatchPass must skip (single target is not a meaningful dispatch)."""
    from zcu_tools.program.v2.ir.node import IRDispatch
    from zcu_tools.program.v2.ir.passes.control_flow import DmemDispatchPass

    labels = [Label("only_case")]
    dispatch = IRDispatch(name="br", value_reg=Register("r2"), target_labels=labels)
    ctx = PipeLineContext(config=PipeLineConfig(pmem_capacity=4096), pmem_budget=1024)
    result = DmemDispatchPass().transform(dispatch, ctx)
    assert result is None


def test_dmem_dispatch_small_pmem_uses_label_jump():
    """pmem_capacity=512 (<=2048): guard uses label-mode JumpInst, not s15-indirect."""
    from zcu_tools.program.v2.ir.node import IRDispatch
    from zcu_tools.program.v2.ir.passes.control_flow import DmemDispatchPass

    labels = [Label(f"case{i}") for i in range(3)]
    dispatch = IRDispatch(
        name="br",
        value_reg=Register("r2"),
        target_labels=labels,
    )
    root = BlockNode(insts=[dispatch])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1024,
    )
    out, _ = _apply_tree_pass_to_root(root, DmemDispatchPass(), ctx)

    all_bbs = list(_walk_basic_blocks(out))
    guard_bbs = [
        bb for bb in all_bbs if bb.branch is not None and bb.branch.if_cond == "NS"
    ]
    assert len(guard_bbs) == 1, "expected exactly one guard block"
    guard_branch = guard_bbs[0].branch
    assert isinstance(guard_branch, JumpInst)
    # label-mode: has a label ref, no addr (no s15 register)
    assert guard_branch.label is not None
    assert guard_branch.addr is None


def test_dmem_dispatch_big_pmem_uses_s15_indirect():
    """pmem_capacity=4096 (>2048): guard uses RegWriteInst(s15=LABEL) + JumpInst(addr=s15)."""
    from zcu_tools.program.v2.ir.node import IRDispatch
    from zcu_tools.program.v2.ir.passes.control_flow import DmemDispatchPass

    labels = [Label(f"case{i}") for i in range(3)]
    dispatch = IRDispatch(
        name="br",
        value_reg=Register("r2"),
        target_labels=labels,
    )
    root = BlockNode(insts=[dispatch])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=4096),
        pmem_budget=1024,
    )
    out, _ = _apply_tree_pass_to_root(root, DmemDispatchPass(), ctx)

    all_bbs = list(_walk_basic_blocks(out))
    guard_bbs = [
        bb for bb in all_bbs if bb.branch is not None and bb.branch.if_cond == "NS"
    ]
    assert len(guard_bbs) == 1, "expected exactly one guard block"
    guard_bb = guard_bbs[0]
    guard_branch = guard_bb.branch
    assert isinstance(guard_branch, JumpInst)
    # big pmem: addr=s15 (indirect), no label ref on the jump
    assert guard_branch.addr == Register("s15")
    assert guard_branch.label is None
    # insts must contain a RegWriteInst that loads LABEL into s15
    write_insts = [i for i in guard_bb.insts if isinstance(i, RegWriteInst)]
    assert any(getattr(i, "src", None) == SrcKeyword.LABEL for i in write_insts), (
        "expected RegWriteInst src=LABEL in guard block"
    )


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
