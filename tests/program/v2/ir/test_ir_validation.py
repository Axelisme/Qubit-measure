"""Phase 6 Validation tests for the Two-Tier Optimization Pipeline.

Validation 1: Dispatch-table island invariants
  Assert that only dispatch-table stubs are fixed-width, while body copies
  remain free-form.

Validation 2: Fully-unrolled loops produce a single fused block with zero
  internal dead writes after the full pipeline runs.

Validation 3: disable_opt=True blocks are conservatively skipped by passes.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from zcu_tools.program.v2.ir.factory import IRLexer, IRParser
from zcu_tools.program.v2.ir.instructions import (
    DmemReadInst,
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRDispatch,
    IRLoop,
    IRNode,
)
from zcu_tools.program.v2.ir.operands import Immediate, Register, SrcKeyword
from zcu_tools.program.v2.ir.passes import BranchEliminationPass
from zcu_tools.program.v2.ir.passes.control_flow import SimplifyDispatchPass
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    make_default_pipeline,
)


def _walk_instructions(node: IRNode) -> Iterator[Instruction]:
    if isinstance(node, BasicBlockNode):
        yield from node.labels
        yield from node.insts
        if node.branch is not None:
            yield node.branch
    elif isinstance(node, BlockNode):
        for child in node.insts:
            yield from _walk_instructions(child)
    elif isinstance(node, IRLoop):
        yield from _walk_instructions(node.body)
    elif isinstance(node, IRBranch):
        for case in node.cases:
            yield from _walk_instructions(case)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_chunk_passes_on_root(root: BlockNode, passes: list) -> BlockNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    for pass_ in passes:
        chunks, _ = pass_.process(chunks, ctx)
    return parser.parse(chunks)


def _collect_fixed_entry_blocks(root: BlockNode) -> list[BasicBlockNode]:
    """Collect BasicBlockNodes that are disable_opt=True AND have labels."""
    result: list[BasicBlockNode] = []
    stack: list[IRNode] = list(root.insts)
    while stack:
        node = stack.pop()
        if isinstance(node, BasicBlockNode):
            if node.disable_opt and node.labels:
                result.append(node)
        elif isinstance(node, BlockNode):
            stack.extend(node.insts)
    return result


def _collect_all_basic_blocks(root: BlockNode) -> list[BasicBlockNode]:
    result: list[BasicBlockNode] = []
    stack: list[IRNode] = list(root.insts)
    while stack:
        node = stack.pop()
        if isinstance(node, BasicBlockNode):
            result.append(node)
        elif isinstance(node, BlockNode):
            stack.extend(node.insts)
    return result


def _run_full_pipeline_on_root(
    root: BlockNode, *, pmem: int = 512, max_unroll_factor: int | None = None
) -> BlockNode:
    out, _ctx = _run_full_pipeline_capture_ctx(
        root, pmem=pmem, max_unroll_factor=max_unroll_factor
    )
    return out


def _run_full_pipeline_capture_ctx(
    root: BlockNode, *, pmem: int = 512, max_unroll_factor: int | None = None
):
    """Run the full pipeline; return (reconstructed BlockNode, PipeLineContext)."""
    pipeline = make_default_pipeline(
        pmem_capacity=pmem, max_unroll_factor=max_unroll_factor
    )
    lexer = IRLexer()
    parser = IRParser(pmem_size=pmem)
    insts = lexer.flatten(parser.unparse(root))
    out_insts, ctx = pipeline(insts)
    return parser.parse(lexer.lex(out_insts)), ctx


# ---------------------------------------------------------------------------
# Validation 1: Jump-table stride alignment
# ---------------------------------------------------------------------------


def test_v1_dispatch_uses_dmem_table_no_pmem_stubs():
    """k>=3 dispatch is lowered by DmemDispatchPass to a dmem address table.

    No fixed-width pmem stub island is produced — the disable_opt invariant is
    gone; case bodies stay free. (Phase 7: dmem dispatch replaces the pmem
    table island for k>=3.)
    """
    k = 4
    body_words = 3
    root = BlockNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg=Register("r0"),
                n=Register("r1"),
                body=BlockNode(
                    insts=[BasicBlockNode(insts=[NopInst()]) for _ in range(body_words)]
                ),
            )
        ]
    )

    out, ctx = _run_full_pipeline_capture_ctx(root, pmem=512, max_unroll_factor=k)

    # No pmem dispatch-table stubs survive: dmem dispatch has no disable_opt blocks.
    assert _collect_fixed_entry_blocks(out) == []

    # One dmem dispatch table with k entries was reserved.
    assert len(ctx.dmem_tables) == 1
    assert len(ctx.dmem_tables[0]) == k

    # The dmem dispatch instruction pattern is present: a dmem read into s15
    # followed by an indirect JUMP [s15].
    blocks = _collect_all_basic_blocks(out)
    has_dmem_dispatch = any(
        any(
            isinstance(inst, DmemReadInst) and inst.dst == Register("s15")
            for inst in bb.insts
        )
        for bb in blocks
    )
    assert has_dmem_dispatch

    # The k unrolled body entries stay free (not disable_opt).
    plain_entry_blocks = [
        block
        for block in blocks
        if any(lbl.name.name.startswith("loop_jt_entry_") for lbl in block.labels)
    ]
    assert plain_entry_blocks
    assert all(not block.disable_opt for block in plain_entry_blocks)


def test_v1_dmem_dispatch_tables_deduplicated():
    """The resolve step dedupes DmemAddr by table_labels value.

    Two dispatch instructions referencing identical entry-label tuples must
    share a single dmem table; differing tuples get distinct tables.
    """
    from zcu_tools.program.v2.ir.operands import AluExpr, AluOp, DmemAddr, Immediate
    from zcu_tools.program.v2.ir.pipeline import (
        PipeLineConfig,
        PipeLineContext,
        _resolve_dmem_dispatch,
    )

    labels_a = tuple(Label(f"a{i}") for i in range(4))
    labels_b = tuple(Label(f"b{i}") for i in range(3))

    def _dispatch_block(table: tuple[Label, ...]) -> BasicBlockNode:
        return BasicBlockNode(
            insts=[
                RegWriteInst(
                    dst=Register("s15"),
                    src=SrcKeyword.OP,
                    op=AluExpr(Register("r0"), AluOp.ADD, DmemAddr(table)),
                )
            ]
        )

    # Two blocks reference labels_a (must dedup), one references labels_b.
    chunks = [
        _dispatch_block(labels_a),
        _dispatch_block(labels_b),
        _dispatch_block(labels_a),
    ]
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024, dmem_base_offset=7)
    _resolve_dmem_dispatch(chunks, ctx)

    # labels_a + labels_b → exactly 2 tables (labels_a shared by 2 blocks).
    assert len(ctx.dmem_tables) == 2
    assert [lbl.name for lbl in ctx.dmem_tables[0]] == [lbl.name for lbl in labels_a]
    assert [lbl.name for lbl in ctx.dmem_tables[1]] == [lbl.name for lbl in labels_b]

    # The two labels_a blocks resolved to the same base; allocation starts at
    # dmem_base_offset (7); labels_b follows after labels_a's 4 words.
    bases: list[int] = []
    for b in chunks:
        inst = b.insts[0]
        assert isinstance(inst, RegWriteInst)
        op = inst.op
        assert isinstance(op, AluExpr) and isinstance(op.rhs, Immediate)
        bases.append(op.rhs.value)
    assert bases == [7, 11, 7]  # a@7, b@11, a@7 (shared)


def test_v1_pipeline_keeps_body_blocks_free_after_unroll():
    """After the full pipeline, body entry blocks must remain non-fixed."""
    root = BlockNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg=Register("r0"),
                n=Register("r1"),
                body=BlockNode(
                    insts=[
                        BasicBlockNode(insts=[NopInst()]),
                        BasicBlockNode(insts=[NopInst()]),
                    ]
                ),
            )
        ]
    )

    out = _run_full_pipeline_on_root(root)

    body_entry_blocks = [
        block
        for block in _collect_all_basic_blocks(out)
        if any(
            lbl.name.name.startswith("loop_jt_entry_")
            and "_dispatch_" not in lbl.name.name
            for lbl in block.labels
        )
    ]
    assert body_entry_blocks
    assert all(not block.disable_opt for block in body_entry_blocks)


def test_branch_parse_rejects_missing_case_end():
    items = [
        MetaInst(type="BRANCH_START", name="sel", info={"compare_reg": "r3"}),
        MetaInst(type="BRANCH_CASE_START", name="0"),
        BasicBlockNode(insts=[NopInst()]),
        MetaInst(type="BRANCH_END", name="sel"),
    ]

    with pytest.raises(ValueError, match="BRANCH_CASE_END"):
        IRParser().parse(items)


def test_branch_parse_rejects_branch_without_cases():
    items = [
        MetaInst(type="BRANCH_START", name="sel", info={"compare_reg": "r3"}),
        BasicBlockNode(insts=[NopInst()]),
        MetaInst(type="BRANCH_END", name="sel"),
    ]

    with pytest.raises(ValueError, match="does not contain any cases"):
        IRParser().parse(items)


def test_sese_rejects_jump_into_loop_control_region():
    loop_start = Label("loop_start")
    loop_end = Label("loop_end")
    outside = Label("outside")

    items = [
        BasicBlockNode(
            labels=[LabelInst(name=outside)],
            branch=JumpInst(label=LabelRef(loop_end)),
        ),
        MetaInst(
            type="LOOP_START",
            name="L",
            info={"counter_reg": "r0", "n": 2},
        ),
        BasicBlockNode(labels=[LabelInst(name=loop_start, can_remove=True)]),
        MetaInst(type="LOOP_BODY_START", name="L"),
        BasicBlockNode(insts=[NopInst()]),
        MetaInst(type="LOOP_BODY_END", name="L"),
        BasicBlockNode(labels=[LabelInst(name=loop_end, can_remove=True)]),
        MetaInst(type="LOOP_END", name="L"),
    ]

    with pytest.raises(ValueError, match="violates SESE assumption"):
        IRParser().parse(items)


# ---------------------------------------------------------------------------
# Validation 2: Fully-unrolled loops produce a single fused block
# ---------------------------------------------------------------------------


def test_v2_fully_unrolled_loop_produces_single_fused_block():
    """A constant loop that fully unrolls (n <= k) must result in a single
    BasicBlockNode with no internal labels after the full pipeline runs.

    body: REG_WR r1 imm #1 (dead write — overwritten each iteration)
    After 3 full unroll copies the pipeline should merge them and eliminate
    the two dead writes, leaving a single write.
    """
    root = BlockNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg=Register("r0"),
                n=3,
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

    out = _run_full_pipeline_on_root(root)

    bbs = _collect_all_basic_blocks(out)
    # After merge, all non-fixed BasicBlockNodes with plain insts should have
    # been fused into a minimal set. Specifically, no two adjacent non-fixed
    # blocks should be mergeable (i.e., one without branch followed by one
    # without alive labels).
    plain_blocks = [b for b in bbs if not b.disable_opt]
    for i in range(len(plain_blocks) - 1):
        a, b = plain_blocks[i], plain_blocks[i + 1]
        if a.branch is None and not b.labels:
            raise AssertionError(
                f"Two adjacent plain blocks are still mergeable after pipeline: "
                f"block[{i}]={a}, block[{i + 1}]={b}"
            )


def test_v2_fully_unrolled_dead_writes_eliminated_across_boundaries():
    """n=3 loop with body writing r2=imm #42 each iteration.

    After full unroll + DeadWriteElimination only the last write should
    survive (the first two are dead: overwritten before being read).
    Uses _walk_instructions to cover all BasicBlockNode paths.
    """
    root = BlockNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg=Register("r0"),
                n=3,
                body=BlockNode(
                    insts=[
                        BasicBlockNode(
                            insts=[
                                RegWriteInst(
                                    dst=Register("r2"),
                                    src=SrcKeyword.IMM,
                                    lit=Immediate(42),
                                )
                            ]
                        ),
                    ]
                ),
            )
        ]
    )

    out = _run_full_pipeline_on_root(root)

    writes_to_r_out = [
        inst
        for inst in _walk_instructions(out)
        if isinstance(inst, RegWriteInst) and inst.dst.name == "r2"
    ]
    assert len(writes_to_r_out) == 1, (
        f"expected 1 surviving write to r2, got {len(writes_to_r_out)}"
    )


# ---------------------------------------------------------------------------
# Validation 3: disable_opt=True blocks are skipped conservatively
# ---------------------------------------------------------------------------


def test_v3_fixed_block_branch_elim_skips_block():
    """BranchEliminationPass should leave fixed-width stubs unchanged."""
    lbl = Label("next")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=LabelRef(lbl)),
                disable_opt=True,
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )

    out = _run_chunk_passes_on_root(root, [BranchEliminationPass()])

    fixed = out.insts[0]
    assert isinstance(fixed, BasicBlockNode)
    assert fixed.branch is not None
    assert len(fixed.insts) == 1


def test_v3_non_fixed_block_branch_elim_removes_branch():
    """Sanity check: disable_opt=False block loses its branch (no NOP added)."""
    lbl = Label("next_free")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=LabelRef(lbl)),
                disable_opt=False,
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )

    out = _run_chunk_passes_on_root(root, [BranchEliminationPass()])

    free = out.insts[0]
    assert isinstance(free, BasicBlockNode)
    assert free.branch is None
    # No NOP should have been injected.
    assert len(free.insts) == 1
    assert isinstance(free.insts[0], NopInst)


# ---------------------------------------------------------------------------
# Validation 4: SimplifyDispatchPass collapses 2-target IRDispatch
# ---------------------------------------------------------------------------


def _apply_simplify_dispatch(node: IRNode, pmem: int = 512):
    ctx = PipeLineContext(config=PipeLineConfig(pmem_capacity=pmem), pmem_budget=1024)
    return SimplifyDispatchPass().transform(node, ctx)


def test_v4_simplify_dispatch_k2_produces_single_cond_jump():
    """2-target IRDispatch must be replaced by a single BasicBlockNode with a
    conditional jump to target_labels[1]; no dispatch table stubs produced."""
    t0 = Label("entry_0")
    t1 = Label("entry_1")
    node = IRDispatch(name="d", value_reg=Register("r1"), target_labels=[t0, t1])

    result = _apply_simplify_dispatch(node, pmem=512)

    assert isinstance(result, BasicBlockNode)
    assert result.branch is not None
    assert isinstance(result.branch, JumpInst)
    assert result.branch.if_cond == "NZ"
    assert result.branch.label == LabelRef(t1)


def test_v4_simplify_dispatch_k2_big_pmem_uses_indirect_jump():
    """big-PMEM (pmem=4096): 2-target IRDispatch emits REG_WR s15 + indirect JUMP."""
    t0 = Label("entry_0")
    t1 = Label("entry_1")
    node = IRDispatch(name="d", value_reg=Register("r1"), target_labels=[t0, t1])

    result = _apply_simplify_dispatch(node, pmem=4096)

    assert isinstance(result, BasicBlockNode)
    assert len(result.insts) == 1
    wr = result.insts[0]
    assert isinstance(wr, RegWriteInst)
    assert wr.dst.name == "s15"
    assert wr.label == LabelRef(t1)
    assert result.branch is not None
    assert isinstance(result.branch, JumpInst)
    assert result.branch.if_cond == "NZ"
    assert result.branch.addr is not None


def test_v4_simplify_dispatch_k_gt2_unchanged():
    """IRDispatch with more than 2 targets must be left unchanged.

    Under the AbsIRTreePass contract, "unchanged" is signalled by returning
    None (not the same object).
    """
    targets = [Label(f"entry_{i}") for i in range(4)]
    node = IRDispatch(name="d", value_reg=Register("r1"), target_labels=targets)

    result = _apply_simplify_dispatch(node)

    assert result is None
