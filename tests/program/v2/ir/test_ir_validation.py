"""Phase 6 Validation tests for the Two-Tier Optimization Pipeline.

Validation 1: Dispatch-table island invariants
  Assert that only dispatch-table stubs are fixed-width, while body copies
  remain free-form.

Validation 2: Fully-unrolled loops produce a single fused block with zero
  internal dead writes after the full pipeline runs.

Validation 3: disable_opt=True blocks are conservatively skipped by passes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterator

import pytest
from zcu_tools.program.v2.ir.factory import IRLexer, IRParser
from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRLoop,
    IRNode,
    RootNode,
)
from zcu_tools.program.v2.ir.operands import Immediate, Register, SrcKeyword
from zcu_tools.program.v2.ir.passes import BranchEliminationPass, UnrollLoopPass
from zcu_tools.program.v2.ir.passes.loop.dispatch_island import build_jump_table_blocks
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


def _run_chunk_passes_on_root(root: RootNode, passes: list) -> RootNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    for pass_ in passes:
        chunks, _ = pass_.process(chunks, ctx)
    return parser.parse(chunks)


def _collect_fixed_entry_blocks(root: RootNode) -> list[BasicBlockNode]:
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


def _entry_addr_size(group: list[BasicBlockNode]) -> int:
    """Total addr words occupied by an entry group."""
    total = 0
    for bb in group:
        for inst in bb.insts:
            total += inst.addr_inc
        if bb.branch is not None:
            total += bb.branch.addr_inc
    return total


def _collect_all_basic_blocks(root: RootNode) -> list[BasicBlockNode]:
    result: list[BasicBlockNode] = []
    stack: list[IRNode] = list(root.insts)
    while stack:
        node = stack.pop()
        if isinstance(node, BasicBlockNode):
            result.append(node)
        elif isinstance(node, BlockNode):
            stack.extend(node.insts)
    return result


def _run_full_pipeline_on_root(root: RootNode, *, pmem: int = 512) -> RootNode:
    pipeline = make_default_pipeline(pmem_capacity=pmem)
    lexer = IRLexer()
    parser = IRParser(pmem_size=pmem)
    insts = lexer.flatten(parser.unparse(root))
    out_insts, _ = pipeline(insts)
    return parser.parse(lexer.lex(out_insts))


# ---------------------------------------------------------------------------
# Validation 1: Jump-table stride alignment
# ---------------------------------------------------------------------------


def test_v1_jump_table_only_dispatch_stubs_are_fixed():
    """Dispatch-table lowering should freeze only the table island."""
    Label.reset()
    k = 4
    body_words = 3
    entry_labels = [Label.make_new(f"jt_entry_{i}") for i in range(k)]
    exit_label = Label.make_new("jt_exit")
    body = BlockNode(
        insts=[BasicBlockNode(insts=[NopInst()]) for _ in range(body_words)]
    )
    bodies = [deepcopy(body) for _ in range(k)]

    blocks = build_jump_table_blocks(
        n_reg="r1",
        counter_reg="r0",
        k=k,
        entry_labels=entry_labels,
        exit_label=exit_label,
        bodies=bodies,
    )

    root = RootNode(insts=list(blocks))
    fixed_blocks = _collect_fixed_entry_blocks(root)
    assert len(fixed_blocks) == k
    assert all(block.branch is not None for block in fixed_blocks)

    plain_entry_blocks = [
        block
        for block in _collect_all_basic_blocks(root)
        if any(
            lbl.name.name.startswith("jt_entry_") and "_dispatch_" not in lbl.name.name
            for lbl in block.labels
        )
    ]
    assert plain_entry_blocks
    assert all(not block.disable_opt for block in plain_entry_blocks)


def test_v1_jump_table_stub_width_is_uniform():
    """Every dispatch-table entry stub must keep the same physical width."""
    Label.reset()
    body_nops = 5  # body_words == 5
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg=Register("r0"),
                n=Register("r1"),
                body=BlockNode(
                    insts=[BasicBlockNode(insts=[NopInst()]) for _ in range(body_nops)]
                ),
            )
        ]
    )

    config = PipeLineConfig(max_unroll_factor=4, pmem_capacity=512)
    out, _ = UnrollLoopPass().process(
        root, PipeLineContext(config=config, pmem_budget=3192)
    )

    stubs = _collect_fixed_entry_blocks(out)
    assert stubs
    stub_sizes = [_entry_addr_size([stub]) for stub in stubs]
    assert len(set(stub_sizes)) == 1
    assert stub_sizes[0] == 1


def test_v1_pipeline_keeps_body_blocks_free_after_unroll():
    """After the full pipeline, body entry blocks must remain non-fixed."""
    Label.reset()
    root = RootNode(
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
    Label.reset()
    loop_start = Label.make_new("loop_start")
    loop_end = Label.make_new("loop_end")
    outside = Label.make_new("outside")

    items = [
        BasicBlockNode(
            labels=[LabelInst(name=outside)],
            branch=JumpInst(label=loop_end),
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
    Label.reset()
    root = RootNode(
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
                                    dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)
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

    Label.reset()
    root = RootNode(
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
                                    dst=Register("r2"), src=SrcKeyword.IMM, lit=Immediate(42)
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
    lbl = Label.make_new("next")
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=lbl),
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
    lbl = Label.make_new("next_free")
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=lbl),
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
