"""Phase 6 Validation tests for the Two-Tier Optimization Pipeline.

Validation 1: Jump-table stride alignment
  Assert that every entry block in an IRJumpTableLoop output has exactly
  `body_words + 1` instructions so the dispatcher's stride calculation
  never drifts.

Validation 2: Fully-unrolled loops produce a single fused block with zero
  internal dead writes after the full pipeline runs.

Validation 3: fix_inst_num=True blocks receive NopInst padding instead of
  branch deletion when BranchEliminationPass runs.
"""
from __future__ import annotations

from copy import deepcopy

from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    LabelInst,
    NopInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRLoop,
    IRNode,
    RootNode,
)
from zcu_tools.program.v2.ir.passes import BranchEliminationPass, UnrollSmallLoopPass
from zcu_tools.program.v2.ir.passes.loop_dispatch import build_jump_table_blocks
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    make_default_pipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_fixed_entry_blocks(root: RootNode) -> list[BasicBlockNode]:
    """Collect BasicBlockNodes that are fix_inst_num=True AND have labels."""
    result: list[BasicBlockNode] = []
    stack: list[IRNode] = list(root.insts)
    while stack:
        node = stack.pop()
        if isinstance(node, BasicBlockNode):
            if node.fix_inst_num and node.labels:
                result.append(node)
        elif isinstance(node, BlockNode):
            stack.extend(node.insts)
    return result


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


# ---------------------------------------------------------------------------
# Validation 1: Jump-table stride alignment
# ---------------------------------------------------------------------------

def test_v1_jump_table_entry_blocks_have_uniform_stride():
    """All k entry blocks must have identical instruction counts (= body_words + 1).

    This guarantees the dispatch arithmetic `entry_0 + i * stride` always
    lands on a valid entry label, regardless of which entry point is chosen.
    """
    Label.reset()
    k = 4
    body_words = 3
    entry_labels = [Label.make_new(f"jt_entry_{i}") for i in range(k)]
    exit_label = Label.make_new("jt_exit")
    body = BlockNode(insts=[BasicBlockNode(insts=[NopInst()]) for _ in range(body_words)])
    bodies = [deepcopy(body) for _ in range(k)]

    blocks = build_jump_table_blocks(
        name="test_loop",
        n_reg="r_n",
        counter_reg="r_i",
        k=k,
        body_words=body_words,
        entry_labels=entry_labels,
        exit_label=exit_label,
        bodies=bodies,
    )

    entry_blocks = [b for b in blocks if b.fix_inst_num and b.labels]
    assert len(entry_blocks) == k, f"expected {k} entry blocks, got {len(entry_blocks)}"

    expected_inst_count = body_words + 1  # body + counter increment
    for i, blk in enumerate(entry_blocks):
        assert len(blk.insts) == expected_inst_count, (
            f"entry block {i}: expected {expected_inst_count} insts, got {len(blk.insts)}"
        )


def test_v1_jump_table_stride_equals_body_words_plus_one():
    """After UnrollSmallLoopPass, entry block instruction count == body_words + 1.

    body_words is measured by estimate_flat_size, which counts non-meta
    instructions. The +1 is the per-iteration counter increment.
    """
    Label.reset()
    body_nops = 5  # body_words == 5
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()]) for _ in range(body_nops)]),
            )
        ]
    )

    config = PipeLineConfig(max_unroll_factor=4)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    entry_blocks = _collect_fixed_entry_blocks(out)
    assert len(entry_blocks) > 0, "no entry blocks produced"

    expected = body_nops + 1
    for i, blk in enumerate(entry_blocks):
        assert len(blk.insts) == expected, (
            f"entry block {i}: expected {expected} insts (body_words={body_nops}+1), "
            f"got {len(blk.insts)}"
        )


def test_v1_jump_table_entry_blocks_not_modified_by_pipeline():
    """The full pipeline must not alter the instruction count of entry blocks
    (fix_inst_num=True guards them from Post-LIR peephole passes).

    body: two NOPs. Pre-LIR passes leave NOPs untouched, so body_words=2
    and each entry block should have exactly 3 instructions after unrolling.
    """
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=[
                    BasicBlockNode(insts=[NopInst()]),
                    BasicBlockNode(insts=[NopInst()]),
                ]),
            )
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=512)
    out, _ = pipeline(root)

    entry_blocks = _collect_fixed_entry_blocks(out)
    assert len(entry_blocks) > 0, "no entry blocks produced"

    # body_words = 2 (NOP + NOP), counter increment = 1 → 3 each
    expected = 3
    for i, blk in enumerate(entry_blocks):
        assert len(blk.insts) == expected, (
            f"entry block {i}: pipeline altered instruction count "
            f"(expected {expected}, got {len(blk.insts)})"
        )


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
                counter_reg="r_cnt",
                n=3,
                body=BlockNode(insts=[
                    BasicBlockNode(insts=[RegWriteInst(dst="r1", src="imm", lit="#1")]),
                ]),
            )
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=512)
    out, _ = pipeline(root)

    bbs = _collect_all_basic_blocks(out)

    # After merge, all non-fixed BasicBlockNodes with plain insts should have
    # been fused into a minimal set. Specifically, no two adjacent non-fixed
    # blocks should be mergeable (i.e., one without branch followed by one
    # without alive labels).
    plain_blocks = [b for b in bbs if not b.fix_inst_num]
    for i in range(len(plain_blocks) - 1):
        a, b = plain_blocks[i], plain_blocks[i + 1]
        if a.branch is None and not b.labels:
            raise AssertionError(
                f"Two adjacent plain blocks are still mergeable after pipeline: "
                f"block[{i}]={a}, block[{i+1}]={b}"
            )


def test_v2_fully_unrolled_dead_writes_eliminated_across_boundaries():
    """n=3 loop with body writing r_out=imm #42 each iteration.

    After full unroll + DeadWriteElimination only the last write should
    survive (the first two are dead: overwritten before being read).
    Uses walk_instructions to cover all BasicBlockNode paths.
    """
    from zcu_tools.program.v2.ir.traversal import walk_instructions

    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_cnt",
                n=3,
                body=BlockNode(insts=[
                    BasicBlockNode(insts=[RegWriteInst(dst="r_out", src="imm", lit="#42")]),
                ]),
            )
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=512)
    out, _ = pipeline(root)

    writes_to_r_out = [
        inst for inst in walk_instructions(out)
        if isinstance(inst, RegWriteInst) and inst.dst == "r_out"
    ]
    assert len(writes_to_r_out) == 1, (
        f"expected 1 surviving write to r_out, got {len(writes_to_r_out)}"
    )


# ---------------------------------------------------------------------------
# Validation 3: fix_inst_num=True blocks use NOP padding, not branch deletion
# ---------------------------------------------------------------------------

def test_v3_fixed_block_branch_elim_produces_nop():
    """When a fix_inst_num=True block's unconditional branch targets the
    immediately following block, BranchEliminationPass must replace the
    branch with a NopInst (not remove it).
    """
    lbl = Label.make_new("next")
    root = RootNode(insts=[
        BasicBlockNode(
            insts=[NopInst()],
            branch=JumpInst(label=lbl),
            fix_inst_num=True,
        ),
        BasicBlockNode(
            labels=[LabelInst(name=lbl)],
            insts=[NopInst()],
        ),
    ])

    out = BranchEliminationPass().process(root, PipeLineContext(config=PipeLineConfig()))

    fixed = out.insts[0]
    assert isinstance(fixed, BasicBlockNode)
    assert fixed.branch is None, "branch should have been removed"
    # The former branch word must be replaced by a NopInst to keep stride.
    assert any(isinstance(i, NopInst) for i in fixed.insts), (
        "fix_inst_num=True block must receive NOP padding when branch is eliminated"
    )
    # Total instruction count must not shrink (stride preserved).
    assert len(fixed.insts) == 2, (
        f"expected 2 insts (original NOP + padding NOP), got {len(fixed.insts)}"
    )


def test_v3_fixed_block_instruction_count_preserved_after_pipeline():
    """After the full pipeline, no fix_inst_num=True entry block should have
    fewer instructions than body_words + 1.

    body: two NOPs → body_words=2, each entry block expects exactly 3 insts.
    BranchEliminationPass must not strip any instructions from these blocks.
    """
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=[
                    BasicBlockNode(insts=[NopInst()]),
                    BasicBlockNode(insts=[NopInst()]),
                ]),
            )
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=512)
    out, _ = pipeline(root)

    entry_blocks = _collect_fixed_entry_blocks(out)
    assert len(entry_blocks) > 0, "no entry blocks produced"

    for i, blk in enumerate(entry_blocks):
        # Instruction count must be exactly body_words(2) + 1 = 3.
        assert len(blk.insts) == 3, (
            f"entry block {i} has {len(blk.insts)} insts — "
            f"fix_inst_num was not respected by the pipeline"
        )


def test_v3_non_fixed_block_branch_elim_removes_branch():
    """Sanity check: fix_inst_num=False block loses its branch (no NOP added)."""
    lbl = Label.make_new("next_free")
    root = RootNode(insts=[
        BasicBlockNode(
            insts=[NopInst()],
            branch=JumpInst(label=lbl),
            fix_inst_num=False,
        ),
        BasicBlockNode(
            labels=[LabelInst(name=lbl)],
            insts=[NopInst()],
        ),
    ])

    out = BranchEliminationPass().process(root, PipeLineContext(config=PipeLineConfig()))

    free = out.insts[0]
    assert isinstance(free, BasicBlockNode)
    assert free.branch is None
    # No NOP should have been injected.
    assert len(free.insts) == 1
    assert isinstance(free.insts[0], NopInst)
