"""Phase 6 Validation tests for the Two-Tier Optimization Pipeline.

Validation 1: Jump-table stride alignment
  Assert that every entry block in an IRJumpTableLoop output has exactly
  `body_words + 1` instructions so the dispatcher's stride calculation
  never drifts.

Validation 2: Fully-unrolled loops produce a single fused block with zero
  internal dead writes after the full pipeline runs.

Validation 3: fix_addr_size=True blocks receive NopInst padding instead of
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
    """Collect BasicBlockNodes that are fix_addr_size=True AND have labels."""
    result: list[BasicBlockNode] = []
    stack: list[IRNode] = list(root.insts)
    while stack:
        node = stack.pop()
        if isinstance(node, BasicBlockNode):
            if node.fix_addr_size and node.labels:
                result.append(node)
        elif isinstance(node, BlockNode):
            stack.extend(node.insts)
    return result


def _collect_entry_groups(root: RootNode) -> list[list[BasicBlockNode]]:
    """Group fix_addr_size=True blocks into per-entry groups.

    A new group starts at each fix_addr_size block that has labels.
    Groups end (and are not continued) when a fix_addr_size block that has
    no labels AND has a branch is encountered — that signals the back-edge,
    which is not part of any body entry.
    """
    groups: list[list[BasicBlockNode]] = []
    current: list[BasicBlockNode] = []
    for node in root.insts:
        if not isinstance(node, BasicBlockNode) or not node.fix_addr_size:
            continue
        # Back-edge blocks: no labels, has branch → end of entries.
        if not node.labels and node.branch is not None:
            if current:
                groups.append(current)
                current = []
            break
        # New entry starts when a labelled block is encountered.
        if node.labels and current:
            groups.append(current)
            current = []
        current.append(node)
    if current:
        groups.append(current)
    return groups


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

    root = RootNode(insts=list(blocks))
    groups = _collect_entry_groups(root)
    assert len(groups) == k, f"expected {k} entry groups, got {len(groups)}"

    expected_addr = body_words + 1  # body addr words + counter increment
    for i, group in enumerate(groups):
        actual = _entry_addr_size(group)
        assert actual == expected_addr, (
            f"entry {i}: expected addr size {expected_addr}, got {actual}"
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

    groups = _collect_entry_groups(out)
    assert len(groups) > 0, "no entry groups produced"

    expected_addr = body_nops + 1
    for i, group in enumerate(groups):
        actual = _entry_addr_size(group)
        assert actual == expected_addr, (
            f"entry {i}: expected addr size {expected_addr} (body_words={body_nops}+1), "
            f"got {actual}"
        )


def test_v1_jump_table_entry_blocks_not_modified_by_pipeline():
    """The full pipeline must not alter the instruction count of entry blocks
    (fix_addr_size=True guards them from Post-LIR peephole passes).

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

    groups = _collect_entry_groups(out)
    assert len(groups) > 0, "no entry groups produced"

    # body_words = 2 (NOP + NOP), counter increment = 1 → addr size 3 per entry
    expected_addr = 3
    for i, group in enumerate(groups):
        actual = _entry_addr_size(group)
        assert actual == expected_addr, (
            f"entry {i}: pipeline altered addr size "
            f"(expected {expected_addr}, got {actual})"
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
    plain_blocks = [b for b in bbs if not b.fix_addr_size]
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
# Validation 3: fix_addr_size=True blocks use NOP padding, not branch deletion
# ---------------------------------------------------------------------------

def test_v3_fixed_block_branch_elim_produces_nop():
    """When a fix_addr_size=True block's unconditional branch targets the
    immediately following block, BranchEliminationPass must replace the
    branch with a NopInst (not remove it).
    """
    lbl = Label.make_new("next")
    root = RootNode(insts=[
        BasicBlockNode(
            insts=[NopInst()],
            branch=JumpInst(label=lbl),
            fix_addr_size=True,
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
        "fix_addr_size=True block must receive NOP padding when branch is eliminated"
    )
    # Total instruction count must not shrink (stride preserved).
    assert len(fixed.insts) == 2, (
        f"expected 2 insts (original NOP + padding NOP), got {len(fixed.insts)}"
    )


def test_v3_fixed_block_instruction_count_preserved_after_pipeline():
    """After the full pipeline, no fix_addr_size=True entry block should have
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

    groups = _collect_entry_groups(out)
    assert len(groups) > 0, "no entry groups produced"

    for i, group in enumerate(groups):
        # addr size must be exactly body_words(2) + 1 = 3.
        actual = _entry_addr_size(group)
        assert actual == 3, (
            f"entry {i} has addr size {actual} — "
            f"fix_addr_size was not respected by the pipeline"
        )


def test_v3_non_fixed_block_branch_elim_removes_branch():
    """Sanity check: fix_addr_size=False block loses its branch (no NOP added)."""
    lbl = Label.make_new("next_free")
    root = RootNode(insts=[
        BasicBlockNode(
            insts=[NopInst()],
            branch=JumpInst(label=lbl),
            fix_addr_size=False,
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
