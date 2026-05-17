"""Tests for Post-LIR control-flow passes:
- DeadLabelEliminationPass (BasicBlockNode path)
- BranchEliminationPass
- BlockMergePass
"""

from __future__ import annotations

from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    LabelInst,
    NopInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRLoop,
)
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    Register,
    SrcKeyword,
)
from zcu_tools.program.v2.ir.passes.control_flow import (
    BlockMergePass,
    BranchEliminationPass,
    DeadLabelEliminationPass,
)
from zcu_tools.program.v2.ir.passes.dataflow import DeadWriteEliminationPass
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
)


def _label(name: str) -> Label:
    return Label(name)


def _lref(name: str) -> LabelRef:
    return LabelRef(Label(name))


def _run_chunk_passes_on_root(root: BlockNode, passes: list) -> BlockNode:
    parser = IRParser()
    chunks = parser.unparse(root)
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    for pass_ in passes:
        chunks, _ = pass_.process(chunks, ctx)
    return parser.parse(chunks)


# ---------------------------------------------------------------------------
# DeadLabelEliminationPass — BasicBlockNode path
# ---------------------------------------------------------------------------


def test_dead_label_elim_removes_unreferenced_label_from_basic_block():
    lbl = _label("unused")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                labels=[LabelInst(name=lbl, can_remove=True)], insts=[NopInst()]
            ),
        ]
    )

    out = _run_chunk_passes_on_root(root, [DeadLabelEliminationPass()])

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert bb.labels == []
    assert len(bb.insts) == 1


def test_dead_label_elim_keeps_referenced_label_in_basic_block():
    lbl = _label("used_lbl")
    root = BlockNode(
        insts=[
            BasicBlockNode(labels=[LabelInst(name=lbl)], insts=[NopInst()]),
            BasicBlockNode(branch=JumpInst(label=LabelRef(lbl))),
        ]
    )

    out = _run_chunk_passes_on_root(root, [DeadLabelEliminationPass()])

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.labels) == 1
    assert bb.labels[0].name == lbl


# ---------------------------------------------------------------------------
# BranchEliminationPass
# ---------------------------------------------------------------------------


def test_branch_elim_removes_unconditional_fallthrough():
    lbl = _label("next")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=LabelRef(lbl)),
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )

    out = _run_chunk_passes_on_root(root, [BranchEliminationPass()])

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is None
    assert len(a.insts) == 1  # NopInst still present


def test_branch_elim_skips_fixed_block():
    lbl = _label("next_fixed")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[],
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

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is not None
    assert len(a.insts) == 0


def test_branch_elim_keeps_conditional_branch():
    lbl = _label("cond_target")
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(
                    label=LabelRef(lbl),
                    if_cond="Z",
                    op=AluExpr(Register("r0"), AluOp.SUB, Immediate(0)),
                ),
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )

    out = _run_chunk_passes_on_root(root, [BranchEliminationPass()])

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is not None


def test_branch_elim_keeps_non_adjacent_branch():
    lbl = _label("far")
    other = _label("middle")
    root = BlockNode(
        insts=[
            BasicBlockNode(branch=JumpInst(label=LabelRef(lbl))),
            BasicBlockNode(labels=[LabelInst(name=other)], insts=[NopInst()]),
            BasicBlockNode(labels=[LabelInst(name=lbl)], insts=[NopInst()]),
        ]
    )

    out = _run_chunk_passes_on_root(root, [BranchEliminationPass()])

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is not None


# ---------------------------------------------------------------------------
# BlockMergePass
# ---------------------------------------------------------------------------


def test_block_merge_merges_fallthrough_blocks():
    root = BlockNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),  # no branch
            BasicBlockNode(insts=[NopInst()]),  # no labels → mergeable
        ]
    )

    out = _run_chunk_passes_on_root(root, [BlockMergePass()])

    assert len(out.insts) == 1
    merged = out.insts[0]
    assert isinstance(merged, BasicBlockNode)
    assert len(merged.insts) == 2


def test_block_merge_does_not_merge_labeled_block():
    lbl = _label("entry")
    root = BlockNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
                branch=JumpInst(label=LabelRef(lbl)),  # makes lbl referenced
            ),
        ]
    )

    out = _run_chunk_passes_on_root(root, [BlockMergePass()])

    assert len(out.insts) == 2


def test_block_merge_does_not_merge_fixed_blocks():
    root = BlockNode(
        insts=[
            BasicBlockNode(insts=[NopInst()], disable_opt=True),
            BasicBlockNode(insts=[NopInst()]),
        ]
    )

    out = _run_chunk_passes_on_root(root, [BlockMergePass()])

    assert len(out.insts) == 2


def test_block_merge_cross_boundary_dead_write_cleared_by_post_linear():
    """A dead write crossing the merge boundary is cleared when BlockMergePass + DeadWriteElimination run together."""
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1)
                    )
                ]
            ),
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(2)
                    )
                ]
            ),
        ]
    )

    # After merge: one block with two writes — dead write still present.
    merged_root = _run_chunk_passes_on_root(root, [BlockMergePass()])
    assert len(merged_root.insts) == 1
    assert len(merged_root.insts[0].insts) == 2  # type: ignore[union-attr]

    # DeadWriteElimination clears the first (dead) write.
    out = _run_chunk_passes_on_root(merged_root, [DeadWriteEliminationPass()])
    merged = out.insts[0]
    assert isinstance(merged, BasicBlockNode)
    assert len(merged.insts) == 1
    assert str(merged.insts[0].lit) == "#2"  # type: ignore[attr-defined]


def test_block_merge_chains_three_blocks():
    root = BlockNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),
            BasicBlockNode(insts=[NopInst()]),
            BasicBlockNode(insts=[NopInst()]),
        ]
    )

    out = _run_chunk_passes_on_root(root, [BlockMergePass()])

    assert len(out.insts) == 1
    assert len(out.insts[0].insts) == 3  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# BlockMergePass / BranchEliminationPass — recursion into IRLoop / IRBranch
# ---------------------------------------------------------------------------


def test_block_merge_inside_irloop_body():
    """BlockMergePass must merge blocks inside IRLoop bodies (via unparse/parse round-trip)."""
    body = BlockNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),  # no branch
            BasicBlockNode(insts=[NopInst()]),  # no labels → should be merged
        ]
    )
    insts: list = [IRLoop(name="L", counter_reg=Register("c"), body=body, n=4)]
    root = BlockNode(insts=insts)

    out = _run_chunk_passes_on_root(root, [BlockMergePass()])

    loop = out.insts[0]
    assert isinstance(loop, IRLoop)
    assert isinstance(loop.body, BlockNode)
    assert len(loop.body.insts) == 1
    merged = loop.body.insts[0]
    assert isinstance(merged, BasicBlockNode)
    assert len(merged.insts) == 2


def test_block_merge_inside_irbranch_cases():
    """BlockMergePass must merge blocks inside IRBranch cases (via unparse/parse round-trip)."""
    case = BlockNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),
            BasicBlockNode(insts=[NopInst()]),
        ]
    )
    insts: list = [IRBranch(name="B", compare_reg=Register("c"), cases=[case])]
    root = BlockNode(insts=insts)

    out = _run_chunk_passes_on_root(root, [BlockMergePass()])

    branch = out.insts[0]
    assert isinstance(branch, IRBranch)
    assert isinstance(branch.cases[0], BlockNode)
    assert len(branch.cases[0].insts) == 1
    assert isinstance(branch.cases[0].insts[0], BasicBlockNode)
    assert len(branch.cases[0].insts[0].insts) == 2


def test_branch_elim_inside_irloop_body():
    """BranchEliminationPass must recurse into IRLoop bodies."""
    lbl = _label("loop_next")
    body = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=LabelRef(lbl)),
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )
    insts: list = [IRLoop(name="L", counter_reg=Register("c"), body=body, n=4)]
    root = BlockNode(insts=insts)

    out = _run_chunk_passes_on_root(root, [BranchEliminationPass()])

    loop = out.insts[0]
    assert isinstance(loop, IRLoop)
    assert isinstance(loop.body, BlockNode)
    first_block = loop.body.insts[0]
    assert isinstance(first_block, BasicBlockNode)
    assert first_block.branch is None


def test_branch_elim_inside_irbranch_cases():
    """BranchEliminationPass must recurse into every IRBranch case."""
    lbl = _label("branch_next")
    case = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=LabelRef(lbl)),
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )
    insts: list = [IRBranch(name="B", compare_reg=Register("c"), cases=[case])]
    root = BlockNode(insts=insts)

    out = _run_chunk_passes_on_root(root, [BranchEliminationPass()])

    branch = out.insts[0]
    assert isinstance(branch, IRBranch)
    assert isinstance(branch.cases[0], BlockNode)
    first_block = branch.cases[0].insts[0]
    assert isinstance(first_block, BasicBlockNode)
    assert first_block.branch is None


# ---------------------------------------------------------------------------
# BranchEliminationPass — no next block (8.6)
# ---------------------------------------------------------------------------


def test_branch_elim_no_next_block_keeps_jump():
    """Unconditional jump at the very end of the chunk list (no next block)
    must NOT be eliminated — there is no physically following block to fall through to.
    """
    lbl = _label("far_target")
    # The jump target is NOT in this chunk list — simulates a forward reference
    # to code in a different section.  Even if it were present, when the jumping
    # block is the last BasicBlockNode in the flat list the pass returns False.
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=LabelRef(lbl)),
            ),
            # No BasicBlockNode after this one — only a non-BB node follows.
        ]
    )
    out = _run_chunk_passes_on_root(root, [BranchEliminationPass()])
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    # Branch must be preserved
    assert bb.branch is not None
    assert isinstance(bb.branch, JumpInst)
