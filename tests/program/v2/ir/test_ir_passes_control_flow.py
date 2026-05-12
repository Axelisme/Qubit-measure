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
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRLoop,
    RootNode,
)
from zcu_tools.program.v2.ir.operands import AluExpr, ImmValue, Register, AluOp
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


def _ctx() -> PipeLineContext:
    return PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)


def _label(name: str) -> Label:
    return Label(name)


def _run_chunk_passes_on_root(root: RootNode, passes: list) -> RootNode:
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
    root = RootNode(
        insts=[
            BasicBlockNode(
                labels=[LabelInst(name=lbl, can_remove=True)], insts=[NopInst()]
            ),
        ]
    )

    out, _ = DeadLabelEliminationPass().process(root, _ctx())

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert bb.labels == []
    assert len(bb.insts) == 1


def test_dead_label_elim_keeps_referenced_label_in_basic_block():
    lbl = _label("used_lbl")
    root = RootNode(
        insts=[
            BasicBlockNode(labels=[LabelInst(name=lbl)], insts=[NopInst()]),
            BasicBlockNode(branch=JumpInst(label=lbl)),
        ]
    )

    out, _ = DeadLabelEliminationPass().process(root, _ctx())

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.labels) == 1
    assert bb.labels[0].name == lbl


# ---------------------------------------------------------------------------
# BranchEliminationPass
# ---------------------------------------------------------------------------


def test_branch_elim_removes_unconditional_fallthrough():
    lbl = _label("next")
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=lbl),
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )

    out, _ = BranchEliminationPass().process(root, _ctx())

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is None
    assert len(a.insts) == 1  # NopInst still present


def test_branch_elim_skips_fixed_block():
    lbl = _label("next_fixed")
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[],
                branch=JumpInst(label=lbl),
                fix_addr_size=True,
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )

    out, _ = BranchEliminationPass().process(root, _ctx())

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is not None
    assert len(a.insts) == 0


def test_branch_elim_keeps_conditional_branch():
    lbl = _label("cond_target")
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(
                    label=lbl,
                    if_cond="Z",
                    op=AluExpr(Register("r0"), AluOp.SUB, ImmValue(0, prefix="#")),
                ),
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )

    out, _ = BranchEliminationPass().process(root, _ctx())

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is not None


def test_branch_elim_keeps_non_adjacent_branch():
    lbl = _label("far")
    other = _label("middle")
    root = RootNode(
        insts=[
            BasicBlockNode(branch=JumpInst(label=lbl)),
            BasicBlockNode(labels=[LabelInst(name=other)], insts=[NopInst()]),
            BasicBlockNode(labels=[LabelInst(name=lbl)], insts=[NopInst()]),
        ]
    )

    out, _ = BranchEliminationPass().process(root, _ctx())

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is not None


# ---------------------------------------------------------------------------
# BlockMergePass
# ---------------------------------------------------------------------------


def test_block_merge_merges_fallthrough_blocks():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),  # no branch
            BasicBlockNode(insts=[NopInst()]),  # no labels → mergeable
        ]
    )

    out, _ = BlockMergePass().process(root, _ctx())

    assert len(out.insts) == 1
    merged = out.insts[0]
    assert isinstance(merged, BasicBlockNode)
    assert len(merged.insts) == 2


def test_block_merge_does_not_merge_labeled_block():
    lbl = _label("entry")
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
                branch=JumpInst(label=lbl),  # makes lbl referenced
            ),
        ]
    )

    out, _ = BlockMergePass().process(root, _ctx())

    assert len(out.insts) == 2


def test_block_merge_does_not_merge_fixed_blocks():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[NopInst()], fix_addr_size=True),
            BasicBlockNode(insts=[NopInst()]),
        ]
    )

    out, _ = BlockMergePass().process(root, _ctx())

    assert len(out.insts) == 2


def test_block_merge_cross_boundary_dead_write_cleared_by_post_linear():
    """A dead write crossing the merge boundary is cleared when post_linear runs after BlockMergePass."""
    root = RootNode(
        insts=[
            BasicBlockNode(
                insts=[RegWriteInst(dst=Register("r0"), src="imm", lit=ImmValue(1, prefix="#"))]
            ),
            BasicBlockNode(
                insts=[RegWriteInst(dst=Register("r0"), src="imm", lit=ImmValue(2, prefix="#"))]
            ),
        ]
    )

    _, _ = BlockMergePass().process(root, _ctx())
    # After merge: one block with two writes — dead write still present.
    assert len(root.insts) == 1
    assert len(root.insts[0].insts) == 2  # type: ignore[union-attr]

    # Post-linear clears the dead write.
    root = _run_chunk_passes_on_root(root, [DeadWriteEliminationPass()])
    merged = root.insts[0]
    assert isinstance(merged, BasicBlockNode)
    assert len(merged.insts) == 1
    assert str(merged.insts[0].lit) == "#2"  # type: ignore[attr-defined]


def test_block_merge_chains_three_blocks():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),
            BasicBlockNode(insts=[NopInst()]),
            BasicBlockNode(insts=[NopInst()]),
        ]
    )

    out, _ = BlockMergePass().process(root, _ctx())

    assert len(out.insts) == 1
    assert len(out.insts[0].insts) == 3  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# BlockMergePass / BranchEliminationPass — recursion into IRLoop / IRBranch
# ---------------------------------------------------------------------------


def test_block_merge_inside_irloop_body():
    """BlockMergePass must recurse into IRLoop bodies."""
    body = BlockNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),  # no branch
            BasicBlockNode(insts=[NopInst()]),  # no labels → should be merged
        ]
    )
    insts: list = [IRLoop(name="L", counter_reg="c", body=body, n=4)]
    root = RootNode(insts=insts)

    out, _ = BlockMergePass().process(root, _ctx())

    loop = out.insts[0]
    assert isinstance(loop, IRLoop)
    assert len(loop.body.insts) == 1
    merged = loop.body.insts[0]
    assert isinstance(merged, BasicBlockNode)
    assert len(merged.insts) == 2


def test_block_merge_inside_irbranch_cases():
    """BlockMergePass must recurse into every IRBranch case."""
    case = BlockNode(
        insts=[
            BasicBlockNode(insts=[NopInst()]),
            BasicBlockNode(insts=[NopInst()]),
        ]
    )
    insts: list = [IRBranch(name="B", compare_reg="c", cases=[case])]
    root = RootNode(insts=insts)

    out, _ = BlockMergePass().process(root, _ctx())

    branch = out.insts[0]
    assert isinstance(branch, IRBranch)
    assert len(branch.cases[0].insts) == 1
    assert len(branch.cases[0].insts[0].insts) == 2  # type: ignore[union-attr]


def test_branch_elim_inside_irloop_body():
    """BranchEliminationPass must recurse into IRLoop bodies."""
    lbl = _label("loop_next")
    body = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[NopInst()],
                branch=JumpInst(label=lbl),
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )
    insts: list = [IRLoop(name="L", counter_reg="c", body=body, n=4)]
    root = RootNode(insts=insts)

    out, _ = BranchEliminationPass().process(root, _ctx())

    loop = out.insts[0]
    assert isinstance(loop, IRLoop)
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
                branch=JumpInst(label=lbl),
            ),
            BasicBlockNode(
                labels=[LabelInst(name=lbl)],
                insts=[NopInst()],
            ),
        ]
    )
    insts: list = [IRBranch(name="B", compare_reg="c", cases=[case])]
    root = RootNode(insts=insts)

    out, _ = BranchEliminationPass().process(root, _ctx())

    branch = out.insts[0]
    assert isinstance(branch, IRBranch)
    first_block = branch.cases[0].insts[0]
    assert isinstance(first_block, BasicBlockNode)
    assert first_block.branch is None
