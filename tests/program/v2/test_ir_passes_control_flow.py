"""Tests for Post-LIR control-flow passes:
  - DeadLabelEliminationPass (BasicBlockNode path)
  - BranchEliminationPass
  - BlockMergePass
"""
from __future__ import annotations

from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    LabelInst,
    NopInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import BasicBlockNode, RootNode
from zcu_tools.program.v2.ir.passes.control_flow import (
    BlockMergePass,
    BranchEliminationPass,
    DeadLabelEliminationPass,
)
from zcu_tools.program.v2.ir.passes.dataflow import DeadWriteEliminationLinear
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    _run_linear_passes,
)


def _ctx() -> PipeLineContext:
    return PipeLineContext(config=PipeLineConfig())


def _label(name: str) -> Label:
    return Label(name)


# ---------------------------------------------------------------------------
# DeadLabelEliminationPass — BasicBlockNode path
# ---------------------------------------------------------------------------

def test_dead_label_elim_removes_unreferenced_label_from_basic_block():
    lbl = _label("unused")
    root = RootNode(insts=[
        BasicBlockNode(labels=[LabelInst(name=lbl)], insts=[NopInst()]),
    ])

    out = DeadLabelEliminationPass().process(root, _ctx())

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert bb.labels == []
    assert len(bb.insts) == 1


def test_dead_label_elim_keeps_referenced_label_in_basic_block():
    lbl = _label("used_lbl")
    root = RootNode(insts=[
        BasicBlockNode(labels=[LabelInst(name=lbl)], insts=[NopInst()]),
        BasicBlockNode(branch=JumpInst(label=lbl)),
    ])

    out = DeadLabelEliminationPass().process(root, _ctx())

    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.labels) == 1
    assert bb.labels[0].name == lbl


# ---------------------------------------------------------------------------
# BranchEliminationPass
# ---------------------------------------------------------------------------

def test_branch_elim_removes_unconditional_fallthrough():
    lbl = _label("next")
    root = RootNode(insts=[
        BasicBlockNode(
            insts=[NopInst()],
            branch=JumpInst(label=lbl),
        ),
        BasicBlockNode(
            labels=[LabelInst(name=lbl)],
            insts=[NopInst()],
        ),
    ])

    out = BranchEliminationPass().process(root, _ctx())

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is None
    assert len(a.insts) == 1  # NopInst still present


def test_branch_elim_nop_pads_fixed_block():
    lbl = _label("next_fixed")
    root = RootNode(insts=[
        BasicBlockNode(
            insts=[],
            branch=JumpInst(label=lbl),
            fix_inst_num=True,
        ),
        BasicBlockNode(
            labels=[LabelInst(name=lbl)],
            insts=[NopInst()],
        ),
    ])

    out = BranchEliminationPass().process(root, _ctx())

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is None
    assert len(a.insts) == 1
    assert isinstance(a.insts[0], NopInst)


def test_branch_elim_keeps_conditional_branch():
    lbl = _label("cond_target")
    root = RootNode(insts=[
        BasicBlockNode(
            insts=[NopInst()],
            branch=JumpInst(label=lbl, if_cond="Z", op="r0 - #0"),
        ),
        BasicBlockNode(
            labels=[LabelInst(name=lbl)],
            insts=[NopInst()],
        ),
    ])

    out = BranchEliminationPass().process(root, _ctx())

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is not None


def test_branch_elim_keeps_non_adjacent_branch():
    lbl = _label("far")
    other = _label("middle")
    root = RootNode(insts=[
        BasicBlockNode(branch=JumpInst(label=lbl)),
        BasicBlockNode(labels=[LabelInst(name=other)], insts=[NopInst()]),
        BasicBlockNode(labels=[LabelInst(name=lbl)], insts=[NopInst()]),
    ])

    out = BranchEliminationPass().process(root, _ctx())

    a = out.insts[0]
    assert isinstance(a, BasicBlockNode)
    assert a.branch is not None


# ---------------------------------------------------------------------------
# BlockMergePass
# ---------------------------------------------------------------------------

def test_block_merge_merges_fallthrough_blocks():
    root = RootNode(insts=[
        BasicBlockNode(insts=[NopInst()]),   # no branch
        BasicBlockNode(insts=[NopInst()]),   # no labels → mergeable
    ])

    out = BlockMergePass().process(root, _ctx())

    assert len(out.insts) == 1
    merged = out.insts[0]
    assert isinstance(merged, BasicBlockNode)
    assert len(merged.insts) == 2


def test_block_merge_does_not_merge_labeled_block():
    lbl = _label("entry")
    root = RootNode(insts=[
        BasicBlockNode(insts=[NopInst()]),
        BasicBlockNode(
            labels=[LabelInst(name=lbl)],
            insts=[NopInst()],
            branch=JumpInst(label=lbl),  # makes lbl referenced
        ),
    ])

    out = BlockMergePass().process(root, _ctx())

    assert len(out.insts) == 2


def test_block_merge_does_not_merge_fixed_blocks():
    root = RootNode(insts=[
        BasicBlockNode(insts=[NopInst()], fix_inst_num=True),
        BasicBlockNode(insts=[NopInst()]),
    ])

    out = BlockMergePass().process(root, _ctx())

    assert len(out.insts) == 2


def test_block_merge_cross_boundary_dead_write_cleared_by_post_linear():
    """A dead write crossing the merge boundary is cleared when post_linear runs after BlockMergePass."""
    root = RootNode(insts=[
        BasicBlockNode(insts=[RegWriteInst(dst="r0", src="imm", lit="#1")]),
        BasicBlockNode(insts=[RegWriteInst(dst="r0", src="imm", lit="#2")]),
    ])

    BlockMergePass().process(root, _ctx())
    # After merge: one block with two writes — dead write still present.
    assert len(root.insts) == 1
    assert len(root.insts[0].insts) == 2  # type: ignore[union-attr]

    # Post-linear clears the dead write.
    _run_linear_passes([DeadWriteEliminationLinear()], root)
    merged = root.insts[0]
    assert isinstance(merged, BasicBlockNode)
    assert len(merged.insts) == 1
    assert merged.insts[0].lit == "#2"  # type: ignore[attr-defined]


def test_block_merge_chains_three_blocks():
    root = RootNode(insts=[
        BasicBlockNode(insts=[NopInst()]),
        BasicBlockNode(insts=[NopInst()]),
        BasicBlockNode(insts=[NopInst()]),
    ])

    out = BlockMergePass().process(root, _ctx())

    assert len(out.insts) == 1
    assert len(out.insts[0].insts) == 3  # type: ignore[union-attr]
