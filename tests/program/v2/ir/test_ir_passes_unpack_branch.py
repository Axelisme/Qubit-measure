"""Tests for ir/passes/control_flow/unpack_branch.py.

Covers:
- _first_basic_block(): BasicBlockNode, nested BlockNode, no-head cases
- _case_entry_label(): existing label reuse, label synthesis (head/no-head)
- UnpackIRBranchPass.transform(): non-IRBranch passthrough, 2-case, 3-case,
  big-pmem jump, end-label uniqueness
"""

from __future__ import annotations

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
    IRDispatch,
    IRLoop,
    IRNode,
)
from zcu_tools.program.v2.ir.operands import Register, SrcKeyword
from zcu_tools.program.v2.ir.passes.control_flow.unpack_branch import (
    UnpackIRBranchPass,
    _case_entry_label,
    _first_basic_block,
)
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_inst(name: str, can_remove: bool = False) -> LabelInst:
    return LabelInst(name=Label(name), can_remove=can_remove)


def _nop_block(label_name: str | None = None) -> BasicBlockNode:
    labels = [_label_inst(label_name)] if label_name else []
    return BasicBlockNode(labels=labels, insts=[NopInst()])


def _ctx(pmem_capacity: int = 512) -> PipeLineContext:
    return PipeLineContext(
        config=PipeLineConfig(pmem_capacity=pmem_capacity), pmem_budget=1024
    )


def _apply_unpack(root: BlockNode, pmem_capacity: int = 512) -> tuple[BlockNode, bool]:
    """Run UnpackIRBranchPass on root via the real post-order driver."""
    from zcu_tools.program.v2.ir.pipeline import _optimize_tree

    ctx = _ctx(pmem_capacity)
    result = _optimize_tree(root, [UnpackIRBranchPass()], ctx)
    if isinstance(result, BlockNode):
        return result, result is not root
    return BlockNode(insts=[result]), True


def _collect_nodes(root: IRNode, cls: type) -> list:
    """Recursively collect all nodes of a given type in the tree."""
    found = []
    if isinstance(root, cls):
        found.append(root)
    if isinstance(root, (BlockNode,)):
        for child in root.insts:
            found.extend(_collect_nodes(child, cls))
    elif isinstance(root, BasicBlockNode):
        pass  # leaf
    return found


# ---------------------------------------------------------------------------
# _first_basic_block
# ---------------------------------------------------------------------------


def test_first_basic_block_returns_itself_when_given_bb():
    bb = _nop_block("a")
    assert _first_basic_block(bb) is bb


def test_first_basic_block_recurses_into_block_node():
    inner = _nop_block("inner")
    outer = BlockNode(insts=[inner, _nop_block("other")])
    assert _first_basic_block(outer) is inner


def test_first_basic_block_returns_none_for_empty_block_node():
    empty = BlockNode(insts=[])
    assert _first_basic_block(empty) is None


def test_first_basic_block_returns_none_for_non_block_node():
    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=3,
        body=BlockNode(insts=[_nop_block()]),
    )
    assert _first_basic_block(loop) is None


# ---------------------------------------------------------------------------
# _case_entry_label
# ---------------------------------------------------------------------------


def test_case_entry_label_reuses_existing_label():
    existing = _label_inst("my_entry")
    bb = BasicBlockNode(labels=[existing], insts=[NopInst()])
    case2, entry = _case_entry_label(bb, "fallback", set())

    assert case2 is bb
    assert entry == Label("my_entry")
    # original labels untouched
    assert bb.labels == [existing]


def test_case_entry_label_synthesises_label_when_head_has_none():
    bb = BasicBlockNode(labels=[], insts=[NopInst()])
    case2, entry = _case_entry_label(bb, "sel_case_entry_0", set())

    # case2 is the same bb object (label inserted in-place)
    assert case2 is bb
    assert entry == Label("sel_case_entry_0")
    assert bb.labels[0].name == Label("sel_case_entry_0")


def test_case_entry_label_synthesises_unique_label_on_conflict():
    bb = BasicBlockNode(labels=[], insts=[NopInst()])
    # "sel_case_entry_0" is already allocated
    _, entry = _case_entry_label(bb, "sel_case_entry_0", {"sel_case_entry_0"})

    # make_label must have produced a suffixed variant
    assert entry != Label("sel_case_entry_0")
    assert bb.labels[0].name == entry


def test_case_entry_label_no_head_block_node_wraps_in_block():
    # A BlockNode whose first child is not a BasicBlockNode has no BB head.
    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=2,
        body=BlockNode(insts=[_nop_block()]),
    )
    inner = BlockNode(insts=[loop])
    case2, entry = _case_entry_label(inner, "br_case_entry_0", set())

    assert isinstance(case2, BlockNode)
    # First child of the new BlockNode must be a labelled BasicBlockNode
    head = case2.insts[0]
    assert isinstance(head, BasicBlockNode)
    assert head.labels[0].name == entry


def test_case_entry_label_no_head_non_block_node_wraps_in_block():
    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=2,
        body=BlockNode(insts=[_nop_block()]),
    )
    case2, entry = _case_entry_label(loop, "br_case_entry_0", set())

    assert isinstance(case2, BlockNode)
    head = case2.insts[0]
    assert isinstance(head, BasicBlockNode)
    assert head.labels[0].name == entry
    assert case2.insts[1] is loop


# ---------------------------------------------------------------------------
# UnpackIRBranchPass.transform — non-IRBranch passthrough
# ---------------------------------------------------------------------------


def test_unpack_branch_ignores_non_branch_node():
    bb = _nop_block("x")
    root = BlockNode(insts=[bb])
    out, changed = _apply_unpack(root)

    assert not changed
    assert out.insts[0] is bb


# ---------------------------------------------------------------------------
# UnpackIRBranchPass.transform — 2-case branch (small pmem)
# ---------------------------------------------------------------------------


def _make_branch(name: str, n_cases: int) -> IRBranch:
    cases: list[IRNode] = [
        BasicBlockNode(
            labels=[_label_inst(f"{name}_case_entry_{i}")],
            insts=[NopInst()],
        )
        for i in range(n_cases)
    ]
    return IRBranch(
        name=name,
        compare_reg=Register("r_sel"),
        cases=cases,
    )


def test_unpack_two_case_produces_ir_dispatch():
    branch = _make_branch("br", 2)
    root = BlockNode(insts=[branch])
    out, _ = _apply_unpack(root)

    dispatches = _collect_nodes(out, IRDispatch)
    assert len(dispatches) == 1
    assert len(dispatches[0].target_labels) == 2


def test_unpack_two_case_dispatch_targets_match_case_entry_labels():
    branch = _make_branch("br", 2)
    root = BlockNode(insts=[branch])
    out, _ = _apply_unpack(root)

    dispatch = _collect_nodes(out, IRDispatch)[0]
    assert dispatch.target_labels[0] == Label("br_case_entry_0")
    assert dispatch.target_labels[1] == Label("br_case_entry_1")


def test_unpack_two_case_first_case_has_jump_to_end():
    branch = _make_branch("br", 2)
    root = BlockNode(insts=[branch])
    out, _ = _apply_unpack(root)

    # Flatten the children of the result BlockNode
    expanded = out.insts[0]  # the BlockNode replacing IRBranch
    assert isinstance(expanded, BlockNode)
    children = expanded.insts

    # Order: IRDispatch, case_0, jump_to_end_bb, case_1, end_label_bb
    assert isinstance(children[0], IRDispatch)
    # Jump-to-end block after case 0 (index 2)
    jump_bb = children[2]
    assert isinstance(jump_bb, BasicBlockNode)
    assert isinstance(jump_bb.branch, JumpInst)
    assert jump_bb.branch.if_cond is None  # unconditional


def test_unpack_two_case_last_case_has_no_extra_jump():
    branch = _make_branch("br", 2)
    root = BlockNode(insts=[branch])
    out, _ = _apply_unpack(root)

    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    children = expanded.insts

    # children: IRDispatch, case_0, jump_bb, case_1, end_label_bb
    # case_1 is children[3]; end_label_bb is children[4]
    # No extra jump between case_1 and end_label
    assert len(children) == 5
    end_bb = children[-1]
    assert isinstance(end_bb, BasicBlockNode)
    assert end_bb.labels  # has the end label


def test_unpack_three_case_produces_two_inter_case_jumps():
    branch = _make_branch("br", 3)
    root = BlockNode(insts=[branch])
    out, _ = _apply_unpack(root)

    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    children = expanded.insts

    # IRDispatch + case_0 + jump + case_1 + jump + case_2 + end_label
    assert len(children) == 7
    assert isinstance(children[2], BasicBlockNode)  # jump after case_0
    assert isinstance(children[2].branch, JumpInst)
    assert isinstance(children[4], BasicBlockNode)  # jump after case_1
    assert isinstance(children[4].branch, JumpInst)


# ---------------------------------------------------------------------------
# UnpackIRBranchPass.transform — big-pmem jump (pmem_capacity > 2048)
# ---------------------------------------------------------------------------


def test_unpack_big_pmem_uses_reg_write_s15_jump():
    branch = _make_branch("br", 2)
    root = BlockNode(insts=[branch])
    out, _ = _apply_unpack(root, pmem_capacity=4096)

    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    children = expanded.insts

    # Jump after case_0 should be a BasicBlockNode with RegWriteInst(s15) + JumpInst(addr=s15)
    jump_bb = children[2]
    assert isinstance(jump_bb, BasicBlockNode)
    assert len(jump_bb.insts) == 1
    rw = jump_bb.insts[0]
    assert isinstance(rw, RegWriteInst)
    assert rw.dst == Register("s15")
    assert rw.src == SrcKeyword.LABEL

    assert isinstance(jump_bb.branch, JumpInst)
    assert jump_bb.branch.addr == Register("s15")


def test_unpack_small_pmem_does_not_use_s15():
    branch = _make_branch("br", 2)
    root = BlockNode(insts=[branch])
    out, _ = _apply_unpack(root, pmem_capacity=512)

    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    jump_bb = expanded.insts[2]
    assert isinstance(jump_bb, BasicBlockNode)
    assert not jump_bb.insts  # no RegWriteInst
    assert isinstance(jump_bb.branch, JumpInst)
    assert jump_bb.branch.label is not None  # label-mode jump


# ---------------------------------------------------------------------------
# End-label uniqueness when name would conflict with existing labels
# ---------------------------------------------------------------------------


def test_unpack_end_label_is_unique_when_name_conflicts():
    # Seed one of the cases with a label "br_end" to force make_label to suffix.
    case_0 = BasicBlockNode(
        labels=[_label_inst("br_case_entry_0"), _label_inst("br_end")],
        insts=[NopInst()],
    )
    case_1 = BasicBlockNode(
        labels=[_label_inst("br_case_entry_1")],
        insts=[NopInst()],
    )
    branch = IRBranch(name="br", compare_reg=Register("r_sel"), cases=[case_0, case_1])
    root = BlockNode(insts=[branch])
    out, _ = _apply_unpack(root)

    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    end_bb = expanded.insts[-1]
    assert isinstance(end_bb, BasicBlockNode)
    # end_label must differ from "br_end"
    assert end_bb.labels[0].name != Label("br_end")


def test_unpack_same_named_branches_keep_labels_globally_unique():
    from zcu_tools.program.v2.ir.factory import IRParser
    from zcu_tools.program.v2.ir.pipeline import _optimize_tree

    def _unlabeled_branch() -> IRBranch:
        return IRBranch(
            name="sel",
            compare_reg=Register("r_sel"),
            cases=[
                BasicBlockNode(insts=[NopInst()]),
                BasicBlockNode(insts=[NopInst()]),
            ],
        )

    root = BlockNode(insts=[_unlabeled_branch(), _unlabeled_branch()])
    parser = IRParser(pmem_size=512)
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1024,
        allocated_names={"sel"},
    )

    optimized = _optimize_tree(root, [UnpackIRBranchPass()], ctx)
    assert isinstance(optimized, BlockNode)

    chunks = parser.unparse(optimized)
    labels = [
        lbl.name.name
        for chunk in chunks
        if isinstance(chunk, BasicBlockNode)
        for lbl in chunk.labels
    ]
    assert len(labels) == len(set(labels))


def test_unpack_after_branch_roundtrip_does_not_duplicate_old_skeleton():
    from zcu_tools.program.v2.ir.factory import IRParser

    parser = IRParser(pmem_size=512)
    branch = IRBranch(
        name="sel",
        compare_reg=Register("r_sel"),
        cases=[
            BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
            BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
            BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
        ],
    )
    rebuilt = parser.parse(parser.unparse(BlockNode(insts=[branch])))
    out, _ = _apply_unpack(rebuilt)

    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    children = expanded.insts
    assert len(children) == 7
    assert isinstance(children[0], IRDispatch)
    jump_blocks = [
        child
        for child in children
        if isinstance(child, BasicBlockNode) and isinstance(child.branch, JumpInst)
    ]
    assert len(jump_blocks) == 2
