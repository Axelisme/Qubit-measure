"""Tests for ir/passes/control_flow/unpack_branch.py.

Covers public branch expansion and pipeline lowering:
- case-entry label reuse and synthesis for head/no-head cases
- non-IRBranch passthrough, 2-case and 3-case control flow
- big-pmem jumps and whole-pipeline label uniqueness
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
)
from zcu_tools.program.v2.ir.pipeline import IRPipeLine, PipeLineConfig, PipeLineContext

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


def _apply_unpack(root: BlockNode, pmem_capacity: int = 512) -> BlockNode:
    """Exercise the public branch-pass contract on a prepared branch."""
    assert len(root.insts) == 1
    node = root.insts[0]
    result = UnpackIRBranchPass().transform(node, _ctx(pmem_capacity))
    return BlockNode(insts=[node if result is None else result])


def _collect_nodes[T: IRNode](root: IRNode, cls: type[T]) -> list[T]:
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


def _expand_single_case(case: IRNode) -> tuple[BlockNode, Label]:
    branch = IRBranch(name="sel", compare_reg=Register("r_sel"), cases=[case])
    ctx = _ctx()
    out = UnpackIRBranchPass().transform(branch, ctx)
    assert isinstance(out, BlockNode)
    dispatch = out.insts[0]
    assert isinstance(dispatch, IRDispatch)
    expanded_case = out.insts[1]
    if not isinstance(expanded_case, BlockNode):
        expanded_case = BlockNode(insts=[expanded_case])
    return expanded_case, dispatch.target_labels[0]


def test_unpack_reuses_basic_block_entry_label():
    existing = _label_inst("my_entry")
    case, target = _expand_single_case(
        BasicBlockNode(labels=[existing], insts=[NopInst()])
    )
    head = case.insts[0]
    assert isinstance(head, BasicBlockNode)
    assert head.labels == [existing]
    assert target == existing.name
    assert head.insts == [NopInst()]


def test_unpack_reuses_nested_block_entry_label():
    inner = _nop_block("inner")
    case, target = _expand_single_case(
        BlockNode(insts=[BlockNode(insts=[inner]), _nop_block("other")])
    )
    assert target == Label("inner")
    assert case.insts[0].children()[0] == inner


def test_unpack_synthesises_entry_for_unlabelled_head():
    case, target = _expand_single_case(_nop_block())
    head = case.insts[0]
    assert isinstance(head, BasicBlockNode)
    assert [label.name for label in head.labels] == [target]
    assert head.insts == [NopInst()]


def test_unpack_synthesises_entry_for_empty_case():
    case, target = _expand_single_case(BlockNode())
    head = case.insts[0]
    assert isinstance(head, BasicBlockNode)
    assert [label.name for label in head.labels] == [target]
    assert not head.insts


def test_unpack_synthesises_entry_before_loop_case():
    loop = IRLoop(
        name="lp", counter_reg=Register("r0"), n=2, body=BlockNode(insts=[_nop_block()])
    )
    case, target = _expand_single_case(loop)
    head = case.insts[0]
    assert isinstance(head, BasicBlockNode)
    assert [label.name for label in head.labels] == [target]
    assert case.insts[1] == loop


def test_unpack_synthesises_entry_for_block_starting_with_loop():
    loop = IRLoop(
        name="lp", counter_reg=Register("r0"), n=2, body=BlockNode(insts=[_nop_block()])
    )
    case, target = _expand_single_case(BlockNode(insts=[loop]))
    head = case.insts[0]
    assert isinstance(head, BasicBlockNode)
    assert [label.name for label in head.labels] == [target]
    assert case.insts[1] == loop


# ---------------------------------------------------------------------------
# UnpackIRBranchPass.transform — non-IRBranch passthrough
# ---------------------------------------------------------------------------


def test_unpack_branch_ignores_non_branch_node():
    assert UnpackIRBranchPass().transform(_nop_block("x"), _ctx()) is None


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
    out = _apply_unpack(root)

    dispatches = _collect_nodes(out, IRDispatch)
    assert len(dispatches) == 1
    assert len(dispatches[0].target_labels) == 2


def test_unpack_two_case_dispatch_targets_match_case_entry_labels():
    branch = _make_branch("br", 2)
    root = BlockNode(insts=[branch])
    out = _apply_unpack(root)

    dispatch = _collect_nodes(out, IRDispatch)[0]
    assert dispatch.target_labels[0] == Label("br_case_entry_0")
    assert dispatch.target_labels[1] == Label("br_case_entry_1")


def test_unpack_two_case_first_case_has_jump_to_end():
    branch = _make_branch("br", 2)
    root = BlockNode(insts=[branch])
    out = _apply_unpack(root)

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
    out = _apply_unpack(root)

    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    children = expanded.insts

    # The IRDispatch node precedes the first case and that case's jump block.
    # The last case and the end-label block follow the jump block.
    # The last case occupies the fourth child position.
    # The end-label block occupies the fifth child position.
    # No extra jump between case_1 and end_label
    assert len(children) == 5
    end_bb = children[-1]
    assert isinstance(end_bb, BasicBlockNode)
    assert end_bb.labels  # has the end label


def test_unpack_three_case_produces_two_inter_case_jumps():
    branch = _make_branch("br", 3)
    root = BlockNode(insts=[branch])
    out = _apply_unpack(root)

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
    out = _apply_unpack(root, pmem_capacity=4096)

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
    out = _apply_unpack(root, pmem_capacity=512)

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
    out = _apply_unpack(root)

    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    end_bb = expanded.insts[-1]
    assert isinstance(end_bb, BasicBlockNode)
    # end_label must differ from "br_end"
    assert end_bb.labels[0].name != Label("br_end")


def test_unpack_same_named_branches_keep_labels_globally_unique():
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser

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
    insts = IRLexer().flatten(parser.unparse(root))
    pipeline = IRPipeLine(
        PipeLineConfig(pmem_capacity=512), [], [], [UnpackIRBranchPass()]
    )
    out, _ = pipeline(insts)
    labels = [inst.name for inst in out if isinstance(inst, LabelInst)]
    assert labels
    assert len(labels) == len(set(labels))
    references = [
        inst.label.as_label()
        for inst in out
        if isinstance(inst, JumpInst) and inst.label is not None
    ]
    assert references
    assert set(references) <= set(labels)


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
    out = _apply_unpack(rebuilt)

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
