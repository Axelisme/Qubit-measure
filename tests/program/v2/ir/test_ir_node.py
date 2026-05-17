from __future__ import annotations

from dataclasses import dataclass

import pytest
from zcu_tools.program.v2.ir.instructions import (
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
    _clone_node,
    _collect_subtree_names,
    clone_renamed,
)
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    Register,
    SrcKeyword,
)


@dataclass
class DummyNode(IRNode):
    child: IRNode | None = None

    def children(self) -> list[IRNode]:
        return [self.child] if self.child is not None else []

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        if self.child is not old:
            raise ValueError("not found")
        self.child = new


def test_basic_block_leaf_api_string_and_validation():
    block = BasicBlockNode(
        labels=[LabelInst(name=Label("entry"))],
        insts=[NopInst()],
        branch=JumpInst(label=LabelRef(Label("done"))),
        disable_opt=True,
    )

    assert block.children() == []
    assert block.addr_size == 2
    assert "disable_opt=2" in str(block)
    assert "name=entry" in str(block)
    assert "JumpInst" in str(block)

    with pytest.raises(TypeError, match="leaf node"):
        block.replace_child(DummyNode(), DummyNode())

    with pytest.raises(ValueError, match="LabelInst"):
        BasicBlockNode(insts=[LabelInst(name=Label("bad"))])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="JumpInst"):
        BasicBlockNode(insts=[JumpInst(label=LabelRef(Label("bad")))])  # type: ignore[list-item]


def test_block_node_mutation_and_string():
    first = BasicBlockNode(insts=[NopInst()])
    second = BasicBlockNode(labels=[LabelInst(name=Label("second"))], insts=[NopInst()])
    root = BlockNode(insts=[first])

    root.append(second)
    assert root.children() == [first, second]
    assert "BlockNode()" in str(root)

    replacement = BasicBlockNode(insts=[NopInst(), NopInst()])
    root.replace_child(second, replacement)
    assert root.insts == [first, replacement]


def test_irloop_tree_api_and_string():
    body = BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
    loop = IRLoop(
        name="loop",
        counter_reg=Register("r0"),
        n=3,
        body=body,
        range_hint=(0, 3),
    )

    assert loop.children() == [body]
    assert "range_hint=(0, 3)" in str(loop)

    new_body = BlockNode(insts=[BasicBlockNode(insts=[NopInst(), NopInst()])])
    loop.replace_child(body, new_body)
    assert loop.body is new_body

    with pytest.raises(ValueError, match="not a child"):
        loop.replace_child(body, new_body)


def test_irbranch_and_irdispatch_tree_api_and_string():
    case0 = BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
    case1 = BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
    branch = IRBranch(name="branch", compare_reg=Register("r1"), cases=[case0, case1])
    dispatch = IRDispatch(
        name="dispatch",
        value_reg=Register("r2"),
        target_labels=[Label("a"), Label("b")],
    )

    assert branch.children() == [case0, case1]
    assert "compare_reg=r1" in str(branch)
    branch.replace_child(case0, dispatch)
    assert branch.cases[0] is dispatch

    assert dispatch.children() == []
    assert "targets=[&a, &b]" in str(dispatch)

    with pytest.raises(TypeError, match="leaf node"):
        dispatch.replace_child(case1, case0)


def test_collect_subtree_names_includes_branch_and_dispatch_nodes():
    inner_dispatch = IRDispatch(
        name="dispatch",
        value_reg=Register("r2"),
        target_labels=[Label("case0"), Label("case1")],
    )
    tree = BlockNode(
        insts=[
            IRBranch(
                name="branch",
                compare_reg=Register("r1"),
                cases=[
                    BlockNode(
                        insts=[
                            BasicBlockNode(
                                labels=[LabelInst(name=Label("entry"))],
                                insts=[NopInst()],
                            ),
                            inner_dispatch,
                        ]
                    )
                ],
            )
        ]
    )

    labels, structs = _collect_subtree_names(tree)
    assert labels == {"entry"}
    assert structs == {"branch", "dispatch"}


def test_clone_renamed_remaps_regwrite_label_and_dispatch_structure_names():
    label = Label("entry")
    dispatch = IRDispatch(
        name="dispatch",
        value_reg=Register("r2"),
        target_labels=[label],
    )
    branch = IRBranch(
        name="branch",
        compare_reg=Register("r1"),
        cases=[
            BlockNode(
                insts=[
                    BasicBlockNode(
                        labels=[LabelInst(name=label)],
                        insts=[
                            RegWriteInst(
                                dst=Register("r0"),
                                src=SrcKeyword.LABEL,
                                label=LabelRef(label),
                            )
                        ],
                        branch=JumpInst(label=LabelRef("HERE")),
                    ),
                    dispatch,
                ]
            )
        ],
    )

    cloned = clone_renamed(branch, {"entry", "branch", "dispatch"})

    assert isinstance(cloned, IRBranch)
    assert cloned.name != "branch"
    cloned_case = cloned.cases[0]
    assert isinstance(cloned_case, BlockNode)

    cloned_block = cloned_case.insts[0]
    assert isinstance(cloned_block, BasicBlockNode)
    cloned_label = cloned_block.labels[0].name
    assert cloned_label != label

    inst = cloned_block.insts[0]
    assert isinstance(inst, RegWriteInst)
    assert inst.label is not None
    assert inst.label.as_label() == cloned_label

    assert cloned_block.branch is not None
    assert cloned_block.branch.label is not None
    assert cloned_block.branch.label.is_pseudo()
    assert cloned_block.branch.label.target == "HERE"

    cloned_dispatch = cloned_case.insts[1]
    assert isinstance(cloned_dispatch, IRDispatch)
    assert cloned_dispatch.name != "dispatch"
    assert cloned_dispatch.target_labels == [cloned_label]


def test_clone_node_rejects_unexpected_irnode_type():
    with pytest.raises(TypeError, match="unexpected node type DummyNode"):
        _clone_node(DummyNode(), {}, {})
