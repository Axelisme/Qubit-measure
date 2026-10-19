from __future__ import annotations

from dataclasses import dataclass

import pytest
from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    LabelInst,
    NopInst,
    RegWriteInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRDispatch,
    IRLoop,
    IRNode,
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
        # tolerated: Reject label markers inside a basic instruction block.
        BasicBlockNode(insts=[LabelInst(name=Label("bad"))])  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="JumpInst"):
        BasicBlockNode(insts=[JumpInst(label=LabelRef(Label("bad")))])


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

    with pytest.raises(TypeError, match="IRLoop.body"):
        loop.replace_child(new_body, BasicBlockNode(insts=[NopInst()]))


def test_irloop_body_rejects_non_block_node():
    with pytest.raises(TypeError, match="IRLoop.body must be BlockNode"):
        IRLoop(
            name="loop",
            counter_reg=Register("r0"),
            n=3,
            # tolerated: Reject a loop body that is not a BlockNode.
            body=BasicBlockNode(insts=[NopInst()]),  # pyright: ignore[reportArgumentType]
        )


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
    assert "targets=[a, b]" in str(dispatch)

    with pytest.raises(TypeError, match="leaf node"):
        dispatch.replace_child(case1, case0)


def test_clone_renamed_allocates_label_branch_and_dispatch_names():
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

    allocated: set[str] = set()
    clone_renamed(tree, allocated)
    assert allocated == {"entry", "branch", "dispatch"}


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


def test_clone_renamed_rejects_unexpected_irnode_type():
    with pytest.raises(TypeError, match="unexpected node type DummyNode"):
        clone_renamed(DummyNode(), set())


def test_clone_renamed_remaps_internal_label_refs():
    """clone_renamed must remap LabelRef targets inside the cloned body.

    A body containing a conditional back-jump to its own internal label must
    have that LabelRef updated to point to the cloned (renamed) label, not the
    original label in the un-cloned body.
    """
    from zcu_tools.program.v2.ir.node import clone_renamed

    inner_label = Label("inner")
    body = BlockNode(
        insts=[
            BasicBlockNode(
                labels=[LabelInst(name=inner_label)],
                insts=[TimeInst(c_op="inc_ref", lit=Immediate(1))],
                branch=JumpInst(
                    label=LabelRef(inner_label),
                    if_cond="S",
                    op=AluExpr(Register("r0"), AluOp.SUB, Immediate(1)),
                ),
            )
        ]
    )

    allocated: set[str] = {"inner"}
    cloned = clone_renamed(body, allocated)

    assert isinstance(cloned, BlockNode)
    assert len(cloned.insts) == 1
    bb = cloned.insts[0]
    assert isinstance(bb, BasicBlockNode)

    # The cloned label must have a different name (suffix added).
    cloned_label_name = bb.labels[0].name
    assert cloned_label_name != inner_label, "cloned LabelInst must be renamed"

    # The branch LabelRef must point to the same renamed label.
    assert bb.branch is not None
    assert isinstance(bb.branch, JumpInst)
    assert bb.branch.label is not None
    assert not bb.branch.label.is_pseudo()
    assert bb.branch.label.as_label() == cloned_label_name, (
        "cloned branch LabelRef must point to the renamed label"
    )


def test_clone_renamed_uniquifies_nested_structure_names():
    """clone_renamed must give a nested IRLoop a fresh name so the labels its
    later unparse() synthesises (``{name}_start`` etc.) do not collide."""
    from zcu_tools.program.v2.ir.node import clone_renamed

    inner = IRLoop(
        name="inner",
        counter_reg=Register("r2"),
        n=2,
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )
    body = BlockNode(insts=[inner])

    allocated: set[str] = {"inner"}
    cloned = clone_renamed(body, allocated)

    assert isinstance(cloned, BlockNode)
    cloned_loop = cloned.insts[0]
    assert isinstance(cloned_loop, IRLoop)
    assert cloned_loop.name != "inner", "nested loop name must be uniquified"


def test_clone_renamed_remaps_dmem_addr_table_labels():
    """clone_renamed must remap the entry labels inside a DmemAddr.

    A dmem dispatch instruction (REG_WR s15 op (idx + DmemAddr)) inside a
    cloned loop body must have its DmemAddr.table_labels remapped to the
    cloned case-entry labels — otherwise every unrolled copy's dispatch would
    point at the first copy's case entries.
    """
    from zcu_tools.program.v2.ir.node import clone_renamed
    from zcu_tools.program.v2.ir.operands import AluExpr, AluOp, DmemAddr

    entry = Label("case_entry_0")
    # A block that both *defines* `entry` and references it via a DmemAddr.
    body = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("s15"),
                        src=SrcKeyword.OP,
                        op=AluExpr(Register("r0"), AluOp.ADD, DmemAddr((entry,))),
                    )
                ],
            ),
            BasicBlockNode(labels=[LabelInst(name=entry)], insts=[NopInst()]),
        ]
    )

    cloned = clone_renamed(body, {"case_entry_0"})
    assert isinstance(cloned, BlockNode)

    # The defining LabelInst was renamed.
    entry_block = cloned.insts[1]
    assert isinstance(entry_block, BasicBlockNode)
    cloned_entry = entry_block.labels[0].name
    assert cloned_entry != entry

    # The DmemAddr.table_labels must point at the *cloned* entry label.
    disp_block = cloned.insts[0]
    assert isinstance(disp_block, BasicBlockNode)
    inst = disp_block.insts[0]
    assert isinstance(inst, RegWriteInst)
    assert isinstance(inst.op, AluExpr) and isinstance(inst.op.rhs, DmemAddr)
    assert inst.op.rhs.table_labels == (cloned_entry,), (
        "DmemAddr.table_labels must be remapped to the cloned entry label"
    )
