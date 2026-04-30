import pytest
from zcu_tools.program.v2.ir.instructions import GenericInst, LabelInst, MetaInst
from zcu_tools.program.v2.ir.node import BlockNode, IRBranch, IRLoop, RootNode
from zcu_tools.program.v2.ir.passes.dce import LabelDCEPass
from zcu_tools.program.v2.ir.passes.validation import (
    IRStructureValidationPass,
    LabelReferenceValidationPass,
)
from zcu_tools.program.v2.ir.pipeline import PipeLineContext
from zcu_tools.program.v2.ir.traversal import walk_instructions


def test_label_dce_pass():
    """
    Test that the LabelDCEPass removes unreferenced labels from ir.labels.
    """
    insts = [
        GenericInst(cmd="NOP"),
        GenericInst(cmd="JUMP", args={"LABEL": "used_label"}),
        LabelInst(name="unused_definition"),
    ]

    labels = {
        "used_label": "&1",
        "dead_label": "&2"
    }

    ir = RootNode(insts=insts, labels=labels)
    
    dce_pass = LabelDCEPass()
    opt_ir = dce_pass.process(ir, PipeLineContext())
    
    assert "used_label" in opt_ir.labels
    assert "dead_label" not in opt_ir.labels
    assert "unused_definition" not in opt_ir.labels


def test_label_dce_pass_traverses_irloop_sections():
    loop = IRLoop(name="r")
    loop.stop_check.append(GenericInst(cmd="JUMP", args={"LABEL": "used_in_loop"}))
    loop.body.append(GenericInst(cmd="NOP"))

    ir = RootNode(
        insts=[loop],
        labels={"used_in_loop": "&1", "dead_label": "&2"},
    )

    dce_pass = LabelDCEPass()
    opt_ir = dce_pass.process(ir, PipeLineContext())

    assert "used_in_loop" in opt_ir.labels
    assert "dead_label" not in opt_ir.labels


def test_walk_instructions_traverses_irloop_sections_once():
    loop = IRLoop(name="r")
    loop.initial.append(GenericInst(cmd="INIT"))
    loop.stop_check.append(GenericInst(cmd="CHECK"))
    loop.body.append(GenericInst(cmd="BODY"))
    loop.update.append(GenericInst(cmd="UPDATE"))
    ir = RootNode(insts=[loop])

    assert [inst.cmd for inst in walk_instructions(ir)] == [
        "INIT",
        "CHECK",
        "BODY",
        "UPDATE",
    ]


def test_structure_validation_rejects_meta_inst_in_tree():
    ir = RootNode(insts=[MetaInst(type="LOOP_START", name="r")])

    with pytest.raises(ValueError, match="MetaInst"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_structure_validation_rejects_branch_case_not_in_body():
    case = BlockNode()
    branch = IRBranch(name="sel", cases=[case])
    ir = RootNode(insts=[branch])

    with pytest.raises(ValueError, match="case is not present"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_label_reference_validation_rejects_missing_label():
    ir = RootNode(
        insts=[GenericInst(cmd="JUMP", args={"LABEL": "missing"})],
        labels={"other": "&1"},
    )

    with pytest.raises(ValueError, match="Undefined label reference"):
        LabelReferenceValidationPass().process(ir, PipeLineContext())


def test_label_reference_validation_ignores_qick_pseudo_labels():
    ir = RootNode(insts=[GenericInst(cmd="JUMP", args={"LABEL": "HERE"})])

    out = LabelReferenceValidationPass().process(ir, PipeLineContext())

    assert out is ir
