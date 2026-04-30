import pytest

from zcu_tools.program.v2.ir.node import IRNode
from zcu_tools.program.v2.ir.instructions import LabelInst, GenericInst
from zcu_tools.program.v2.ir.passes.dce import LabelDCEPass
from zcu_tools.program.v2.ir.pipeline import PipeLineContext


def test_label_dce_pass():
    """
    Test that the LabelDCEPass removes unused labels from the ir.labels dictionary.
    """
    insts = [
        GenericInst(cmd="NOP"),
        # This acts as a reference to "used_label"
        LabelInst(name="used_label"),
    ]
    
    labels = {
        "used_label": "&1",
        "dead_label": "&2"
    }
    
    ir = IRNode(insts=insts, labels=labels)
    
    dce_pass = LabelDCEPass()
    opt_ir = dce_pass.process(ir, PipeLineContext())
    
    assert "used_label" in opt_ir.labels
    assert "dead_label" not in opt_ir.labels
