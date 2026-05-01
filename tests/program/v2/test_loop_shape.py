import pytest
from zcu_tools.program.v2.ir.instructions import GenericInst, JumpInst, TestInst
from zcu_tools.program.v2.ir.node import IRLoop, RootNode
from zcu_tools.program.v2.ir.passes.validation import IRStructureValidationPass
from zcu_tools.program.v2.ir.pipeline import PipeLineContext


def test_loop_validation_accepts_well_formed_loop():
    loop = IRLoop(name="r", counter_reg="c", n=10)
    ir = RootNode(insts=[loop])
    IRStructureValidationPass().process(ir, PipeLineContext())


def test_loop_validation_rejects_empty_name():
    loop = IRLoop(name="", counter_reg="c", n=10)
    ir = RootNode(insts=[loop])
    with pytest.raises(ValueError, match="non-empty name"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_loop_validation_rejects_empty_counter_reg():
    loop = IRLoop(name="r", counter_reg="", n=10)
    ir = RootNode(insts=[loop])
    with pytest.raises(ValueError, match="requires a counter_reg"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_loop_validation_rejects_non_blocknode_body():
    loop = IRLoop(name="r", counter_reg="c", n=10)
    loop.body = None  # type: ignore
    ir = RootNode(insts=[loop])
    with pytest.raises(ValueError, match="body must be a BlockNode"):
        IRStructureValidationPass().process(ir, PipeLineContext())
