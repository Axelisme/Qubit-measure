import pytest
from zcu_tools.program.v2.ir.instructions import TestInst, JumpInst, GenericInst
from zcu_tools.program.v2.ir.node import IRLoop, RootNode
from zcu_tools.program.v2.ir.passes.validation import IRStructureValidationPass
from zcu_tools.program.v2.ir.pipeline import PipeLineContext

def test_loop_validation_accepts_well_formed_loop():
    loop = IRLoop(name="r")
    loop.stop_check.append(TestInst(op="r1-r2"))
    loop.stop_check.append(JumpInst(label="end", if_cond="eq"))
    loop.jump_back.append(JumpInst(label="start"))
    
    ir = RootNode(insts=[loop])
    IRStructureValidationPass().process(ir, PipeLineContext())

def test_loop_validation_rejects_missing_stop_check():
    loop = IRLoop(name="r")
    # Missing stop_check content
    loop.jump_back.append(JumpInst(label="start"))
    
    ir = RootNode(insts=[loop])
    with pytest.raises(ValueError, match=r"stop_check must end with TEST \+ JUMP pair"):
        IRStructureValidationPass().process(ir, PipeLineContext())

def test_loop_validation_rejects_non_test_in_stop_check():
    loop = IRLoop(name="r")
    loop.stop_check.append(GenericInst(cmd="NOP"))
    loop.stop_check.append(JumpInst(label="end", if_cond="eq"))
    loop.jump_back.append(JumpInst(label="start"))
    
    ir = RootNode(insts=[loop])
    with pytest.raises(ValueError, match=r"stop_check must end with TEST \+ conditional JUMP pair"):
        IRStructureValidationPass().process(ir, PipeLineContext())

def test_loop_validation_rejects_unconditional_jump_in_stop_check():
    loop = IRLoop(name="r")
    loop.stop_check.append(TestInst(op="r1-r2"))
    loop.stop_check.append(JumpInst(label="end")) # Missing if_cond
    loop.jump_back.append(JumpInst(label="start"))
    
    ir = RootNode(insts=[loop])
    with pytest.raises(ValueError, match=r"stop_check must end with TEST \+ conditional JUMP pair"):
        IRStructureValidationPass().process(ir, PipeLineContext())

def test_loop_validation_rejects_missing_jump_back():
    loop = IRLoop(name="r")
    loop.stop_check.append(TestInst(op="r1-r2"))
    loop.stop_check.append(JumpInst(label="end", if_cond="eq"))
    # Missing jump_back
    
    ir = RootNode(insts=[loop])
    with pytest.raises(ValueError, match="jump_back cannot be empty"):
        IRStructureValidationPass().process(ir, PipeLineContext())

def test_loop_validation_rejects_conditional_jump_in_jump_back():
    loop = IRLoop(name="r")
    loop.stop_check.append(TestInst(op="r1-r2"))
    loop.stop_check.append(JumpInst(label="end", if_cond="eq"))
    loop.jump_back.append(JumpInst(label="start", if_cond="nz")) # Should be unconditional
    
    ir = RootNode(insts=[loop])
    with pytest.raises(ValueError, match="jump_back must end with unconditional JUMP"):
        IRStructureValidationPass().process(ir, PipeLineContext())
