import pytest

from lib.zcu_tools.program.v2.ir.instructions import BaseInst, PortWriteInst, RegWriteInst
from lib.zcu_tools.program.v2.ir.node import BasicBlockNode
from lib.zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    ImmValue,
    Register,
    SrcKeyword,
)
from lib.zcu_tools.program.v2.ir.passes.dataflow.dead_write import (
    DeadWriteEliminationPass,
)
from lib.zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext


@pytest.fixture
def ctx():
    return PipeLineContext(config=PipeLineConfig(), pmem_budget=4096)

def test_w0_w1_independence(ctx):
    # REG_WR w0 #1
    # REG_WR w1 #2
    # Both should survive.
    inst1 = RegWriteInst(dst=Register("w0"), src=SrcKeyword.IMM, lit=Immediate(1))
    inst2 = RegWriteInst(dst=Register("w1"), src=SrcKeyword.IMM, lit=Immediate(2))
    block = BasicBlockNode(insts=[inst1, inst2])
    
    pass_obj = DeadWriteEliminationPass()
    pass_obj.process([block], ctx)
    
    assert len(block.insts) == 2
    assert block.insts[0] == inst1
    assert block.insts[1] == inst2

def test_w0_shadowed_by_r_wave(ctx):
    # REG_WR w0 #1
    # REG_WR r_wave wmem [&0]
    # w0 write is dead because r_wave overwrites all w0-w5.
    inst1 = RegWriteInst(dst=Register("w0"), src=SrcKeyword.IMM, lit=Immediate(1))
    inst2 = RegWriteInst(dst=Register("r_wave"), src=SrcKeyword.WMEM) # Simplified for test
    block = BasicBlockNode(insts=[inst1, inst2])
    
    pass_obj = DeadWriteEliminationPass()
    pass_obj.process([block], ctx)
    
    # inst2 (wmem read) is never removed, but it shadows inst1.
    assert len(block.insts) == 1
    assert block.insts[0] == inst2

def test_r_wave_not_shadowed_by_w0(ctx):
    # REG_WR r_wave wmem [&0]
    # REG_WR w0 #1
    # r_wave write is NOT dead because w1-w5 are still valid from r_wave.
    inst1 = RegWriteInst(dst=Register("r_wave"), src=SrcKeyword.WMEM)
    inst2 = RegWriteInst(dst=Register("w0"), src=SrcKeyword.IMM, lit=Immediate(1))
    block = BasicBlockNode(insts=[inst1, inst2])
    
    pass_obj = DeadWriteEliminationPass()
    pass_obj.process([block], ctx)
    
    assert len(block.insts) == 2
    assert block.insts[0] == inst1
    assert block.insts[1] == inst2

def test_r_wave_fully_shadowed_by_individual_writes(ctx):
    # REG_WR r_wave op (r_wave + #0)  -- Using OP to make it eligible for DCE (wmem is a barrier)
    # REG_WR w0 #0
    # REG_WR w1 #1
    # REG_WR w2 #2
    # REG_WR w3 #3
    # REG_WR w4 #4
    # REG_WR w5 #5
    # First r_wave write should be dead.
    
    # We need an instruction that writes r_wave but isn't a barrier.
    # REG_WR r_wave op r_wave + #0 (pseudo-copy)
    inst_r = RegWriteInst(dst=Register("r_wave"), src=SrcKeyword.OP, op=AluExpr(Register("r_wave"), AluOp.ADD, Immediate(0)))
    
    writes: list[BaseInst] = [RegWriteInst(dst=Register(f"w{i}"), src=SrcKeyword.IMM, lit=Immediate(i)) for i in range(6)]
    block = BasicBlockNode(insts=[inst_r] + writes)
    
    pass_obj = DeadWriteEliminationPass()
    pass_obj.process([block], ctx)
    
    assert len(block.insts) == 6
    for i in range(6):
        assert block.insts[i] == writes[i]

def test_r_wave_read_dependency(ctx):
    # REG_WR w0 #1
    # WPORT_WR p0 r_wave @10
    # w0 write must survive because r_wave read depends on w0-w5.
    inst1 = RegWriteInst(dst=Register("w0"), src=SrcKeyword.IMM, lit=Immediate(1))
    inst2 = PortWriteInst(dst=ImmValue(0), src=Register("r_wave"))
    block = BasicBlockNode(insts=[inst1, inst2])
    
    pass_obj = DeadWriteEliminationPass()
    pass_obj.process([block], ctx)
    
    assert len(block.insts) == 2
    assert block.insts[0] == inst1
    assert block.insts[1] == inst2
