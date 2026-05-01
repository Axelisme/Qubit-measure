import pytest
from zcu_tools.program.v2.ir.instructions import LabelInst, WaitInst, RegWriteInst
from zcu_tools.program.v2.ir.node import RootNode, InstNode
from zcu_tools.program.v2.ir.linker import IRLinker

def test_linker_wait_address_calculation():
    """Verify that IRLinker correctly handles WAIT with addr_inc=2."""
    
    # Sequence:
    # L1:
    #   REG_WR
    # L2:
    #   WAIT
    # L3:
    #   REG_WR
    # L4:
    
    ir = RootNode(insts=[
        InstNode(LabelInst(name="L1")),
        InstNode(RegWriteInst(dst="r1", src="imm", extra_args={"LIT": "#1"})),
        InstNode(LabelInst(name="L2")),
        InstNode(WaitInst()),
        InstNode(LabelInst(name="L3")),
        InstNode(RegWriteInst(dst="r2", src="imm", extra_args={"LIT": "#2"})),
        InstNode(LabelInst(name="L4")),
    ])
    
    linker = IRLinker()
    prog_list, labels = linker.link(ir)
    
    # Expected addresses:
    # L1: 0
    # REG_WR (at 0): occupies 1 word -> next addr: 1
    # L2: 1
    # WAIT (at 1): occupies 2 words -> next addr: 3
    # L3: 3
    # REG_WR (at 3): occupies 1 word -> next addr: 4
    # L4: 4
    
    assert labels["L1"] == "&0"
    assert labels["L2"] == "&1"
    assert labels["L3"] == "&3"
    assert labels["L4"] == "&4"
    
    assert prog_list[1]["P_ADDR"] == 1
    assert prog_list[1]["CMD"] == "WAIT"
    assert prog_list[2]["P_ADDR"] == 3

def test_linker_wait_roundtrip():
    """Verify that unlink() correctly restores labels after WAIT."""
    
    ir = RootNode(insts=[
        InstNode(LabelInst(name="L1")),
        InstNode(WaitInst()),
        InstNode(LabelInst(name="L2")),
    ])
    
    linker = IRLinker()
    prog_list, labels = linker.link(ir)
    
    # Roundtrip: unlink
    logical_prog_list = linker.unlink(prog_list, labels)
    
    expected = [
        {"LABEL": "L1"},
        {"CMD": "WAIT"},
        {"LABEL": "L2"},
    ]
    
    # Compare only LABEL and CMD for simplicity
    actual = []
    for item in logical_prog_list:
        if "LABEL" in item and "CMD" not in item:
            actual.append({"LABEL": item["LABEL"]})
        else:
            actual.append({"CMD": item["CMD"]})
            
    assert actual == expected

if __name__ == "__main__":
    pytest.main([__file__])
