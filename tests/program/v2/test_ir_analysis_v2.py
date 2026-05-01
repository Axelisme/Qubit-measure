import pytest
from zcu_tools.program.v2.ir.analysis import instruction_reads, instruction_writes
from zcu_tools.program.v2.ir.instructions import (
    GenericInst,
    JumpInst,
    PortWriteInst,
    RegWriteInst,
    TestInst,
    TimeInst,
)


def test_time_inst_analysis():
    # Literal lit: no reads
    inst = TimeInst(c_op="inc_ref", lit="#100")
    assert instruction_reads(inst) == set()
    assert instruction_writes(inst) == set()

    # Register r1: reads r1
    inst = TimeInst(c_op="inc_ref", r1="s1")
    assert instruction_reads(inst) == {"s1"}
    assert instruction_writes(inst) == set()

def test_test_inst_analysis():
    # Reads registers from OP string
    inst = TestInst(op="s1-s2")
    assert instruction_reads(inst) == {"s1", "s2"}
    assert instruction_writes(inst) == set()

    inst = TestInst(op="r5+#10")
    assert instruction_reads(inst) == {"r5"}
    assert instruction_writes(inst) == set()

def test_jump_inst_analysis():
    # Jumps read nothing
    inst = JumpInst(label="loop", if_cond="eq")
    assert instruction_reads(inst) == set()
    assert instruction_writes(inst) == set()

def test_reg_write_inst_analysis():
    # imm src: no reads
    inst = RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#42"})
    assert instruction_writes(inst) == {"s1"}
    assert instruction_reads(inst) == set()

    # reg src: reads src register
    inst = RegWriteInst(dst="s1", src="s2")
    assert instruction_writes(inst) == {"s1"}
    assert instruction_reads(inst) == {"s2"}

    # op src: reads registers in OP
    inst = RegWriteInst(dst="s1", src="op", extra_args={"OP": "s2+s3"})
    assert instruction_writes(inst) == {"s1"}
    assert instruction_reads(inst) == {"s2", "s3"}

def test_port_write_inst_analysis():
    # reads registers in TIME and extra_args
    inst = PortWriteInst(dst="p1", time="s1", extra_args={"PHASE": "s2"})
    assert instruction_writes(inst) == {"p1"}
    assert instruction_reads(inst) == {"s1", "s2"}

def test_generic_inst_analysis():
    inst = GenericInst(cmd="UNKNOWN", args={"DST": "s1", "R1": "s2", "OP": "s3+s4"})
    assert instruction_writes(inst) == {"s1"}
    assert instruction_reads(inst) == {"s2", "s3", "s4"}

def test_strip_write_modifier():
    # Test DST with modifiers (though not common in REG_WR DST, good for robustness)
    inst = RegWriteInst(dst="s1 << 16", src="imm", extra_args={"LIT": "#1"})
    assert instruction_writes(inst) == {"s1"}

def test_mixed_registers():
    # Test complex OP strings
    inst = TestInst(op="s_test + temp_reg_1 - w5")
    assert instruction_reads(inst) == {"s_test", "temp_reg_1", "w5"}

def test_property_types():
    inst = RegWriteInst(dst="s1", src="op", extra_args={"OP": "s2+s3"})
    assert isinstance(inst.reg_read, list)
    assert isinstance(inst.reg_write, list)
    assert inst.reg_read == ["s2", "s3"]
    assert inst.reg_write == ["s1"]

def test_need_label():
    # JumpInst needs label
    inst = JumpInst(label="target")
    assert inst.need_label == "target"

    # JumpInst with special labels should NOT need it as a dependency
    assert JumpInst(label="HERE").need_label is None
    assert JumpInst(label="NEXT").need_label is None

    # RegWriteInst with label (WR_ADDR style)
    inst = RegWriteInst(dst="s1", src="imm", extra_args={"LABEL": "data_table"})
    assert inst.need_label == "data_table"

    # GenericInst with label
    inst = GenericInst(cmd="WR_ADDR", args={"LABEL": "my_label"})
    assert inst.need_label == "my_label"
