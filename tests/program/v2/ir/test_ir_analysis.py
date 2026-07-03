from __future__ import annotations

from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    PortWriteInst,
    RegWriteInst,
    TestInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    ImmValue,
    Register,
    SrcKeyword,
)


def test_time_inst_analysis():
    inst = TimeInst(c_op="inc_ref", lit=Immediate(100))
    assert set(inst.reg_read) == {"s14"}
    assert set(inst.reg_write) == {"s14"}

    inst = TimeInst(c_op="inc_ref", r1=Register("s1"))
    assert set(inst.reg_read) == {"s1", "s14"}
    assert set(inst.reg_write) == {"s14"}


def test_test_inst_analysis():
    inst = TestInst(op=AluExpr(Register("s1"), AluOp.SUB, Register("s2")))
    assert set(inst.reg_read) == {"s1", "s2"}
    assert set(inst.reg_write) == set()

    inst = TestInst(op=AluExpr(Register("r5"), AluOp.ADD, Immediate(10)))
    assert set(inst.reg_read) == {"r5"}
    assert set(inst.reg_write) == set()


def test_jump_inst_analysis():
    inst = JumpInst(label=LabelRef(Label("loop")), if_cond="Z")
    assert set(inst.reg_read) == set()
    assert set(inst.reg_write) == set()


def test_reg_write_inst_analysis():
    inst = RegWriteInst(dst=Register("s1"), src=SrcKeyword.IMM, lit=Immediate(42))
    assert set(inst.reg_write) == {"s1"}
    assert set(inst.reg_read) == set()

    inst = RegWriteInst(dst=Register("s1"), src=Register("s2"))
    assert set(inst.reg_write) == {"s1"}
    assert set(inst.reg_read) == {"s2"}

    inst = RegWriteInst(
        dst=Register("s1"),
        src=SrcKeyword.OP,
        op=AluExpr(Register("s2"), AluOp.ADD, Register("s3")),
    )
    assert set(inst.reg_write) == {"s1"}
    assert set(inst.reg_read) == {"s2", "s3"}


def test_port_write_inst_analysis():
    inst = PortWriteInst(dst=ImmValue(2), time=Register("s1"), addr=Register("s2"))
    assert set(inst.reg_write) == set()
    assert set(inst.reg_read) == {"s1", "s2", "s14"}


def test_mixed_registers():
    inst = TestInst(op=AluExpr(Register("s_test"), AluOp.ADD, Register("temp_reg_1")))
    assert set(inst.reg_read) == {"s_test", "temp_reg_1"}


def test_property_types():
    inst = RegWriteInst(
        dst=Register("s1"),
        src=SrcKeyword.OP,
        op=AluExpr(Register("s2"), AluOp.ADD, Register("s3")),
    )
    assert isinstance(inst.reg_read, frozenset)
    assert isinstance(inst.reg_write, frozenset)
    assert inst.reg_read == frozenset({"s2", "s3"})
    assert inst.reg_write == frozenset({"s1"})


def test_need_label():
    inst = JumpInst(label=LabelRef(Label("target")))
    assert str(inst.need_label) == "target"

    assert JumpInst(label=LabelRef("HERE")).need_label is None
    assert JumpInst(label=LabelRef("NEXT")).need_label is None

    inst = RegWriteInst(
        dst=Register("s1"), src=SrcKeyword.IMM, label=LabelRef(Label("data_table"))
    )
    assert str(inst.need_label) == "data_table"
