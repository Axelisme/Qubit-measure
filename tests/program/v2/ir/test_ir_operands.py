from typing import Any

import pytest
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    DmemAddr,
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    SideWrite,
    SrcKeyword,
    TimeOffset,
    canonical_reg,
    parse_addr,
    parse_alu_expr,
    parse_imm_value,
    parse_immediate,
    parse_label,
    parse_mem_addr,
    parse_register,
    parse_side_write,
    parse_src,
    parse_time,
    parse_time_offset,
    parse_value,
)


def test_canonical_reg_and_register_properties():
    assert canonical_reg("w_freq") == "w0"
    assert canonical_reg("plain") == "plain"

    assert Register("w_freq").canonical_name == "w0"
    assert Register("w_freq").regs() == frozenset({"w0"})
    assert Register("r_wave").regs() == frozenset({"w0", "w1", "w2", "w3", "w4", "w5"})

    assert Register("r3").is_general_reg() is True
    assert Register("r_cnt").is_general_reg() is False
    assert Register("w2").is_wave_reg() is True
    assert Register("r_wave").is_wave_reg() is True
    assert Register("s14").is_volatile_reg() is True
    assert Register("r3").is_volatile_reg() is False
    assert str(Register("s14")) == "s14"


def test_scalar_operands_and_dmemaddr_behaviour():
    assert ImmValue(7).regs() == frozenset()
    assert Immediate(-3).regs() == frozenset()
    assert TimeOffset(10).regs() == frozenset()
    assert MemAddr(5).regs() == frozenset()
    assert str(ImmValue(7)) == "7"
    assert str(Immediate(-3)) == "#-3"
    assert str(TimeOffset(10)) == "@10"
    assert str(MemAddr(5)) == "&5"

    unresolved = DmemAddr((Label("a"),))
    assert unresolved.regs() == frozenset()
    with pytest.raises(RuntimeError, match="must replace every DmemAddr"):
        str(unresolved)


def test_aluexpr_and_sidewrite_string_and_regs():
    unary = AluExpr(Register("r1"), AluOp.ABS)
    bare = AluExpr(Register("r2"), AluOp.NONE)
    binary = AluExpr(Register("r3"), AluOp.ADD, Immediate(4))

    assert unary.regs() == frozenset({"r1"})
    assert bare.regs() == frozenset({"r2"})
    assert binary.regs() == frozenset({"r3"})
    assert str(unary) == "ABS r1"
    assert str(bare) == "r2"
    assert str(binary) == "r3 + #4"

    side = SideWrite(Register("s2"), "imm")
    assert side.regs() == frozenset({"s2"})
    assert str(side) == "s2 imm"


def test_parse_register_paths():
    invalid: Any = 123
    reg = Register("r5")
    assert parse_register(reg) is reg
    assert parse_register("r_wave") == Register("r_wave")
    assert parse_register("w_freq") == Register("w_freq")
    assert parse_register("&s14") == Register("s14")
    assert parse_register("w3") == Register("w3")
    assert parse_register(invalid) is None
    assert parse_register("s_typo") is None


def test_parse_immediate_paths():
    imm = Immediate(3)
    assert parse_immediate(imm) is imm
    assert parse_immediate(5) == Immediate(5)
    assert parse_immediate("#7") == Immediate(7)
    assert parse_immediate("#u0x10") == Immediate(16)
    assert parse_immediate(None) is None
    assert parse_immediate("7") is None
    assert parse_immediate("#bogus") is None


def test_parse_time_offset_paths():
    offset = TimeOffset(3)
    assert parse_time_offset(offset) is offset
    assert parse_time_offset(8) == TimeOffset(8)
    assert parse_time_offset("@0x10") == TimeOffset(16)
    assert parse_time_offset(None) is None
    assert parse_time_offset("#7") is None
    assert parse_time_offset("@bad") is None


def test_parse_mem_addr_and_imm_value_paths():
    mem = MemAddr(4)
    assert parse_mem_addr(mem) is mem
    assert parse_mem_addr(9) == MemAddr(9)
    assert parse_mem_addr("&0x20") == MemAddr(32)
    assert parse_mem_addr(None) is None
    assert parse_mem_addr("r1") is None
    assert parse_mem_addr("&r1") is None

    bare = ImmValue(11)
    assert parse_imm_value(bare) is bare
    assert parse_imm_value(-5) == ImmValue(-5)
    assert parse_imm_value("12") == ImmValue(12)
    assert parse_imm_value("-7") == ImmValue(-7)
    assert parse_imm_value("0x20") == ImmValue(32)
    assert parse_imm_value(None) is None
    assert parse_imm_value("1.2") is None
    assert parse_imm_value("0xGG") is None


def test_parse_label_paths():
    ref = LabelRef(Label("loop"))
    assert parse_label(ref) is ref
    assert parse_label(Label("target")) == LabelRef(Label("target"))
    assert parse_label("&NEXT") == LabelRef("NEXT")
    assert parse_label("body") == LabelRef(Label("body"))
    assert parse_label(None) is None
    assert parse_label("&") is None


def test_parse_alu_expr_success_paths():
    expr = parse_alu_expr("r1 + #3")
    assert expr == AluExpr(Register("r1"), AluOp.ADD, Immediate(3))

    reg_rhs = parse_alu_expr("r1 AND s2")
    assert reg_rhs == AluExpr(Register("r1"), AluOp.AND, Register("s2"))

    unary = parse_alu_expr("ABS r4")
    assert unary == AluExpr(Register("r4"), AluOp.ABS)

    bare = parse_alu_expr("w1")
    assert bare == AluExpr(Register("w1"), AluOp.NONE)

    existing = AluExpr(Register("r0"), AluOp.SUB, Immediate(1))
    assert parse_alu_expr(existing) is existing
    assert parse_alu_expr(None) is None


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "Invalid ALU expression"),
        ("r1 + #1 + #2", "too many tokens"),
        ("r1 BAD #2", "Unknown ALU operator"),
        ("BAD r1", "Unknown ALU unary operator"),
        ("ABS bad_name", "Cannot parse ALU operand"),
        ("nope", "Cannot parse ALU expression"),
        ("bad + #1", "Cannot parse ALU operands"),
    ],
)
def test_parse_alu_expr_error_paths(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_alu_expr(raw)


def test_parse_side_write_paths():
    invalid: Any = 1
    side = SideWrite(Register("r1"), "imm")
    assert parse_side_write(side) is side
    assert parse_side_write("r1 op") == SideWrite(Register("r1"), "op")
    assert parse_side_write("s2") == SideWrite(Register("s2"), "op")
    assert parse_side_write("   ") is None
    assert parse_side_write(None) is None
    assert parse_side_write(invalid) is None
    assert parse_side_write("bad_name imm") is None


def test_parse_value_paths_and_fallback():
    reg = Register("r1")
    imm = Immediate(2)
    bare = ImmValue(3)

    assert parse_value(None) is None
    assert parse_value(reg) is reg
    assert parse_value(imm) is imm
    assert parse_value(bare) is bare
    assert parse_value(7) == ImmValue(7)
    assert parse_value("#8") == Immediate(8)
    assert parse_value("s14") == Register("s14")
    assert parse_value("9") == ImmValue(9)
    assert parse_value("  custom_name  ") == Register("custom_name")


def test_parse_addr_paths():
    reg = Register("r1")
    mem = MemAddr(6)
    assert parse_addr(None) is None
    assert parse_addr(reg) is reg
    assert parse_addr(mem) is mem
    assert parse_addr("&10") == MemAddr(10)
    assert parse_addr("&r2") == Register("r2")
    assert parse_addr("label_name") is None


def test_parse_time_paths_and_fallback():
    reg = Register("s14")
    offset = TimeOffset(12)
    assert parse_time(None) is None
    assert parse_time(reg) is reg
    assert parse_time(offset) is offset
    assert parse_time(5) == TimeOffset(5)
    assert parse_time("@7") == TimeOffset(7)
    assert parse_time("r1") == Register("r1")
    assert parse_time("custom_time_reg") == Register("custom_time_reg")
    assert parse_time("") is None


def test_parse_src_paths():
    reg = Register("r2")
    assert parse_src(None) is None
    assert parse_src(SrcKeyword.IMM) is SrcKeyword.IMM
    assert parse_src(reg) is reg
    assert parse_src("op") is SrcKeyword.OP
    assert parse_src("r_wave") == Register("r_wave")
    assert parse_src("unknown") is None
