"""Tests for macro/write_reg.py: format_alu_op and WriteRegOp."""

from __future__ import annotations

import pytest
from qick.asm_v2 import AsmInst
from zcu_tools.program.v2.macro.write_reg import WriteRegOp, format_alu_op

# ---------------------------------------------------------------------------
# format_alu_op
# ---------------------------------------------------------------------------


def test_format_alu_op_none_rhs_returns_lhs(mock_prog):
    result = format_alu_op(mock_prog, "r0", "+", None)
    assert result == "r0"


def test_format_alu_op_int_rhs(mock_prog):
    result = format_alu_op(mock_prog, "r0", "+", 5)
    assert result == "r0 + #5"


def test_format_alu_op_str_rhs(mock_prog):
    result = format_alu_op(mock_prog, "r0", "-", "r1")
    assert result == "r0 - r1"


def test_format_alu_op_invalid_rhs_raises(mock_prog):
    with pytest.raises(RuntimeError, match="invalid rhs"):
        format_alu_op(mock_prog, "r0", "+", 3.14)  # type: ignore[arg-type]


def test_format_alu_op_resolves_via_get_reg(mock_prog):
    mock_prog._get_reg.side_effect = lambda name: f"mapped_{name}"
    result = format_alu_op(mock_prog, "r0", "+", "r1")
    assert result == "mapped_r0 + mapped_r1"


# ---------------------------------------------------------------------------
# WriteRegOp.expand
# ---------------------------------------------------------------------------


def test_write_reg_op_expand_int_rhs(mock_prog):
    macro = WriteRegOp(dst="r0", lhs="r1", op="+", rhs=3)
    insts = macro.expand(mock_prog)

    assert len(insts) == 1
    inst = insts[0]
    assert isinstance(inst, AsmInst)
    assert inst.inst["CMD"] == "REG_WR"
    assert inst.inst["DST"] == "r0"
    assert inst.inst["SRC"] == "op"
    assert inst.inst["OP"] == "r1 + #3"


def test_write_reg_op_expand_str_rhs(mock_prog):
    macro = WriteRegOp(dst="r2", lhs="r0", op="-", rhs="r1")
    insts = macro.expand(mock_prog)

    assert len(insts) == 1
    assert insts[0].inst["OP"] == "r0 - r1"
    assert insts[0].inst["DST"] == "r2"


def test_write_reg_op_expand_none_rhs_plain_copy(mock_prog):
    macro = WriteRegOp(dst="r2", lhs="r0", op="+", rhs=None)
    insts = macro.expand(mock_prog)

    assert len(insts) == 1
    assert insts[0].inst["OP"] == "r0"
    assert insts[0].inst["DST"] == "r2"


def test_write_reg_op_addr_inc_is_one(mock_prog):
    macro = WriteRegOp(dst="r0", lhs="r0", op="+", rhs=1)
    insts = macro.expand(mock_prog)
    assert insts[0].addr_inc == 1
