"""Tests for macro/delay.py: DelayRegAuto."""

from __future__ import annotations

from unittest.mock import MagicMock

from qick.asm_v2 import AsmInst
from zcu_tools.program.v2.macro.delay import DelayRegAuto


def _make_delay_auto(time_reg="r0", gens=True, ros=True):
    return DelayRegAuto(time_reg=time_reg, gens=gens, ros=ros)


def _make_prog_with_auto_t(auto_t_value, time_reg="r0"):
    prog = MagicMock()
    prog._get_reg.side_effect = lambda name: name
    macro = _make_delay_auto(time_reg=time_reg)
    macro.t_regs = {"auto_t": auto_t_value}
    return prog, macro


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_calls_get_reg(mock_prog):
    macro = _make_delay_auto(time_reg="r0")
    mock_prog.get_max_timestamp.return_value = 0.0
    macro.convert_time = MagicMock(return_value=0)
    macro.preprocess(mock_prog)
    mock_prog._get_reg.assert_called_with("r0")


def test_preprocess_calls_decrement_timestamps(mock_prog):
    macro = _make_delay_auto()
    mock_prog.get_max_timestamp.return_value = 1.5
    macro.convert_time = MagicMock(return_value=10)
    macro.preprocess(mock_prog)
    mock_prog.decrement_timestamps.assert_called_once_with(10)


# ---------------------------------------------------------------------------
# expand: auto_t is an integer literal → TIME #N + TIME R
# ---------------------------------------------------------------------------


def test_expand_integer_auto_t_emits_two_time_insts():
    prog, macro = _make_prog_with_auto_t(auto_t_value=7)
    result = macro.expand(prog)

    assert len(result) == 2
    first, second = result
    assert isinstance(first, AsmInst)
    assert first.inst["CMD"] == "TIME"
    assert first.inst["C_OP"] == "inc_ref"
    assert first.inst["LIT"] == "#7"

    assert isinstance(second, AsmInst)
    assert second.inst["CMD"] == "TIME"
    assert second.inst["R1"] == "r0"


def test_expand_zero_integer_auto_t_still_emits_lit_inst():
    prog, macro = _make_prog_with_auto_t(auto_t_value=0)
    result = macro.expand(prog)
    assert result[0].inst["LIT"] == "#0"


# ---------------------------------------------------------------------------
# expand: auto_t is a register name → TIME R_auto + TIME R_time
# ---------------------------------------------------------------------------


def test_expand_register_auto_t_emits_two_time_insts():
    prog, macro = _make_prog_with_auto_t(auto_t_value="s3")
    macro.set_timereg = MagicMock()
    result = macro.expand(prog)

    assert len(result) == 2
    first, second = result
    assert first.inst["R1"] == "s3"
    assert second.inst["R1"] == "r0"


# ---------------------------------------------------------------------------
# expand: auto_t is None → only TIME R_time
# ---------------------------------------------------------------------------


def test_expand_none_auto_t_emits_one_time_inst():
    prog, macro = _make_prog_with_auto_t(auto_t_value=None)
    result = macro.expand(prog)

    assert len(result) == 1
    assert result[0].inst["CMD"] == "TIME"
    assert result[0].inst["R1"] == "r0"
