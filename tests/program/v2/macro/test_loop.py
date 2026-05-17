"""Tests for macro/loop.py: _needs_big_jump, _emit_cond_jump, OpenInnerLoop, CloseInnerLoop."""

from __future__ import annotations

from qick.asm_v2 import AsmInst, Label, WriteLabel, WriteReg
from zcu_tools.program.v2.macro.loop import (
    CloseInnerLoop,
    OpenInnerLoop,
    _emit_cond_jump,
    _needs_big_jump,
)
from zcu_tools.program.v2.macro.meta import MetaMacro
from zcu_tools.program.v2.macro.write_reg import WriteRegOp

# ---------------------------------------------------------------------------
# _needs_big_jump
# ---------------------------------------------------------------------------


def test_needs_big_jump_small_pmem_returns_false(mock_prog):
    mock_prog.tproccfg = {"pmem_size": 512}
    assert _needs_big_jump(mock_prog) is False


def test_needs_big_jump_boundary_2048_returns_false(mock_prog):
    mock_prog.tproccfg = {"pmem_size": 2048}
    assert _needs_big_jump(mock_prog) is False


def test_needs_big_jump_large_pmem_returns_true(large_pmem_prog):
    assert _needs_big_jump(large_pmem_prog) is True


# ---------------------------------------------------------------------------
# _emit_cond_jump
# ---------------------------------------------------------------------------


def test_emit_cond_jump_small_pmem_single_inst(mock_prog):
    result = _emit_cond_jump(mock_prog, label="loop_end", if_cond="Z", op="r0 - #0")
    assert len(result) == 1
    inst = result[0]
    assert isinstance(inst, AsmInst)
    assert inst.inst["CMD"] == "JUMP"
    assert inst.inst["IF"] == "Z"
    assert inst.inst["LABEL"] == "loop_end"
    assert "ADDR" not in inst.inst


def test_emit_cond_jump_large_pmem_two_insts(large_pmem_prog):
    result = _emit_cond_jump(
        large_pmem_prog, label="loop_end", if_cond="S", op="r0 - r1"
    )
    assert len(result) == 2
    assert isinstance(result[0], WriteLabel)
    assert result[0].label == "loop_end"
    assert isinstance(result[1], AsmInst)
    assert result[1].inst["ADDR"] == "s15"
    assert result[1].inst["IF"] == "S"


# ---------------------------------------------------------------------------
# OpenInnerLoop.expand — constant n (no guard jump)
# ---------------------------------------------------------------------------


def test_open_inner_loop_const_n_no_guard(mock_prog):
    macro = OpenInnerLoop("lp", counter_reg="r0", n=10)
    result = macro.expand(mock_prog)

    cmds = [type(x).__name__ for x in result]
    # Expected: MetaMacro(LOOP_START), WriteReg(r0=0), Label(start), MetaMacro(LOOP_BODY_START)
    assert cmds == ["MetaMacro", "WriteReg", "Label", "MetaMacro"]

    assert result[0].type == "LOOP_START"
    assert result[0].name == "lp"
    assert isinstance(result[1], WriteReg)
    assert isinstance(result[2], Label)
    assert result[2].label == "lp_start"
    assert result[3].type == "LOOP_BODY_START"


def test_open_inner_loop_const_n_includes_range_hint(mock_prog):
    macro = OpenInnerLoop("lp", counter_reg="r0", n=5, range_hint=(0, 5))
    result = macro.expand(mock_prog)
    assert result[0].info["range_hint"] == (0, 5)


# ---------------------------------------------------------------------------
# OpenInnerLoop.expand — runtime n (has guard jump)
# ---------------------------------------------------------------------------


def test_open_inner_loop_runtime_n_has_guard(mock_prog):
    macro = OpenInnerLoop("lp", counter_reg="r0", n="n_reg")
    result = macro.expand(mock_prog)

    # MetaMacro(LOOP_START), guard_jump(s), WriteReg, Label, MetaMacro(LOOP_BODY_START)
    assert isinstance(result[0], MetaMacro)
    assert result[0].type == "LOOP_START"

    # guard: at small pmem, single AsmInst JUMP with IF=Z targeting lp_end
    # (may be 1 or 2 instructions depending on pmem)
    guard_insts = result[1:-3]
    assert len(guard_insts) >= 1
    last_guard = guard_insts[-1]
    assert isinstance(last_guard, AsmInst)
    assert last_guard.inst["IF"] == "Z"

    # tail: WriteReg, Label, MetaMacro
    assert isinstance(result[-3], WriteReg)
    assert isinstance(result[-2], Label)
    assert result[-2].label == "lp_start"
    assert result[-1].type == "LOOP_BODY_START"


# ---------------------------------------------------------------------------
# CloseInnerLoop.expand
# ---------------------------------------------------------------------------


def _expand_close(prog, name="lp", counter_reg="r0", n=10):
    macro = CloseInnerLoop(name, counter_reg=counter_reg, n=n)
    return macro.expand(prog)


def test_close_inner_loop_structure(mock_prog):
    result = _expand_close(mock_prog)

    # WriteRegOp, MetaMacro(LOOP_BODY_END), cond_jump(s), Label(end), MetaMacro(LOOP_END)
    assert isinstance(result[0], WriteRegOp)
    assert result[0].op == "+"
    assert result[0].rhs == 1

    assert isinstance(result[1], MetaMacro)
    assert result[1].type == "LOOP_BODY_END"

    assert isinstance(result[-2], Label)
    assert result[-2].label == "lp_end"

    assert isinstance(result[-1], MetaMacro)
    assert result[-1].type == "LOOP_END"


def test_close_inner_loop_cond_jump_targets_start(mock_prog):
    result = _expand_close(mock_prog, name="myloop")
    # find the AsmInst JUMP in the middle
    jump = next(x for x in result if isinstance(x, AsmInst) and x.inst["CMD"] == "JUMP")
    assert jump.inst["IF"] == "S"
    assert jump.inst["LABEL"] == "myloop_start"


def test_close_inner_loop_large_pmem_uses_s15(large_pmem_prog):
    result = _expand_close(large_pmem_prog, name="lp")
    writes = [x for x in result if isinstance(x, WriteLabel)]
    assert len(writes) == 1
    assert writes[0].label == "lp_start"

    jump = next(x for x in result if isinstance(x, AsmInst) and x.inst["CMD"] == "JUMP")
    assert jump.inst["ADDR"] == "s15"
