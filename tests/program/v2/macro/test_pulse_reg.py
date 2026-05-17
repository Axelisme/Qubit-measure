"""Tests for macro/pluse_reg.py: PulseByReg."""

from __future__ import annotations

from unittest.mock import MagicMock

from qick.asm_v2 import AsmInst
from zcu_tools.program.v2.macro.pluse_reg import PulseByReg


def _make_prog(tproc_ch=0):
    prog = MagicMock()
    prog._get_reg.side_effect = lambda name: name
    prog.soccfg = {"gens": [{} for _ in range(4)]}
    prog.soccfg["gens"][0]["tproc_ch"] = tproc_ch
    return prog


def _make_macro_with_t_reg(addr_regs, ch=0, imm_t=None, reg_t=None):
    macro = PulseByReg(ch=ch, addr_regs=addr_regs, t=0.0)
    if imm_t is not None:
        macro.t_regs = {"t": imm_t}
    elif reg_t is not None:
        macro.t_regs = {"t": reg_t}
        macro.set_timereg = MagicMock()
    return macro


# ---------------------------------------------------------------------------
# Single address register (no flat_top)
# ---------------------------------------------------------------------------


def test_pulse_by_reg_single_addr_immediate_time():
    prog = _make_prog(tproc_ch=2)
    macro = _make_macro_with_t_reg(addr_regs=["r0"], imm_t=5)
    result = macro.expand(prog)

    assert len(result) == 1
    inst = result[0]
    assert isinstance(inst, AsmInst)
    assert inst.inst["CMD"] == "WPORT_WR"
    assert inst.inst["DST"] == "2"
    assert inst.inst["SRC"] == "wmem"
    assert inst.inst["ADDR"] == "&r0"
    assert inst.inst["TIME"] == "@5"


def test_pulse_by_reg_single_addr_register_time():
    prog = _make_prog(tproc_ch=3)
    macro = _make_macro_with_t_reg(addr_regs=["r1"], reg_t="t_reg")
    result = macro.expand(prog)

    # register time: set_timereg prepended + 1 WPORT_WR without TIME field
    assert len(result) >= 1
    wport_insts = [x for x in result if isinstance(x, AsmInst) and x.inst["CMD"] == "WPORT_WR"]
    assert len(wport_insts) == 1
    assert "TIME" not in wport_insts[0].inst


# ---------------------------------------------------------------------------
# Multiple address registers (flat_top style)
# ---------------------------------------------------------------------------


def test_pulse_by_reg_three_addr_regs_emits_three_wport_wr():
    prog = _make_prog(tproc_ch=1)
    macro = _make_macro_with_t_reg(addr_regs=["r0", "r1", "r2"], imm_t=10)
    result = macro.expand(prog)

    assert len(result) == 3
    for inst in result:
        assert inst.inst["CMD"] == "WPORT_WR"
        assert inst.inst["TIME"] == "@10"


def test_pulse_by_reg_addr_regs_have_correct_addrs():
    prog = _make_prog(tproc_ch=0)
    macro = _make_macro_with_t_reg(addr_regs=["r0", "r5", "r7"], imm_t=0)
    result = macro.expand(prog)

    addrs = [inst.inst["ADDR"] for inst in result]
    assert addrs == ["&r0", "&r5", "&r7"]
