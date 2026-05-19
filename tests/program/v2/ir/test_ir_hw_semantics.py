"""Tests for ir/hw_semantics.py: register set constants."""

from __future__ import annotations

from zcu_tools.program.v2.ir.hw_semantics import (
    ADDR_REG,
    GENERAL_REGS,
    STATUS_REG,
    TIMED_BASE_REG,
    USR_TIME_REG,
    VOLATILE_REGS,
    WAVE_REGS,
)
from zcu_tools.program.v2.ir.operands import Register


def test_register_sets_are_disjoint():
    assert WAVE_REGS.isdisjoint(VOLATILE_REGS)
    assert WAVE_REGS.isdisjoint(GENERAL_REGS)
    assert VOLATILE_REGS.isdisjoint(GENERAL_REGS)


def test_special_regs_in_volatile():
    assert TIMED_BASE_REG in VOLATILE_REGS
    assert USR_TIME_REG in VOLATILE_REGS
    assert STATUS_REG in VOLATILE_REGS
    assert ADDR_REG in VOLATILE_REGS


def test_wave_regs_count():
    assert len(WAVE_REGS) == 6


def test_general_regs_count():
    assert len(GENERAL_REGS) == 32


def test_volatile_regs_count():
    assert len(VOLATILE_REGS) == 16


def test_register_is_wave_reg():
    for name in WAVE_REGS:
        assert Register(name).is_wave_reg()


def test_register_is_volatile_reg():
    for name in VOLATILE_REGS:
        assert Register(name).is_volatile_reg()


def test_register_is_general_reg():
    for name in GENERAL_REGS:
        assert Register(name).is_general_reg()


def test_wave_reg_not_general_or_volatile():
    for name in WAVE_REGS:
        r = Register(name)
        assert not r.is_general_reg()
        assert not r.is_volatile_reg()


def test_general_reg_not_wave_or_volatile():
    for name in GENERAL_REGS:
        r = Register(name)
        assert not r.is_wave_reg()
        assert not r.is_volatile_reg()
