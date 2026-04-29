"""Simple tests for leaf module lower() implementation."""

from __future__ import annotations

import pytest
from qick.asm_v2 import QickParam

from .modules.pulse import Pulse, PulseCfg
from .modules.delay import Delay, SoftDelay, DelayAuto
from .modules.readout import DirectReadout, DirectReadoutCfg
from .modules.reset import NoneReset, NoneResetCfg
from .lower import LowerCtx, NameAllocator
from .ir import IRPulse, IRDelay, IRReadout, IRSoftDelay, IRSeq


def test_pulse_lower() -> None:
    """Test Pulse.lower() produces IRPulse."""
    from .modules.waveform import ConstWaveformCfg

    pulse_cfg = PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=0.04),
        ch=0,
        nqz=1,
        freq=100.0,
        gain=1.0,
        pre_delay=0.01,
        post_delay=0.02,
    )
    pulse = Pulse(name="test_pulse", cfg=pulse_cfg)
    pulse.pulse_id = "test_pulse_id"

    ctx = LowerCtx(prog=None, name_alloc=NameAllocator())  # type: ignore
    ir_node = pulse.lower(ctx)

    assert isinstance(ir_node, IRPulse)
    assert ir_node.ch == "0"
    assert ir_node.pulse_name == "test_pulse_id"
    assert ir_node.pre_delay == 0.01
    assert ir_node.advance == 0.07


def test_delay_lower() -> None:
    """Test Delay.lower() produces IRDelay."""
    delay = Delay(name="test_delay", delay=0.05, tag=None)

    ctx = LowerCtx(prog=None, name_alloc=NameAllocator())  # type: ignore
    ir_node = delay.lower(ctx)

    assert isinstance(ir_node, IRDelay)
    assert ir_node.duration == 0.05
    assert ir_node.auto is False
    assert ir_node.tag is None


def test_soft_delay_lower() -> None:
    """Test SoftDelay.lower() preserves timeline-only duration."""
    soft_delay = SoftDelay(name="test_soft_delay", delay=0.05)

    ctx = LowerCtx(prog=None, name_alloc=NameAllocator())  # type: ignore
    ir_node = soft_delay.lower(ctx)

    assert isinstance(ir_node, IRSoftDelay)
    assert ir_node.duration == 0.05


def test_delay_auto_lower() -> None:
    """Test DelayAuto.lower() produces IRDelay with auto=True."""
    delay_auto = DelayAuto(name="test_delay_auto", t=0.05, gens=True, ros=True, tag=None)

    ctx = LowerCtx(prog=None, name_alloc=NameAllocator())  # type: ignore
    ir_node = delay_auto.lower(ctx)

    assert isinstance(ir_node, IRDelay)
    assert ir_node.duration == 0.05
    assert ir_node.auto is True
    assert ir_node.gens is True
    assert ir_node.ros is True
    assert ir_node.tag is None


def test_delay_auto_register_lower() -> None:
    """Test DelayAuto.lower() with register-based duration."""
    delay_auto = DelayAuto(name="test_delay_auto_reg", t="time_reg", gens=True, ros=True)

    ctx = LowerCtx(prog=None, name_alloc=NameAllocator())  # type: ignore
    ir_node = delay_auto.lower(ctx)

    assert isinstance(ir_node, IRDelay)
    assert ir_node.duration == "time_reg"
    assert ir_node.auto is True
    assert ir_node.gens is True
    assert ir_node.ros is True


def test_direct_readout_lower() -> None:
    """Test DirectReadout.lower() produces IRReadout."""
    ro_cfg = DirectReadoutCfg(
        type="readout/direct",
        ro_ch=0,
        ro_length=0.002,
        ro_freq=50.0,
        trig_offset=0.001,
        gen_ch=None,
    )
    readout = DirectReadout(name="test_readout", cfg=ro_cfg)

    ctx = LowerCtx(prog=None, name_alloc=NameAllocator())  # type: ignore
    ir_node = readout.lower(ctx)

    assert isinstance(ir_node, IRReadout)
    assert ir_node.ch == "0"
    assert ir_node.ro_chs == ("0",)
    assert ir_node.pulse_name == "test_readout"
    assert ir_node.trig_offset == 0.001


def test_qickparam_preserved_in_lower() -> None:
    """Test lower() keeps QickParam instead of coercing to 0.0."""
    param = QickParam(start=0.25, spans={})

    delay = Delay(name="delay_param", delay=param, tag=None)
    delay_ctx = LowerCtx(prog=None, name_alloc=NameAllocator())  # type: ignore
    delay_ir = delay.lower(delay_ctx)
    assert isinstance(delay_ir, IRDelay)
    assert delay_ir.duration is param

    delay_auto = DelayAuto(name="delay_auto_param", t=param, gens=False, ros=True)
    auto_ctx = LowerCtx(prog=None, name_alloc=NameAllocator())  # type: ignore
    auto_ir = delay_auto.lower(auto_ctx)
    assert isinstance(auto_ir, IRDelay)
    assert auto_ir.duration is param
    assert auto_ir.gens is False
    assert auto_ir.ros is True


def test_none_reset_lower() -> None:
    """Test NoneReset.lower() produces empty IRSeq."""
    reset_cfg = NoneResetCfg(type="reset/none")
    reset = NoneReset(name="test_reset", cfg=reset_cfg)

    ctx = LowerCtx(prog=None, name_alloc=NameAllocator())  # type: ignore
    ir_node = reset.lower(ctx)

    assert isinstance(ir_node, IRSeq)
    assert len(ir_node.body) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
