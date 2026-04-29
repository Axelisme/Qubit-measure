"""Integration tests for IR lowering in ModularProgramV2."""

from __future__ import annotations

import pytest

from .lower import Emitter
from .ir import IRDelay, IRMeta, IRPulse, IRReadout, IRSoftDelay
from .modules.pulse import Pulse, PulseCfg
from .modules.reset import NoneReset, NoneResetCfg
from .modules.waveform import ConstWaveformCfg


def test_module_ir_lowering() -> None:
    """Test that modules can lower to IR without crashing."""
    # Create a simple pulse
    pulse_cfg = PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=0.04),
        ch=0,
        nqz=1,
        freq=100.0,
        gain=1.0,
    )
    pulse = Pulse(name="test_pulse", cfg=pulse_cfg)
    pulse.pulse_id = "test_pulse_id"

    # Create a reset module
    reset = NoneReset(name="reset", cfg=NoneResetCfg(type="reset/none"))

    # Both modules should have lower() methods
    assert hasattr(pulse, "lower")
    assert hasattr(reset, "lower")


def test_feature_flag_env(monkeypatch) -> None:
    """ZCU_TOOLS_USE_IR is read per-call so notebooks can toggle live."""
    from .modular import _ir_enabled

    monkeypatch.delenv("ZCU_TOOLS_USE_IR", raising=False)
    assert _ir_enabled() is False
    monkeypatch.setenv("ZCU_TOOLS_USE_IR", "true")
    assert _ir_enabled() is True
    monkeypatch.setenv("ZCU_TOOLS_USE_IR", "0")
    assert _ir_enabled() is False


class _DummyProg:
    def __init__(self) -> None:
        self.calls = []

    def pulse(self, ch, name, t=0, tag=None):
        self.calls.append(("pulse", ch, name, t, tag))

    def send_readoutconfig(self, ch, name, t=0, tag=None):
        self.calls.append(("send_readoutconfig", ch, name, t, tag))

    def trigger(self, ros=None, t=0, **kwargs):
        self.calls.append(("trigger", tuple(ros or []), t))

    def delay_auto(self, t=0, gens=True, ros=True, tag=None):
        self.calls.append(("delay_auto", t, gens, ros, tag))

    def delay_reg_auto(self, time_reg, gens=True, ros=True):
        self.calls.append(("delay_reg_auto", time_reg, gens, ros))

    def delay(self, t, tag=None):
        self.calls.append(("delay", t, tag))

    def label(self, name):
        self.calls.append(("label", name))

    def write_reg_op(self, dst, lhs, op, rhs):
        self.calls.append(("write_reg_op", dst, lhs, op, rhs))

    def cond_jump(self, target, arg1, test, op=None, arg2=None):
        self.calls.append(("cond_jump", target, arg1, test, op, arg2))

    def jump(self, target):
        self.calls.append(("jump", target))

    def open_loop(self, n, name):
        self.calls.append(("open_loop", n, name))

    def close_loop(self):
        self.calls.append(("close_loop",))

    def open_loop_reg(self, n_reg, name):
        self.calls.append(("open_loop_reg", n_reg, name))

    def close_loop_reg(self, name):
        self.calls.append(("close_loop_reg", name))


def test_emitter_preserves_timing_and_readout_steps() -> None:
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    pulse = IRPulse(
        ch="0",
        pulse_name="p0",
        pre_delay=0.1,
        advance=0.7,
        tag="pulse_tag",
        meta=IRMeta(source_module="test.pulse"),
    )
    readout = IRReadout(
        ch="1",
        ro_chs=("1",),
        pulse_name="ro_cfg",
        trig_offset=0.05,
        meta=IRMeta(source_module="test.readout"),
    )
    soft = IRSoftDelay(duration=0.3, meta=IRMeta(source_module="test.soft"))

    t = emitter.emit(pulse, t=1.0)
    t = emitter.emit(readout, t=t)
    t = emitter.emit(soft, t=t)

    assert ("pulse", 0, "p0", 1.1, "pulse_tag") in prog.calls
    assert ("send_readoutconfig", 1, "ro_cfg", 1.7, None) in prog.calls
    assert ("trigger", (1,), 1.75) in prog.calls
    assert t == pytest.approx(2.0)


def test_emitter_forwards_delay_auto_gens_ros() -> None:
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    node = IRDelay(
        duration=0.2,
        auto=True,
        gens=False,
        ros=True,
        tag="d",
        meta=IRMeta(source_module="test.delay"),
    )
    t = emitter.emit(node, t=1.0)

    assert ("delay_auto", 0.2, False, True, "d") in prog.calls
    assert t == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
