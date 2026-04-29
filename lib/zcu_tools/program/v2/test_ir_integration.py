"""Integration tests for IR builder, emitter, and feature flag (Phase 1R)."""

from __future__ import annotations

import pytest

from .ir import IRDelay, IRDelayAuto, IRMeta, IRPulse, IRReadout
from .ir.builder import IRBuilder
from .lower import Emitter
from .modules.pulse import Pulse
from .modules.reset import NoneReset, NoneResetCfg


def test_feature_flag_env(monkeypatch) -> None:
    """ZCU_TOOLS_USE_IR is read per-call so notebooks can toggle live."""
    from .modular import _ir_enabled

    monkeypatch.delenv("ZCU_TOOLS_USE_IR", raising=False)
    assert _ir_enabled() is False
    monkeypatch.setenv("ZCU_TOOLS_USE_IR", "true")
    assert _ir_enabled() is True
    monkeypatch.setenv("ZCU_TOOLS_USE_IR", "0")
    assert _ir_enabled() is False


def test_module_has_ir_run() -> None:
    reset = NoneReset(name="r", cfg=NoneResetCfg(type="reset/none"))
    assert hasattr(reset, "ir_run")


class _DummyProg:
    def __init__(self) -> None:
        self.calls: list = []

    def pulse(self, ch, name, t=0, tag=None):
        self.calls.append(("pulse", ch, name, t, tag))

    def send_readoutconfig(self, ch, name, t=0):
        self.calls.append(("send_readoutconfig", ch, name, t))

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

    def nop(self):
        self.calls.append(("nop",))

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

    def read_dmem(self, dst, addr):
        self.calls.append(("read_dmem", dst, addr))


# ---------------------------------------------------------------------------
# Emitter tests
# ---------------------------------------------------------------------------


def test_emitter_pulse() -> None:
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    node = IRPulse(ch="0", pulse_name="p0", t=1.1, tag="ptag")
    emitter.emit(node)

    assert ("pulse", 0, "p0", 1.1, "ptag") in prog.calls


def test_emitter_readout_sends_config_and_triggers() -> None:
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    node = IRReadout(ch="1", ro_chs=("1",), pulse_name="ro_cfg", t=0.5)
    emitter.emit(node)

    assert ("send_readoutconfig", 1, "ro_cfg", 0.5) in prog.calls
    assert ("trigger", (1,), 0.5) in prog.calls


def test_emitter_delay() -> None:
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    node = IRDelay(t=0.3, tag="bar")
    emitter.emit(node)

    assert ("delay", 0.3, "bar") in prog.calls


def test_emitter_delay_auto() -> None:
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    node = IRDelayAuto(t=0.0, gens=False, ros=True, tag="da")
    emitter.emit(node)

    assert ("delay_auto", 0.0, False, True, "da") in prog.calls


def test_emitter_delay_auto_register() -> None:
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    node = IRDelayAuto(t="time_reg", gens=True, ros=False)
    emitter.emit(node)

    assert ("delay_reg_auto", "time_reg", True, False) in prog.calls


def test_emitter_seq_order() -> None:
    from .ir.nodes import IRSeq
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    node = IRSeq(body=(
        IRPulse(ch="0", pulse_name="p1", t=0.0),
        IRDelay(t=1.0),
        IRPulse(ch="0", pulse_name="p2", t=0.0),
    ))
    emitter.emit(node)

    names = [c[2] for c in prog.calls if c[0] == "pulse"]
    assert names == ["p1", "p2"]
    assert ("delay", 1.0, None) in prog.calls


# ---------------------------------------------------------------------------
# IRBuilder tests
# ---------------------------------------------------------------------------


def test_builder_scope_loop() -> None:
    from .ir.nodes import IRLoop, IRSeq

    b = IRBuilder()
    b.ir_delay(0.0)
    b.ir_delay_auto(0.0)
    with b.ir_loop("loop0", n=5):
        b.ir_pulse("0", "p", t=0.0)
        b.ir_delay(1.0)
        b.ir_delay_auto(0.0)
    b.ir_delay(0.0)

    root = b.build()
    assert isinstance(root, IRSeq)
    # top-level: delay_auto, loop, delay
    loop = [n for n in root.body if isinstance(n, IRLoop)]
    assert len(loop) == 1
    assert loop[0].n == 5


def test_builder_unclosed_scope_raises() -> None:
    b = IRBuilder()
    b._push()  # manually push without matching pop
    with pytest.raises(RuntimeError, match="unclosed scope"):
        b.build()


def test_builder_branch_arms() -> None:
    from .ir.nodes import IRBranch

    b = IRBuilder()
    with b.ir_branch("reg0") as branch:
        with branch.arm():
            b.ir_delay(1.0)
        with branch.arm():
            b.ir_delay(2.0)

    root = b.build()
    assert isinstance(root, IRBranch)
    assert len(root.arms) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
