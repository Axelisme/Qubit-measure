"""Integration tests for IR builder, emitter, and feature flags."""

from __future__ import annotations

import pytest

from zcu_tools.program.v2.ir import (
    IRBranch,
    IRDelay,
    IRDelayAuto,
    IRMeta,
    IRNop,
    IRPulse,
    IRReadout,
    IRSendReadoutConfig,
    IRSeq,
)
from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.lower import Emitter
from zcu_tools.program.v2.modules.pulse import Pulse
from zcu_tools.program.v2.modules.reset import NoneReset, NoneResetCfg


def test_default_pipeline_is_available() -> None:
    from zcu_tools.program.v2.ir.passes import make_default_pipeline

    pipeline = make_default_pipeline()
    assert pipeline is not None


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

    emitter.emit(IRSendReadoutConfig(ch="1", pulse_name="ro_cfg", t=0.0))
    emitter.emit(IRReadout(ch="1", ro_chs=("1",), pulse_name="ro_cfg", t=0.5))

    assert ("send_readoutconfig", 1, "ro_cfg", 0.0) in prog.calls
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
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    node = IRSeq(
        body=(
            IRPulse(ch="0", pulse_name="p1", t=0.0),
            IRDelay(t=1.0),
            IRPulse(ch="0", pulse_name="p2", t=0.0),
        )
    )
    emitter.emit(node)

    names = [c[2] for c in prog.calls if c[0] == "pulse"]
    assert names == ["p1", "p2"]
    assert ("delay", 1.0, None) in prog.calls


def test_emitter_branch_binary_dispatch() -> None:
    prog = _DummyProg()
    emitter = Emitter(prog)  # type: ignore[arg-type]

    node = IRBranch(
        compare_reg="sel",
        arms=(
            IRSeq(body=(IRNop(), IRDelay(0.1))),
            IRSeq(body=(IRDelay(0.2),)),
            IRSeq(body=(IRDelay(0.3),)),
        ),
    )
    emitter.emit(node)

    cond_calls = [c for c in prog.calls if c[0] == "cond_jump"]
    assert len(cond_calls) == 2
    mids = sorted(c[5] for c in cond_calls)
    assert mids == [1, 2]
    assert all(c[2] == "sel" for c in cond_calls)

    # Branch labels must be unique and deterministic per branch node.
    labels = [c[1] for c in prog.calls if c[0] == "label"]
    assert "irb0_l_0_1" in labels
    assert "irb0_l_1_2" in labels


# ---------------------------------------------------------------------------
# IRBuilder tests
# ---------------------------------------------------------------------------


def test_builder_scope_loop() -> None:
    from zcu_tools.program.v2.ir.nodes import IRLoop, IRSeq

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
    from zcu_tools.program.v2.ir.nodes import IRBranch

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
