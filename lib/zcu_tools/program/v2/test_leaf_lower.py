"""Tests for leaf module ir_run() implementation (Phase 1R)."""

from __future__ import annotations

import pytest
from qick.asm_v2 import QickParam

from .ir.builder import IRBuilder
from .ir.nodes import IRDelay, IRDelayAuto, IRReadout, IRSeq
from .modules.delay import Delay, DelayAuto, SoftDelay
from .modules.readout import DirectReadout, DirectReadoutCfg
from .modules.reset import NoneReset, NoneResetCfg

# Modules that don't use prog in ir_run may receive None safely.
_NO_PROG = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Delay / SoftDelay / DelayAuto
# ---------------------------------------------------------------------------


def test_delay_emits_ir_delay() -> None:
    delay = Delay(name="d", delay=0.05)
    b = IRBuilder()
    next_t = delay.ir_run(b, t=1.0, prog=_NO_PROG)  # type: ignore[call-arg]

    assert next_t == 0.0
    root = b.build()
    assert isinstance(root, IRDelay)
    assert root.t == pytest.approx(0.05)
    assert root.tag is None


def test_delay_with_tag() -> None:
    delay = Delay(name="d", delay=0.05, tag="barrier")
    b = IRBuilder()
    delay.ir_run(b, t=0.0, prog=_NO_PROG)  # type: ignore[call-arg]
    root = b.build()
    assert isinstance(root, IRDelay)
    assert root.tag == "barrier"


def test_soft_delay_no_ir_emitted() -> None:
    soft = SoftDelay(name="s", delay=0.05)
    b = IRBuilder()
    next_t = soft.ir_run(b, t=1.0, prog=_NO_PROG)  # type: ignore[call-arg]

    assert next_t == pytest.approx(1.05)
    root = b.build()
    # Nothing emitted → builder returns empty IRSeq
    assert isinstance(root, IRSeq)
    assert len(root.body) == 0


def test_delay_auto_emits_ir_delay_auto() -> None:
    da = DelayAuto(name="da", t=0.0, gens=True, ros=True)
    b = IRBuilder()
    next_t = da.ir_run(b, t=1.0, prog=_NO_PROG)  # type: ignore[call-arg]

    assert next_t == 0.0
    root = b.build()
    assert isinstance(root, IRDelayAuto)
    assert root.t == pytest.approx(0.0)
    assert root.gens is True
    assert root.ros is True


def test_delay_auto_gens_false() -> None:
    da = DelayAuto(name="da", t=0.5, gens=False, ros=True)
    b = IRBuilder()
    da.ir_run(b, t=0.0, prog=_NO_PROG)  # type: ignore[call-arg]
    root = b.build()
    assert isinstance(root, IRDelayAuto)
    assert root.t == pytest.approx(0.5)
    assert root.gens is False


def test_delay_auto_register_based() -> None:
    da = DelayAuto(name="da", t="time_reg", gens=True, ros=True)
    b = IRBuilder()
    da.ir_run(b, t=0.0, prog=_NO_PROG)  # type: ignore[call-arg]
    root = b.build()
    assert isinstance(root, IRDelayAuto)
    assert root.t == "time_reg"


def test_qickparam_preserved_in_delay() -> None:
    param = QickParam(start=0.25, spans={})
    delay = Delay(name="d", delay=param)
    b = IRBuilder()
    delay.ir_run(b, t=0.0, prog=_NO_PROG)  # type: ignore[call-arg]
    root = b.build()
    assert isinstance(root, IRDelay)
    assert root.t is param


def test_qickparam_preserved_in_delay_auto() -> None:
    param = QickParam(start=0.5, spans={"x": 0.1})
    da = DelayAuto(name="da", t=param, gens=False, ros=True)
    b = IRBuilder()
    da.ir_run(b, t=0.0, prog=_NO_PROG)  # type: ignore[call-arg]
    root = b.build()
    assert isinstance(root, IRDelayAuto)
    assert root.t is param
    assert root.gens is False


# ---------------------------------------------------------------------------
# DirectReadout
# ---------------------------------------------------------------------------


def test_direct_readout_ir_run() -> None:
    ro_cfg = DirectReadoutCfg(
        type="readout/direct",
        ro_ch=3,
        ro_length=0.002,
        ro_freq=50.0,
        trig_offset=0.001,
    )
    ro = DirectReadout(name="ro", cfg=ro_cfg)
    b = IRBuilder()
    next_t = ro.ir_run(b, t=0.5, prog=_NO_PROG)  # type: ignore[call-arg]

    # DirectReadout is non-blocking: returns t unchanged
    assert next_t == pytest.approx(0.5)
    root = b.build()
    assert isinstance(root, IRReadout)
    assert root.ch == "3"
    assert root.ro_chs == ("3",)
    assert root.pulse_name == "ro"
    # trig_offset folded into t: 0.5 + 0.001
    assert root.t == pytest.approx(0.501)


# ---------------------------------------------------------------------------
# NoneReset
# ---------------------------------------------------------------------------


def test_none_reset_no_op() -> None:
    reset = NoneReset(name="r", cfg=NoneResetCfg(type="reset/none"))
    b = IRBuilder()
    next_t = reset.ir_run(b, t=2.0, prog=_NO_PROG)  # type: ignore[call-arg]

    assert next_t == pytest.approx(2.0)
    root = b.build()
    assert isinstance(root, IRSeq)
    assert len(root.body) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
