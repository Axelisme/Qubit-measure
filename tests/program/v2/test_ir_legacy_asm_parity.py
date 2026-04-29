"""Compatibility smoke tests after legacy path removal."""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import pytest
from zcu_tools.program.v2 import make_mock_soccfg
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Repeat, SoftRepeat
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadout, DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

ModuleFactory = Callable[[], Sequence[Module]]


def _make_prog(modules: Sequence[Module], *, n_gens: int = 2, n_readouts: int = 1):
    soccfg = make_mock_soccfg(n_gens=n_gens, n_readouts=n_readouts)
    cfg = ProgramV2Cfg()
    return ModularProgramV2(soccfg, cfg, modules=modules)


def _pulse_cfg(
    *,
    ch: int = 0,
    freq: float = 1000.0,
    gain: float = 0.5,
    nqz: "Literal[1, 2]" = 1,
    length: float = 0.5,
    pre_delay: float = 0.0,
    post_delay: float = 0.0,
) -> PulseCfg:
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=ch,
        nqz=nqz,
        freq=freq,
        gain=gain,
        pre_delay=pre_delay,
        post_delay=post_delay,
    )


def _compile_asm(monkeypatch: pytest.MonkeyPatch, module_factory: ModuleFactory) -> str:
    monkeypatch.delenv("ZCU_TOOLS_USE_IR", raising=False)
    prog = _make_prog(module_factory())
    return prog.asm()


def _case_simple_pulse_readout() -> Sequence[Module]:
    return [
        Pulse("drive", _pulse_cfg(ch=0, pre_delay=0.1, post_delay=0.1)),
        DirectReadout(
            "ro",
            DirectReadoutCfg(
                type="readout/direct",
                ro_ch=0,
                ro_length=1.0,
                ro_freq=100.0,
                trig_offset=0.2,
            ),
        ),
    ]


def _case_empty() -> Sequence[Module]:
    return []


def _case_repeat_softdelay() -> Sequence[Module]:
    rep = Repeat("rep", 3)
    rep.add_content(SoftDelay("d", 0.1))
    return [rep]


def _case_softrepeat_softdelay() -> Sequence[Module]:
    rep = SoftRepeat("srep", 4)
    rep.add_content(SoftDelay("d", 0.1))
    return [rep]


@pytest.mark.parametrize(
    "module_factory",
    [
        _case_empty,
        _case_simple_pulse_readout,
        _case_repeat_softdelay,
        _case_softrepeat_softdelay,
    ],
)
def test_ir_asm_compiles(
    monkeypatch: pytest.MonkeyPatch,
    module_factory: ModuleFactory,
) -> None:
    asm = _compile_asm(monkeypatch, module_factory=module_factory)
    assert isinstance(asm, str)
    assert len(asm) > 0


def test_ir_asm_harness_compiles(monkeypatch: pytest.MonkeyPatch) -> None:
    asm = _compile_asm(monkeypatch, module_factory=_case_empty)
    assert isinstance(asm, str)
