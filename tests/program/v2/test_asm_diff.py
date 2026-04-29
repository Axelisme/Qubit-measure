"""ASM compile smoke tests for representative module graphs."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import pytest
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Branch, Repeat, SoftRepeat
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadout, DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sweep import SweepCfg

from .conftest import make_mock_soccfg

ModuleFactory = Callable[[], Sequence[Module]]
SweepSpec = Optional[list[tuple[str, Union[int, SweepCfg]]]]


def _compile_asm(
    monkeypatch: pytest.MonkeyPatch,
    modules: Sequence[Module],
    *,
    enable_opt: bool,
    sweep: SweepSpec = None,
) -> str:
    monkeypatch.setenv("ZCU_TOOLS_IR_OPT", "1" if enable_opt else "0")
    prog = ModularProgramV2(
        make_mock_soccfg(),
        ProgramV2Cfg(),
        modules=modules,
        sweep=sweep,
    )
    return prog.asm()


def _case_pulse_readout() -> Sequence[Module]:
    return [
        Pulse(
            "drive",
            PulseCfg(
                waveform=ConstWaveformCfg(length=0.4),
                ch=0,
                nqz=1,
                freq=1000.0,
                gain=0.5,
                pre_delay=0.1,
                post_delay=0.1,
            ),
        ),
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


def _case_repeat_softdelay() -> Sequence[Module]:
    rep = Repeat("rep", 3)
    rep.add_content(SoftDelay("d", 0.1))
    return [rep]


def _case_soft_repeat_softdelay() -> Sequence[Module]:
    rep = SoftRepeat("srep", 4)
    rep.add_content(SoftDelay("d", 0.1))
    return [rep]


def _case_branch_softdelay() -> Sequence[Module]:
    branch = Branch(
        "sel",
        [SoftDelay("d0", 0.1)],
        [SoftDelay("d1", 0.2)],
        [SoftDelay("d2", 0.3)],
    )
    return [branch]


@pytest.mark.parametrize(
    "module_factory,sweep",
    [
        (_case_pulse_readout, None),
        (_case_repeat_softdelay, None),
        (_case_soft_repeat_softdelay, None),
        (_case_branch_softdelay, [("sel", 3)]),
    ],
)
def test_asm_compile_with_and_without_opt(
    monkeypatch: pytest.MonkeyPatch,
    module_factory: ModuleFactory,
    sweep: SweepSpec,
) -> None:
    asm_no_opt = _compile_asm(
        monkeypatch, module_factory(), enable_opt=False, sweep=sweep
    )
    asm_opt = _compile_asm(monkeypatch, module_factory(), enable_opt=True, sweep=sweep)
    assert isinstance(asm_no_opt, str)
    assert isinstance(asm_opt, str)
    assert len(asm_no_opt) > 0
    assert len(asm_opt) > 0
