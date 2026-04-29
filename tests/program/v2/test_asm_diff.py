"""ASM diff tests for legacy vs IR program body generation."""

from __future__ import annotations

from difflib import unified_diff
import re
from typing import Callable, Optional, Sequence, Union

import pytest
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Branch, Repeat, SoftRepeat
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadout, DirectReadoutCfg
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from .conftest import make_mock_soccfg

ModuleFactory = Callable[[], Sequence[Module]]
SweepSpec = Optional[list[tuple[str, Union[int, SweepCfg]]]]


def _compile_asm(
    monkeypatch: pytest.MonkeyPatch,
    modules: Sequence[Module],
    *,
    use_ir: bool,
    sweep: SweepSpec = None,
) -> str:
    monkeypatch.setenv("ZCU_TOOLS_USE_IR", "1" if use_ir else "0")
    prog = ModularProgramV2(
        make_mock_soccfg(),
        ProgramV2Cfg(),
        modules=modules,
        sweep=sweep,
    )
    return prog.asm()


def _normalize_branch_labels(asm: str) -> str:
    """Normalize branch label namespace so legacy/IR naming schemes can compare."""
    asm = re.sub(r"irb\d+_(l|e)_(\d+)_(\d+)", r"branch_\1_\2_\3", asm)
    asm = re.sub(r"[A-Za-z0-9]+_branch_(l|e)_(\d+)_(\d+)", r"branch_\1_\2_\3", asm)
    return asm


def _assert_no_asm_diff(
    monkeypatch: pytest.MonkeyPatch,
    module_factory: ModuleFactory,
    *,
    sweep: SweepSpec = None,
) -> None:
    legacy_asm = _normalize_branch_labels(
        _compile_asm(monkeypatch, module_factory(), use_ir=False, sweep=sweep)
    )
    ir_asm = _normalize_branch_labels(
        _compile_asm(monkeypatch, module_factory(), use_ir=True, sweep=sweep)
    )
    if ir_asm != legacy_asm:
        diff = "\n".join(
            unified_diff(
                legacy_asm.splitlines(),
                ir_asm.splitlines(),
                fromfile="legacy",
                tofile="ir",
                lineterm="",
            )
        )
        pytest.fail(f"ASM mismatch between legacy and IR paths:\n{diff}")


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
def test_asm_diff_parity(
    monkeypatch: pytest.MonkeyPatch,
    module_factory: ModuleFactory,
    sweep: SweepSpec,
) -> None:
    """Compile both paths and show unified diff on mismatch."""
    _assert_no_asm_diff(monkeypatch, module_factory, sweep=sweep)
