"""Constructor overrides preserve cfg routing at the public compile boundary."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.program.v2 import TimelineBuilder, TimelineProgramV2
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


ProgramKind = Literal["native", "builder", "legacy"]


def _compile(kind: ProgramKind, modules: Sequence[Module], swept: bool):
    soccfg = make_mock_soccfg(n_gens=3, n_readouts=1)
    if kind == "legacy":
        return ModularProgramV2(
            soccfg, modules=list(modules), sweep=[("point", 3)] if swept else None
        )
    if kind == "builder":
        builder = TimelineBuilder(soccfg)
        if swept:
            builder.add_loop("point", 3)
        for module in modules:
            builder.emit(module)
        return builder.build()

    class Native(TimelineProgramV2):
        def _initialize(self, cfg):
            if swept:
                self.add_loop("point", 3)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            for module in modules:
                self.emit(module)

    return Native(soccfg)


def _pulse_cfg(ch: int = 0) -> PulseCfg:
    return PulseCfg(
        ch=ch, nqz=1, freq=100, gain=0.2,
        waveform=ConstWaveformCfg(length=0.1),
    )


def _assert_same_compile(actual, expected):
    assert actual.prog_list == expected.prog_list
    np.testing.assert_array_equal(actual.binprog["wmem"], expected.binprog["wmem"])
    np.testing.assert_array_equal(actual.compile_datamem(), expected.compile_datamem())


@pytest.mark.parametrize("kind", ["native", "builder", "legacy"])
@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("knob,start,stop", [
    ("freq", 200, 220), ("gain", 0.3, 0.5),
    ("phase", 30, 50), ("length", 0.2, 0.3),
])
def test_pulse_overrides_match_set_param_without_mutating_cfg(kind, swept, knob, start, stop):
    cfg = _pulse_cfg()
    original = cfg.model_dump()
    value = QickSweep1D("point", start, stop) if swept else start
    equivalent = cfg.model_copy(deep=True)
    equivalent.set_param(knob, value)
    actual = _compile(kind, [Pulse("drive", cfg, **{knob: value})], swept)
    expected = _compile(kind, [Pulse("drive", equivalent)], swept)
    _assert_same_compile(actual, expected)
    assert cfg.model_dump() == original


@pytest.mark.parametrize("kind", ["native", "builder", "legacy"])
def test_pulse_none_overrides_keep_cfg_and_registry_identity(kind):
    cfg = _pulse_cfg()
    actual = _compile(kind, [Pulse("first", cfg, freq=None, gain=None, phase=None, length=None),
                             Pulse("second", cfg, gain=0.4)], False)
    changed = cfg.model_copy(deep=True)
    changed.set_param("gain", 0.4)
    expected = _compile(kind, [Pulse("first", cfg), Pulse("second", changed)], False)
    _assert_same_compile(actual, expected)
