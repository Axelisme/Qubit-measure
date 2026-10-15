"""Public simulator trigger and completed-round stop contracts."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim.engine import (
    SimEngine,
)

from . import _engine_support

_RESET_RELAX_DELAY = _engine_support.RESET_RELAX_DELAY
_SIM = _engine_support.SIM
_rf_g_mhz = _engine_support.ground_resonator_frequency_mhz
_pi_pulse_prog = _engine_support.pi_pulse_program
_f_qubit_mhz = _engine_support.qubit_frequency_mhz
_readout = _engine_support.readout


class _CancelFlag:
    def __init__(self) -> None:
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True


def test_engine_compute_round_supports_multiple_read_triggers() -> None:
    """Multiple triggers produce one IQ pair per rep and trigger slot."""

    prog = _pi_pulse_prog(relax_delay=_RESET_RELAX_DELAY, reps=3)
    ((_, ro),) = prog.ro_chs.items()
    ro["trigs"] = 2

    engine = SimEngine(prog, _SIM)
    acc = engine.compute_round(0)[0]
    assert acc.shape == (3, 2, 2)
    assert acc.dtype == np.int64


def test_acquire_round_hook_cancel_flag_is_checked_after_mock_round() -> None:
    """Acquire-level stop flags keep hardware-like round-boundary semantics."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.1,
        freq=_f_qubit_mhz(),
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=20, rounds=5),
        modules=[pulse, _readout(_rf_g_mhz())],
    )

    def stop_at_round_boundary(_round_count: int, _raw, cancel_flag) -> None:
        cancel_flag.set()

    prog.acquire(
        soc,
        progress=False,
        round_hook=stop_at_round_boundary,
    )

    rounds = prog.get_rounds()
    assert rounds is not None
    assert len(rounds) == 1
    raw = prog.get_raw()
    assert raw is not None
    assert raw[0].shape == (20, 1, 2)


def test_acquire_stop_before_first_round_raises_stopped_partial() -> None:
    """Stopping before any completed round never averages an empty rounds buffer."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=20, rounds=5),
        modules=[_readout(_rf_g_mhz())],
    )
    cancel_flag = _CancelFlag()
    cancel_flag.set()

    with pytest.raises(StoppedPartialAcquireError, match="first round"):
        prog.acquire(soc, progress=False, cancel_flag=cancel_flag)

    assert prog.get_rounds() == []
