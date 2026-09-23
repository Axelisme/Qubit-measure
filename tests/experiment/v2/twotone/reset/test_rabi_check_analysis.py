from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import QickParam
from zcu_tools.experiment.v2.twotone.reset.rabi_check import (
    RabiCheckExp,
    RabiCheckModuleCfg,
    RabiCheckResult,
    _rabi_check_sequence,
)
from zcu_tools.program.v2 import (
    Branch,
    DirectReadoutCfg,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    SweepCfg,
)
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules.registry import PulseRegistry
from zcu_tools.program.v2.modules.reset import NoneResetCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def test_reset_check_uses_same_swept_rabi_pulse_twice() -> None:
    modules = RabiCheckModuleCfg(
        rabi_pulse=PulseCfg(
            waveform=ConstWaveformCfg(length=0.1),
            ch=0,
            nqz=1,
            freq=4000.0,
            gain=0.5,
        ),
        tested_reset=NoneResetCfg(),
        readout=DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=6000.0),
    )
    sweep = SweepCfg(start=0.1, stop=0.7, step=0.1, expts=7)

    sequence = _rabi_check_sequence(modules, sweep)

    assert "pi_pulse" not in RabiCheckModuleCfg.model_fields
    first = sequence[1]
    branch = sequence[2]
    assert isinstance(first, Pulse)
    assert isinstance(branch, Branch)
    assert len(branch.branches) == 3
    assert [len(case) for case in branch.branches] == [0, 1, 2]
    second = branch.branches[2][1]
    assert isinstance(second, Pulse)
    assert first.name != second.name
    assert first.cfg is not None and second.cfg is not None
    assert isinstance(first.cfg.gain, QickParam)
    assert isinstance(second.cfg.gain, QickParam)
    assert first.cfg.gain.start == second.cfg.gain.start == sweep.start
    assert first.cfg.gain.spans == second.cfg.gain.spans == {"gain": 0.6}
    registry = PulseRegistry()
    assert registry.calc_name(first.cfg) == registry.calc_name(second.cfg)

    program = ModularProgramV2(
        make_mock_soccfg(n_gens=1, n_readouts=1),
        ProgramV2Cfg(),
        modules=sequence,
        sweep=[("reset_sel", 3), ("gain", sweep)],
    )
    assert program.pulse_registry.count == 1


def test_analyze_shows_live_sweep_branches_with_legend() -> None:
    gains = np.array([0.0, 0.5, 1.0])
    signals = np.array(
        [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]],
        dtype=np.complex128,
    )
    result = RabiCheckResult(gains=gains, signals=signals)

    figure = RabiCheckExp().analyze(result)
    try:
        ax = figure.axes[0]
        assert ax.get_xlabel() == "Pulse gain"
        assert ax.get_ylabel() == "Amplitude"
        legend = ax.get_legend()
        assert legend is not None
        assert [item.get_text() for item in legend.get_texts()] == [
            "Without Tested Reset",
            "With Tested Reset",
            "Tested Reset + Rabi Pulse",
        ]
        assert len(ax.lines) == 3
        for index, line in enumerate(ax.lines):
            np.testing.assert_array_equal(line.get_xdata(), gains)
            np.testing.assert_allclose(
                np.asarray(line.get_ydata(), dtype=np.float64), signals[index].real
            )
    finally:
        plt.close(figure)
