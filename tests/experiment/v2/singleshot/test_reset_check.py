from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.v2.runner import StopSignal, schedule_stop_scope
from zcu_tools.experiment.v2.singleshot.amp_rabi import (
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiModuleCfg,
    AmpRabiResult,
    AmpRabiSweepCfg,
)
from zcu_tools.experiment.v2.singleshot.reset_check import (
    ResetCheckCfg,
    ResetCheckExp,
    ResetCheckModuleCfg,
    ResetCheckResult,
    ResetCheckSweepCfg,
)
from zcu_tools.program.v2 import (
    Branch,
    DirectReadoutCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    SweepCfg,
)
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modules.reset import NoneResetCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def _cfg(reset: bool) -> ResetCheckCfg | AmpRabiCfg:
    pulse = PulseCfg(
        waveform=ConstWaveformCfg(length=0.1), ch=0, nqz=1, freq=4000.0, gain=0.5
    )
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=6000.0)
    sweep = SweepCfg(start=0.1, stop=0.4, step=0.1, expts=4)
    if reset:
        return ResetCheckCfg(
            modules=ResetCheckModuleCfg(
                rabi_pulse=pulse, tested_reset=NoneResetCfg(), readout=readout
            ),
            sweep=ResetCheckSweepCfg(gain=sweep),
            reps=6,
            rounds=2,
        )
    return AmpRabiCfg(
        modules=AmpRabiModuleCfg(qub_pulse=pulse, readout=readout),
        sweep=AmpRabiSweepCfg(gain=sweep),
        reps=6,
        rounds=2,
    )


def _assert_hardware_sequence(program: ModularProgramV2, reset: bool) -> None:
    assert program.loop_dims == ([6, 4, 3] if reset else [6, 4])
    first = program.modules[1]
    assert isinstance(first, Pulse) and first.cfg is not None
    assert isinstance(first.cfg.gain, QickParam)
    if reset:
        branch = program.modules[2]
        assert isinstance(branch, Branch)
        assert [len(case) for case in branch.branches] == [0, 1, 2]
        last = branch.branches[2][1]
        assert isinstance(last, Pulse) and last.cfg is not None
        assert isinstance(last.cfg.gain, QickParam)
        assert last.cfg.gain.start == first.cfg.gain.start
        assert last.cfg.gain.spans == first.cfg.gain.spans


def _install_adc_populations(
    monkeypatch: pytest.MonkeyPatch, reset: bool
) -> list[ModularProgramV2]:
    programs: list[ModularProgramV2] = []
    original_acquire = ModularProgramV2.acquire
    original_process = ModularProgramV2._process_accumulated  # type: ignore[reportPrivateUsage]
    round_index = 0

    def acquire(program: ModularProgramV2, *args: Any, **kwargs: Any) -> Any:
        programs.append(program)
        _assert_hardware_sequence(program, reset)
        return original_acquire(program, *args, **kwargs)

    def process(program: ModularProgramV2, raw: Any) -> Any:
        # Supply deterministic integrated ADC data; exercise real classification
        # and averaging over reps/rounds with the compiled hardware loop order.
        nonlocal round_index
        assert program.loop_dims is not None
        indices = np.indices(program.loop_dims)
        shot, gain = indices[:2]
        g_count = (gain + indices[2]) % 4 + 1 if reset else gain + 1
        g_count = g_count + round_index
        round_index += 1
        iq = np.where(shot < g_count, -1.0, np.where(shot == 5, 5.0, 1.0))
        length = next(iter(program.ro_chs.values()))["length"]
        raw[0][..., 0, 0] = iq * length
        raw[0][..., 0, 1] = 0
        return original_process(program, raw)

    monkeypatch.setattr(ModularProgramV2, "acquire", acquire)
    monkeypatch.setattr(ModularProgramV2, "_process_accumulated", process)
    return programs


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("stop_after", [None, 0, 1])
def test_hardware_population_sweep_rounds_cancel_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reset: bool,
    stop_after: int | None,
) -> None:
    module = "reset_check" if reset else "amp_rabi"
    monkeypatch.setattr(
        f"zcu_tools.experiment.v2.singleshot.{module}.setup_devices",
        lambda *args, **kwargs: None,
    )
    soc, soccfg = make_mock_soc(n_gens=1, n_readouts=1)
    programs = _install_adc_populations(monkeypatch, reset)
    event = Event()
    original_poll = soc.poll_data
    polls = 0

    def poll() -> Any:
        nonlocal polls
        data = original_poll()
        polls += 1
        if stop_after is not None and polls == stop_after + 1:
            event.set()
        return data

    monkeypatch.setattr(soc, "poll_data", poll)
    cfg = _cfg(reset)
    before = cfg.model_dump()
    exp = ResetCheckExp() if reset else AmpRabiExp()
    with schedule_stop_scope(StopSignal(event)):
        result = exp.run(soc, soccfg, cfg, -1, 1, 0.5)  # type: ignore[arg-type]
    assert cfg.model_dump() == before
    assert len(programs) == 1
    assert result.signals.shape == ((4, 3, 2) if reset else (4, 2))
    if stop_after == 0:
        assert np.isnan(result.signals).all()
    else:
        gains = np.arange(4)
        ground = (
            ((gains[:, None] + np.arange(3)) % 4 + 1) / 6 if reset else (gains + 1) / 6
        )
        if stop_after is None:
            ground = ground + 0.5 / 6
        np.testing.assert_allclose(result.signals[..., 0], ground)
        np.testing.assert_allclose(result.signals[..., 1], 5 / 6 - ground)
    assert result.cfg_snapshot is not None and result.cfg_snapshot.rounds == 2
    path = str(tmp_path / "population.hdf5")
    exp.save(path, result)  # type: ignore[arg-type]
    loaded = exp.load(path)
    np.testing.assert_array_equal(loaded.signals, result.signals)
    np.testing.assert_array_equal(loaded.population_states, [0, 1])


@pytest.mark.parametrize("correct", [False, True])
def test_reset_population_analysis_and_stage_styles(correct: bool) -> None:
    gains = np.array([0.3, 0.1, 0.2, 0.4])
    true = np.tile([0.8, 0.15, 0.05], (4, 3, 1))
    true[0, 1] = [0.7, 0.25, 0.05]
    matrix = np.array([[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0, 0, 1]])
    measured = true @ matrix if correct else true.copy()
    measured[-1] = np.nan
    result = ResetCheckResult(gains, np.arange(3), measured[..., :2])
    analysis, figure = ResetCheckExp().analyze(
        result, confusion_matrix=matrix if correct else None
    )
    try:
        np.testing.assert_allclose(analysis.populations[:3], true[:3])
        assert analysis.analyzed_reset_points == 3
        assert analysis.reset_max_excited_population == pytest.approx(0.25)
        assert analysis.worst_sample_gain == 0.3
        lines = figure.axes[0].lines
        assert len(lines) == 9
        assert [line.get_color() for line in lines] == ["blue", "red", "green"] * 3
        assert [line.get_linestyle() for line in lines] == ["-"] * 3 + ["--"] * 3 + [
            ":"
        ] * 3
        expected = analysis.populations[np.argsort(gains)].reshape(4, 9).T
        for line, values in zip(lines, expected, strict=True):
            np.testing.assert_allclose(np.asarray(line.get_ydata()), values)
    finally:
        plt.close(figure)


def test_amp_population_rabi_fit() -> None:
    gains = np.linspace(0, 1, 51)
    ground = 0.5 + 0.4 * np.cos(4 * np.pi * gains)
    result = AmpRabiResult(gains, np.column_stack((ground, 0.95 - ground)))
    fit, figure = AmpRabiExp().analyze(result)
    try:
        assert fit.frequency == pytest.approx(2, rel=0.01)
        assert fit.amplitude == pytest.approx(0.4, abs=0.01)
        assert fit.pi_gain == pytest.approx(0.25, abs=0.01)
        assert len(figure.axes[0].lines) == 4
    finally:
        plt.close(figure)
