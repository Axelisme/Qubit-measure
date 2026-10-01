from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.datafile import save_labber_data
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import StopSignal, schedule_stop_scope
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
from zcu_tools.plotting.plots import NonPresentingHost, Plots
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


@pytest.fixture
def plots() -> Iterator[Plots]:
    session = Plots(NonPresentingHost())
    try:
        yield session
    finally:
        session.finish(present=False)
        session.release()


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
            g_center=-1,
            e_center=1,
            radius=0.5,
        )
    return AmpRabiCfg(
        modules=AmpRabiModuleCfg(qub_pulse=pulse, readout=readout),
        sweep=AmpRabiSweepCfg(gain=sweep),
        reps=6,
        rounds=2,
        g_center=-1,
        e_center=1,
        radius=0.5,
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
        result = exp.run(soc, soccfg, cfg)  # type: ignore[arg-type]
    assert cfg.model_dump() == before
    assert len(programs) == (1 if reset or stop_after == 0 else 2)
    assert result.signals.shape == ((4, 3, 2) if reset else (4, 12))
    if stop_after == 0:
        assert np.isnan(result.signals).all()
    elif reset:
        gains = np.arange(4)
        ground = (
            ((gains[:, None] + np.arange(3)) % 4 + 1) / 6 if reset else (gains + 1) / 6
        )
        if stop_after is None:
            ground = ground + 0.5 / 6
        np.testing.assert_allclose(result.signals[..., 0], ground)
        np.testing.assert_allclose(result.signals[..., 1], 5 / 6 - ground)
    else:
        assert np.iscomplexobj(result.signals)
        completed = 2 if stop_after is None else 1
        for round_index in range(completed):
            shot = np.arange(6)[None, :]
            g_count = np.arange(4)[:, None] + 1 + round_index
            expected = np.where(shot < g_count, -1.0, np.where(shot == 5, 5.0, 1.0))
            np.testing.assert_array_equal(
                result.signals[:, round_index * 6 : (round_index + 1) * 6], expected
            )
        if completed == 1:
            assert np.isnan(result.signals[:, 6:]).all()
    assert result.cfg_snapshot is not None and result.cfg_snapshot.rounds == 2
    _assert_population_roundtrip(exp, result, tmp_path / "population.hdf5")


def _assert_population_roundtrip(
    exp: ResetCheckExp | AmpRabiExp,
    result: ResetCheckResult | AmpRabiResult,
    path: Path,
) -> None:
    if isinstance(exp, ResetCheckExp):
        assert isinstance(result, ResetCheckResult)
        exp.save(RunRecord(cfg=result.cfg_snapshot, result=result), path)
    else:
        assert isinstance(result, AmpRabiResult)
        exp.save(RunRecord(cfg=result.cfg_snapshot, result=result), path)
    source = exp.load(path)
    loaded = source.result
    np.testing.assert_array_equal(loaded.signals, result.signals)
    assert source.cfg == result.cfg_snapshot
    if isinstance(loaded, ResetCheckResult):
        np.testing.assert_array_equal(loaded.population_states, [0, 1])
    else:
        np.testing.assert_array_equal(loaded.shot_indices, np.arange(12))


@pytest.mark.parametrize("reset", [False, True])
def test_invalid_snapshot_calibration_fails_before_device_setup(
    monkeypatch: pytest.MonkeyPatch, reset: bool
) -> None:
    module = "reset_check" if reset else "amp_rabi"
    touched = []
    monkeypatch.setattr(
        f"zcu_tools.experiment.v2.singleshot.{module}.setup_devices",
        lambda *args, **kwargs: touched.append(True),
    )
    cfg = _cfg(reset)
    cfg.radius = -1
    exp = ResetCheckExp() if reset else AmpRabiExp()
    with pytest.raises(ValueError, match="radius"):
        exp.run(None, None, cfg)  # type: ignore[arg-type]
    assert not touched


@pytest.mark.parametrize("correct", [False, True])
def test_reset_population_analysis_and_stage_styles(
    correct: bool, plots: Plots
) -> None:
    gains = np.array([0.3, 0.1, 0.2, 0.4])
    true = np.tile([0.8, 0.15, 0.05], (4, 3, 1))
    true[0, 1] = [0.7, 0.25, 0.05]
    matrix = np.array([[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0, 0, 1]])
    measured = true @ matrix if correct else true.copy()
    measured[-1] = np.nan
    result = ResetCheckResult(gains, np.arange(3), measured[..., :2])
    analysis, figure = ResetCheckExp().analyze(
        result, confusion_matrix=matrix if correct else None, plots=plots
    )
    try:
        assert plots.finish(present=False)["populations"] is figure
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


def test_amp_rejects_population_only_file(tmp_path: Path) -> None:
    path = save_labber_data(
        str(tmp_path / "old_amp.hdf5"),
        ("Population", "a.u.", np.full((5, 2), 0.5)),
        [("GE Population", "None", [0, 1]), ("Gain", "a.u.", np.linspace(0, 1, 5))],
    )
    with pytest.raises(ValueError, match="Shot Index|Signal|axis"):
        AmpRabiExp().load(Path(path))


def test_amp_rejects_partial_raw_sweep() -> None:
    signals = np.ones((5, 10), dtype=np.complex128)
    signals[0, 0] = np.nan
    result = AmpRabiResult(np.linspace(0, 1, 5), np.arange(10), signals)
    with pytest.raises(ValueError, match="finite raw IQ"):
        AmpRabiExp().analyze(result)


@pytest.mark.parametrize("initial_state", ["ground", "excited"])
def test_amp_raw_iq_rabi_fit(initial_state: Literal["ground", "excited"]) -> None:
    rng = np.random.default_rng(238)
    gains = np.linspace(-0.3, 1.2, 25)[::-1]
    p_e0 = 0.1 if initial_state == "ground" else 0.9
    p_e = 0.5 + (p_e0 - 0.5) * np.cos(4 * np.pi * gains)
    excited = rng.random((gains.size, 1000)) < p_e[:, None]
    signals = rng.normal(np.where(excited, 1.0, -1.0), 0.18).astype(np.complex128)
    # Entirely missing rounds can still be analyzed from completed shots.
    signals = np.column_stack((signals, np.full_like(signals, np.nan)))
    result = AmpRabiResult(gains, np.arange(2000), signals)
    fit, figure = AmpRabiExp().analyze(result, initial_state=initial_state)
    try:
        assert fit.joint_fit.backend.valid
        assert fit.joint_fit.phase == 0.0
        assert fit.joint_fit.t_r is None
        assert abs(fit.joint_fit.g_center + 1) < 0.1
        assert abs(fit.joint_fit.e_center - 1) < 0.1
        assert fit.frequency == pytest.approx(2, rel=0.01)
        assert fit.amplitude == pytest.approx(0.4, abs=0.05)
        assert fit.pi_gain == pytest.approx(0.25, abs=0.01)
        assert fit.pi2_gain == pytest.approx(0.125, abs=0.005)
        assert np.isfinite(fit.pi_gain_error) and fit.pi_gain_error > 0
        assert fit.pi2_gain_error == pytest.approx(fit.pi_gain_error / 2)
        assert len(figure.axes) == 3
        assert "gain" in figure.axes[0].get_xlabel()
        assert "rad/gain" in figure.axes[0].get_title()
    finally:
        plt.close(figure)
