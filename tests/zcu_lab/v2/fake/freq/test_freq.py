"""Hardware-free frequency records use the shared numeric and plot contracts."""

from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pytest
from zcu_tools.analysis.fitting import HangerModel, TransmissionModel
from zcu_tools.datafile import load_labber_data
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    LoadDataRequest,
    MetaDictWriteback,
    RunRequest,
    SaveDataRequest,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.zcu_lab.v2.onetone._support import make_freq_result, make_readout
from zcu_lab.v2.fake.freq.core import (
    FakeFreqCfg,
    FakeFreqExp,
    FakeFreqRunResult,
    HangerSimParams,
    TransmissionSimParams,
)
from zcu_lab.v2.fake.freq.gui import FakeFreqAdapter, FakeFreqAnalyzeParams
from zcu_lab.v2.onetone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)


def make_cfg() -> FakeFreqCfg:
    return FakeFreqCfg.model_validate(
        {
            "reps": 4,
            "rounds": 3,
            "fast_mode": True,
            "modules": {"readout": make_readout()},
            "sweep": {
                "freq": {"start": 5965.0, "stop": 6035.0, "step": 0.25, "expts": 281}
            },
        }
    )


@pytest.mark.parametrize("model_type", ["hm", "t"])
@pytest.mark.parametrize("noise_scale", [0.0, 0.1])
def test_real_simulation_preserves_grid_round_averaging_and_native_trace(
    monkeypatch: pytest.MonkeyPatch, model_type: Literal["hm", "t"], noise_scale: float
) -> None:
    cfg = make_cfg()
    before = cfg.model_copy(deep=True)
    params = (
        HangerSimParams(noise_scale=noise_scale)
        if model_type == "hm"
        else TransmissionSimParams(noise_scale=noise_scale)
    )
    monkeypatch.setattr(
        np.random, "default_rng", lambda: np.random.Generator(np.random.PCG64(23))
    )
    plots = Plots(NonPresentingHost())
    result = FakeFreqExp(model_type, params).run(
        cfg,
        context=RunContext(None, None, plots, devices={}, cancel_signal=StopSignal()),
    )
    freqs = np.linspace(cfg.sweep.freq.start, cfg.sweep.freq.stop, cfg.sweep.freq.expts)
    if model_type == "hm":
        clean = HangerModel.calc_signals(
            freqs, freq=6000.0, Ql=5000.0, Qc=6000.0, phi=0.0, a0=1.0, edelay=0.05
        )
    else:
        clean = TransmissionModel.calc_signals(
            freqs, freq=6000.0, Ql=5000.0, a0=1.0, edelay=0.05
        )
    rng = np.random.Generator(np.random.PCG64(23))
    expected = np.zeros(len(freqs), dtype=np.complex128)
    for _ in range(cfg.rounds):
        sigma = noise_scale / np.sqrt(cfg.reps)
        expected += clean + rng.normal(0, sigma, len(freqs))
        expected += 1j * rng.normal(0, sigma, len(freqs))
    expected /= cfg.rounds
    assert cfg == before
    assert result.signals.dtype == np.complex128
    np.testing.assert_array_equal(result.freqs, freqs)
    np.testing.assert_allclose(result.signals, expected, rtol=1e-13, atol=1e-13)
    figures = plots.finish()
    assert tuple(figures) == ("measurement",)
    line = figures["measurement"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), freqs)
    np.testing.assert_allclose(np.asarray(line.get_ydata()), np.abs(expected))
    plots.release()


def test_stop_before_first_round_preserves_zero_buffer() -> None:
    stop = StopSignal()
    stop.set()
    plots = Plots(NonPresentingHost())
    result = FakeFreqExp("hm", HangerSimParams()).run(
        make_cfg(),
        context=RunContext(None, None, plots, devices={}, cancel_signal=stop),
    )
    np.testing.assert_array_equal(result.signals, np.zeros(281, dtype=np.complex128))
    assert tuple(plots.finish()) == ("measurement",)
    plots.release()


def test_stop_after_first_round_preserves_completed_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = StopSignal()
    rng = np.random.Generator(np.random.PCG64(23))

    class InterruptingGenerator:
        def __init__(self) -> None:
            self.draws = 0

        def normal(self, mean: float, sigma: float, size: int) -> np.ndarray:
            data = rng.normal(mean, sigma, size)
            self.draws += 1
            if self.draws == 2:
                stop.set()
            return data

    monkeypatch.setattr(np.random, "default_rng", InterruptingGenerator)
    cfg = make_cfg()
    plots = Plots(NonPresentingHost())
    result = FakeFreqExp("hm", HangerSimParams(noise_scale=0.0)).run(
        cfg, context=RunContext(None, None, plots, devices={}, cancel_signal=stop)
    )
    expected = HangerModel.calc_signals(
        result.freqs, freq=6000.0, Ql=5000.0, Qc=6000.0, phi=0.0, a0=1.0, edelay=0.05
    )
    np.testing.assert_allclose(result.signals, expected)
    plots.finish()
    plots.release()


@pytest.mark.parametrize("model_type", ["hm", "t"])
def test_analysis_is_blind_to_truth_and_accepts_nullable_cfg(
    model_type: Literal["hm", "t"],
) -> None:
    source = RunRecord[FakeFreqCfg, FreqResult](
        cfg=None, result=make_freq_result(hanger=model_type == "hm", background=True)
    )
    core = FakeFreqExp("hm", HangerSimParams(freq=6140.0))
    plots = Plots(NonPresentingHost())
    analysis = core.analyze(
        source,
        FreqAnalyzeOptions(
            model_type=model_type,
            edelay=0.021,
            fit_bg_amp_slope=True,
            fit_bg_phase_curvature=True,
        ),
        plots=plots,
    )
    assert analysis.freq == pytest.approx(6000.0, abs=0.01)
    assert analysis.fwhm == pytest.approx(6000.0 / 700.0, rel=0.01)
    assert analysis.params["bg_amp_slope"] == pytest.approx(0.008, abs=1e-4)
    assert analysis.params["bg_phase_curvature"] == pytest.approx(7e-4, abs=1e-5)
    assert tuple(plots.finish()) == ("fit",)
    plots.release()


@pytest.mark.parametrize("model_type", ["hm", "t"])
def test_gui_run_captures_fake_cfg_and_works_without_soc(
    model_type: Literal["hm", "t"],
) -> None:
    adapter = FakeFreqAdapter(model_type=model_type, fast_mode=True)
    raw_cfg = make_cfg().to_dict()
    plots = Plots(NonPresentingHost())
    source = adapter.run(
        RunRequest(None, None, {}),
        raw_cfg,
        context=RunContext(None, None, plots, devices={}, cancel_signal=StopSignal()),
    )
    expected_cfg = make_cfg()
    expected_cfg.dev = {}
    assert source.cfg == expected_cfg
    assert source.cfg is not None and source.cfg.fast_mode
    assert source.result.freqs.shape == source.result.signals.shape == (281,)
    raw_cfg["reps"] = 100
    assert source.cfg.reps == 4
    assert tuple(plots.finish()) == ("measurement",)
    plots.release()


@pytest.mark.parametrize("model_type", ["hm", "t", "auto"])
def test_gui_options_use_real_core_and_keep_two_scalar_writebacks(
    model_type: Literal["hm", "t", "auto"],
) -> None:
    data = make_freq_result(hanger=model_type != "t")
    source = RunRecord[FakeFreqCfg, FreqResult](cfg=None, result=data)
    params = FakeFreqAnalyzeParams(
        model_type=model_type, fit_bg_amp_slope=True, fit_bg_phase_curvature=True
    )
    expected_plots = Plots(NonPresentingHost())
    expected = FreqExp().analyze(
        RunRecord[FreqCfg, FreqResult](cfg=None, result=data),
        FreqAnalyzeOptions(
            model_type=model_type, fit_bg_amp_slope=True, fit_bg_phase_curvature=True
        ),
        plots=expected_plots,
    )
    expected_plots.finish()
    adapter = FakeFreqAdapter(params=HangerSimParams(freq=6200.0))
    plots = Plots(NonPresentingHost())
    analysis = adapter.analyze(
        AnalyzeRequest(source, params, Mock(), Mock(), None), plots=plots
    )
    assert analysis.freq == expected.freq
    assert analysis.fwhm == expected.fwhm
    assert analysis.params == expected.params
    assert tuple(plots.finish()) == ("fit",)
    items = adapter.get_writeback_items(
        WritebackRequest(source, analysis, Mock(spec=SessionEnv))
    )
    assert [
        (item.target_name, item.proposed_value)
        for item in items
        if isinstance(item, MetaDictWriteback)
    ] == [
        ("r_f", analysis.freq),
        ("rf_w", analysis.fwhm),
    ]
    expected_plots.release()
    plots.release()


def save_request(
    source: FakeFreqRunResult, path: Path
) -> SaveDataRequest[FakeFreqRunResult]:
    return SaveDataRequest(
        run_result=source,
        data_path=str(path),
        md=Mock(),
        ml=Mock(),
        chip_name="chip",
        qub_name="qubit",
        res_name="resonator",
        active_label="zero",
    )


def test_gui_exact_canonical_save_load_preserves_hz_complex_cfg_and_comment(
    tmp_path: Path,
) -> None:
    source = RunRecord(cfg=make_cfg(), result=make_freq_result())
    adapter = FakeFreqAdapter(params=HangerSimParams(freq=6300.0))
    path = tmp_path / "fake.hdf5"
    adapter.save(save_request(source, path))
    loaded = adapter.load(LoadDataRequest(str(path), Mock(), Mock()))
    assert loaded.cfg == source.cfg
    np.testing.assert_array_equal(loaded.result.signals, source.result.signals)
    np.testing.assert_allclose(loaded.result.freqs, source.result.freqs)
    disk = load_labber_data(str(path))
    np.testing.assert_allclose(disk.axes[0].values, source.result.freqs * 1e6)
    assert disk.axes[0].unit == "Hz"
    assert "fake/freq simulated data" in disk.comment
    assert disk.tags == ["fake/freq"]
    with pytest.raises(FileExistsError):
        adapter.save(save_request(source, path))
    np.testing.assert_array_equal(
        adapter.load(LoadDataRequest(str(path), Mock(), Mock())).result.signals,
        source.result.signals,
    )


def test_missing_cfg_save_rejects_but_disabled_persistence_is_noop(
    tmp_path: Path,
) -> None:
    source = RunRecord[FakeFreqCfg, FreqResult](cfg=None, result=make_freq_result())
    path = tmp_path / "missing.hdf5"
    with pytest.raises(ValueError, match="cfg"):
        FakeFreqAdapter().save(save_request(source, path))
    FakeFreqAdapter(persist_data=False).save(save_request(source, path))
    assert not path.exists()


def test_notebook_reuse_keeps_explicit_source_and_native_figure(tmp_path: Path) -> None:
    adapter = NotebookAdapter(host=NonPresentingHost())(
        FakeFreqExp("hm", HangerSimParams(freq=6200.0))
    )
    a = RunRecord(cfg=make_cfg(), result=make_freq_result(freq=6000.0))
    b = adapter.load(
        adapter.save(
            tmp_path / "b.hdf5",
            source=RunRecord(cfg=make_cfg(), result=make_freq_result(freq=6020.0)),
        )
    )
    analysis = adapter.analyze(FreqAnalyzeOptions(edelay=0.021), source=a)
    assert analysis.source is a
    assert analysis.result.freq == pytest.approx(6000.0, abs=0.01)
    assert adapter.last_run is b
    handle = adapter.analysis_presentation
    assert handle is not None
    handle.release()
    figure = analysis.figures["fit"]
    figure.axes[0].set_title("Retained native figure")
    figure.savefig(tmp_path / "native.png")
    assert (tmp_path / "native.png").stat().st_size > 0
    with pytest.raises(ValueError, match="Invalid model type"):
        adapter.analyze(FreqAnalyzeOptions(model_type="invalid"), source=b)  # type: ignore[arg-type]
    assert adapter.analysis is analysis
    assert adapter.last_run is b
    loaded = adapter.load(adapter.save(tmp_path / "a.hdf5", source=a))
    assert loaded.cfg == a.cfg
    np.testing.assert_array_equal(loaded.result.signals, a.result.signals)
