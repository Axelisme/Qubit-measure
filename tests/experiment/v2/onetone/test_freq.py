"""Frequency acquisition, explicit analysis sources and canonical data."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from zcu_tools.datafile import load_labber_data
from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.onetone.freq import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soc

from tests.experiment.v2.onetone._support import make_freq_cfg, make_freq_result


@pytest.mark.parametrize("model_type", ["hm", "t", "auto"])
@pytest.mark.parametrize("background", [False, True])
def test_real_fit_uses_explicit_source_and_trims_endpoints(
    model_type: Literal["hm", "t", "auto"], background: bool
) -> None:
    original = make_freq_result(hanger=model_type != "t", background=background)
    signals = original.signals.copy()
    signals[[0, -1]] = 1e5 + 2e5j
    source = RunRecord[FreqCfg, FreqResult](
        cfg=None, result=FreqResult(original.freqs, signals)
    )
    plots = Plots(NonPresentingHost())
    options = FreqAnalyzeOptions(
        model_type=model_type,
        edelay=0.021,
        fit_bg_amp_slope=background,
        fit_bg_phase_curvature=background,
    )
    answer = FreqExp().analyze(source, options, plots=plots)
    figures = plots.finish()
    assert answer.freq == pytest.approx(6000.0, abs=0.01)
    assert answer.fwhm == pytest.approx(6000.0 / 700.0, rel=0.01)
    assert answer.params["edelay"] == pytest.approx(0.021)
    if background:
        assert answer.params["bg_amp_slope"] == pytest.approx(0.008, abs=1e-4)
        assert answer.params["bg_phase_curvature"] == pytest.approx(7e-4, abs=1e-5)
    assert tuple(figures) == ("fit",)
    assert len(figures["fit"].axes) == 3
    raw_line = next(
        line for line in figures["fit"].axes[2].lines if line.get_label() == "raw data"
    )
    np.testing.assert_array_equal(raw_line.get_xdata(), original.freqs[1:-1])
    np.testing.assert_allclose(
        np.asarray(raw_line.get_ydata()), np.abs(original.signals[1:-1])
    )
    plots.release()


@pytest.mark.parametrize("homophasal", [False, True])
def test_mock_soc_acquisition_preserves_cfg_and_publishes_final_trace(
    homophasal: bool,
) -> None:
    cfg = make_freq_cfg(homophasal=homophasal)
    before = cfg.model_copy(deep=True)
    soc, soccfg = make_mock_soc()
    plots = Plots(NonPresentingHost())
    answer = FreqExp().run(cfg, context=QickContext(soc, soccfg, plots))
    figures = plots.finish()
    assert cfg == before
    assert answer.freqs.shape == answer.signals.shape == (9,)
    assert answer.signals.dtype == np.complex128
    assert np.all(np.isfinite(answer.signals))
    assert np.all(np.diff(answer.freqs) > 0)
    steps = np.diff(answer.freqs)
    assert bool(np.allclose(steps, np.median(steps), rtol=1e-4)) != homophasal
    line = figures["measurement"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), answer.freqs)
    np.testing.assert_array_equal(line.get_ydata(), np.abs(answer.signals))
    plots.release()


def test_canonical_roundtrip_keeps_complex_hz_and_cfg(tmp_path: Path) -> None:
    core = FreqExp()
    source = RunRecord(cfg=make_freq_cfg(homophasal=True), result=make_freq_result())
    path = tmp_path / "frequency.hdf5"
    core.save(source, path)
    loaded = core.load(path)
    assert loaded.cfg == source.cfg
    np.testing.assert_allclose(loaded.result.freqs, source.result.freqs)
    np.testing.assert_array_equal(loaded.result.signals, source.result.signals)
    np.testing.assert_allclose(
        load_labber_data(str(path)).axes[0].values, source.result.freqs * 1e6
    )


def test_reused_core_uses_only_selected_source_and_missing_cfg_is_not_saveable(
    tmp_path: Path,
) -> None:
    core = FreqExp()
    sources = [
        RunRecord[FreqCfg, FreqResult](cfg=None, result=make_freq_result(freq=f))
        for f in [6000.0, 6020.0]
    ]
    for source in [sources[0], sources[1], sources[0]]:
        plots = Plots(NonPresentingHost())
        answer = core.analyze(source, FreqAnalyzeOptions(edelay=0.021), plots=plots)
        plots.finish()
        assert answer.freq == pytest.approx(np.mean(source.result.freqs), abs=0.01)
        plots.release()
    with pytest.raises(ValueError, match="cfg"):
        core.save(sources[0], tmp_path / "missing-cfg.hdf5")
