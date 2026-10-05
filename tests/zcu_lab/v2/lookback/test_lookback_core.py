from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import load_labber_data
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soc

from tests.zcu_lab.v2.lookback._lookback_support import make_lookback_cfg
from zcu_lab.v2.lookback.core import (
    LookbackAnalyzeOptions,
    LookbackCfg,
    LookbackExp,
    LookbackResult,
)


def test_analysis_uses_complex_source_and_publishes_named_fit() -> None:
    times = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    signals = np.array([0.0, 0.5j, 2.0j, 10.0j, 3.0j], dtype=np.complex128)
    source = RunRecord[LookbackCfg, LookbackResult](
        cfg=None, result=LookbackResult(times, signals)
    )
    plots = Plots(NonPresentingHost())

    answer = LookbackExp().analyze(source, LookbackAnalyzeOptions(), plots=plots)
    figures = plots.finish()

    assert answer.predict_offset == pytest.approx(0.4)
    assert tuple(figures) == ("fit",)
    lines = figures["fit"].axes[0].lines
    np.testing.assert_array_equal(lines[0].get_xdata(), times)
    np.testing.assert_array_equal(lines[0].get_ydata(), signals.real)
    np.testing.assert_array_equal(lines[1].get_ydata(), signals.imag)
    np.testing.assert_array_equal(lines[2].get_ydata(), np.abs(signals))
    np.testing.assert_array_equal(source.result.signals, signals)
    plots.release()


@pytest.mark.parametrize("signals", [[10j, 1j, 0j], [5j, 6j, 10j]])
def test_no_prior_threshold_candidate_uses_first_time(signals: list[complex]) -> None:
    source = RunRecord[LookbackCfg, LookbackResult](
        cfg=None,
        result=LookbackResult(np.array([2.0, 3.0, 4.0]), np.array(signals)),
    )
    plots = Plots(NonPresentingHost())
    answer = LookbackExp().analyze(source, LookbackAnalyzeOptions(), plots=plots)
    assert answer.predict_offset == pytest.approx(2.0)
    plots.finish()
    plots.release()


def test_readout_markers_follow_explicit_source_cfg() -> None:
    core = LookbackExp()
    data = LookbackResult(np.arange(3.0), np.array([0j, 1j, 10j]))
    cfg_a = make_lookback_cfg(trig_offset=0.25)
    source_a = RunRecord(cfg=cfg_a, result=data)
    source_b = RunRecord(cfg=make_lookback_cfg(trig_offset=1.5), result=data)
    cfg_a.modules.readout.ro_cfg.trig_offset = 99.0

    for source, expected in (
        (source_b, [1.5, 3.5]),
        (source_a, [0.25, 2.25]),
    ):
        plots = Plots(NonPresentingHost())
        answer = core.analyze(
            source, LookbackAnalyzeOptions(plot_fit=False), plots=plots
        )
        figures = plots.finish()
        lines = figures["fit"].axes[0].lines
        assert answer.predict_offset == 1.0
        assert [
            float(np.asarray(line.get_xdata(), dtype=np.float64)[0])
            for line in lines[3:]
        ] == expected
        plots.release()

    plots = Plots(NonPresentingHost())
    core.analyze(
        RunRecord[LookbackCfg, LookbackResult](cfg=None, result=data),
        LookbackAnalyzeOptions(plot_fit=False),
        plots=plots,
    )
    figures = plots.finish()
    assert len(figures["fit"].axes[0].lines) == 3
    plots.release()


def test_run_keeps_original_reps_and_captured_trigger_offset() -> None:
    cfg = make_lookback_cfg(reps=3, trig_offset=0.25)
    original = cfg.model_copy(deep=True)
    soc, soccfg = make_mock_soc()
    plots = Plots(NonPresentingHost())
    with pytest.warns(UserWarning, match="reps is not 1"):
        result = LookbackExp().run(
            cfg,
            context=RunContext(
                soc, soccfg, plots, devices={}, cancel_signal=StopSignal()
            ),
        )
    figures = plots.finish()

    assert cfg == original and cfg.reps == 3
    assert result.times[0] == pytest.approx(0.25)
    assert result.signals.shape == result.times.shape
    assert result.signals.dtype == np.complex128
    np.testing.assert_array_equal(
        figures["measurement"].axes[0].lines[0].get_xdata(), result.times
    )
    np.testing.assert_array_equal(
        figures["measurement"].axes[0].lines[0].get_ydata(), np.abs(result.signals)
    )
    plots.release()


def test_canonical_roundtrip_preserves_seconds_complex_and_original_cfg(
    tmp_path: Path,
) -> None:
    source = RunRecord(
        cfg=make_lookback_cfg(reps=7),
        result=LookbackResult(np.array([0.4, 0.8]), np.array([1 + 2j, 3 - 4j])),
    )
    path = tmp_path / "trace.hdf5"
    exp = LookbackExp()
    exp.save(source, path)
    loaded = exp.load(path)
    assert loaded.cfg == source.cfg
    np.testing.assert_array_equal(loaded.result.signals, source.result.signals)
    assert loaded.result.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.result.times, source.result.times)
    disk = load_labber_data(str(path))
    np.testing.assert_allclose(disk.axes[0].values, source.result.times * 1e-6)


def test_missing_source_cfg_is_analyzable_but_not_saveable(tmp_path: Path) -> None:
    source = RunRecord[LookbackCfg, LookbackResult](
        cfg=None,
        result=LookbackResult(np.array([0.0, 1.0]), np.array([0j, 10j])),
    )
    plots = Plots(NonPresentingHost())
    answer = LookbackExp().analyze(source, LookbackAnalyzeOptions(), plots=plots)
    assert answer.predict_offset == 0.0
    plots.finish()
    destination = tmp_path / "missing-cfg.hdf5"
    with pytest.raises(ValueError, match="cfg|config"):
        LookbackExp().save(source, destination)
    assert not destination.exists()
    plots.release()
