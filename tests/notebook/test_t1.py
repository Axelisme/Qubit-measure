"""Notebook publication, retained figures, and explicit core delegation."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.v2.twotone.time_domain.t1 import (
    T1Analysis,
    T1AnalyzeOptions,
    T1Cfg,
    T1Result,
)
from zcu_tools.experiment.v2.twotone.time_domain.t1 import (
    T1Exp as T1Core,
)
from zcu_tools.notebook.experiments import T1Exp
from zcu_tools.plotting.plots import Plots


def make_result(t1: float = 20.0) -> T1Result:
    cfg = T1Cfg.model_validate(
        {
            "reps": 2,
            "rounds": 1,
            "uniform": False,
            "modules": {
                "pi_pulse": {
                    "ch": 0,
                    "nqz": 1,
                    "gain": 1.0,
                    "freq": 4000.0,
                    "waveform": {"style": "const", "length": 0.4},
                },
                "readout": {
                    "type": "readout/direct",
                    "ro_ch": 0,
                    "ro_length": 1.0,
                    "ro_freq": 7000.0,
                },
            },
            "sweep": {"length": [0.0, 10.0, 40.0]},
        }
    )
    times = np.linspace(0, 80, 81)
    return T1Result(times, np.exp(-times / t1).astype(np.complex128), cfg)


@pytest.fixture
def loaded(tmp_path: Path) -> T1Exp:
    source = tmp_path / "source.hdf5"
    T1Core().save(make_result(), source)
    exp = T1Exp(present=False)
    exp.load(source)
    return exp


def test_explicit_older_analysis_does_not_replace_latest_run(
    loaded: T1Exp, tmp_path: Path
) -> None:
    latest = loaded.last_result
    initial = loaded.analyze(skip=1)
    previous_plots = loaded.analysis_plots
    older = make_result(12.0)
    analysis = loaded.analyze(older, skip=2)

    assert initial.t1 == pytest.approx(20)
    assert analysis.t1 == pytest.approx(12)
    assert loaded.last_result is latest
    record = loaded.analysis
    assert record is not None
    assert record.source is older
    assert record.result is analysis
    assert record.options == T1AnalyzeOptions(skip=2)
    assert record.plots is loaded.analysis_plots
    previous_plots["fit"].savefig(tmp_path / "previous.png")
    assert (tmp_path / "previous.png").stat().st_size > 0


def test_failed_analysis_and_load_preserve_successful_record(
    loaded: T1Exp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded.analyze()
    before = loaded.analysis
    latest = loaded.last_result

    def fail_analysis(
        self: T1Core, result: T1Result, options: T1AnalyzeOptions, *, plots: Plots
    ) -> T1Analysis:
        plots.subplots("diagnostic")
        raise ValueError("analysis failed")

    monkeypatch.setattr(T1Core, "analyze", fail_analysis)
    with pytest.raises(ValueError, match="analysis failed"):
        loaded.analyze(make_result(10))
    with pytest.raises(FileNotFoundError):
        loaded.load(tmp_path / "missing.hdf5")
    assert loaded.analysis is before
    assert loaded.last_result is latest
    assert list(loaded.analysis_plots) == ["fit"]


@pytest.mark.parametrize("fail", [False, True])
def test_run_publishes_only_after_success(
    loaded: T1Exp, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    loaded.analyze()
    before = loaded.analysis
    latest = loaded.last_result
    partial = make_result(10)
    partial.signals[-1] = np.nan
    assert isinstance(partial.cfg_snapshot, T1Cfg)
    cfg = partial.cfg_snapshot

    def run(self: T1Core, config: T1Cfg, *, context: QickContext) -> T1Result:
        assert config is cfg
        assert context.soc == "soc"
        assert context.soccfg == "soccfg"
        viewer = context.plots.liveplot_1d("measurement", "Time", "Signal")
        viewer.update(partial.times, partial.signals.real)
        if fail:
            raise RuntimeError("run failed")
        return partial

    monkeypatch.setattr(T1Core, "run", run)
    if fail:
        with pytest.raises(RuntimeError, match="run failed"):
            loaded.run("soc", "soccfg", cfg)
        assert loaded.analysis is before
        assert loaded.last_result is latest
    else:
        assert loaded.run("soc", "soccfg", cfg) is partial
        assert loaded.last_result is partial
        assert loaded.analysis is None
        assert loaded.run_plots is not None
        np.testing.assert_allclose(
            np.asarray(loaded.run_plots["measurement"].axes[0].lines[0].get_ydata()),
            partial.signals.real,
            equal_nan=True,
        )
        loaded.run_plots.release()


def test_notebook_save_load_keeps_config_and_clears_analysis(
    loaded: T1Exp, tmp_path: Path
) -> None:
    loaded.analyze()
    previous = loaded.analysis_plots
    path = tmp_path / "saved.hdf5"
    loaded.save(path, comment="retained metadata")
    result = loaded.load(path)
    assert isinstance(result.cfg_snapshot, T1Cfg)
    assert result.cfg_snapshot.uniform is False
    assert loaded.last_result is result
    assert loaded.analysis is None
    previous["fit"].savefig(tmp_path / "retained.png")
    assert (tmp_path / "retained.png").stat().st_size > 0


def test_missing_result_fails_before_plotting(tmp_path: Path) -> None:
    exp = T1Exp(present=False)
    with pytest.raises(ValueError, match="No T1 result"):
        exp.analyze()
    with pytest.raises(ValueError, match="No T1 result"):
        exp.save(tmp_path / "missing.hdf5")


def test_default_notebook_host_presents_and_releases_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipywidgets import Widget

    displayed: list[object] = []
    monkeypatch.setattr("IPython.display.display", displayed.append)
    before = set(Widget.widgets)
    exp = T1Exp()
    try:
        exp.analyze(make_result())
        figure = exp.analysis_plots["fit"]
        assert displayed == [figure.canvas]
        assert figure.canvas.manager is not None
        successful = exp.analysis
        live_widgets = set(Widget.widgets)

        def fail_display(widget: object) -> None:
            raise RuntimeError("publisher failed")

        monkeypatch.setattr("IPython.display.display", fail_display)
        with pytest.raises(RuntimeError, match="publisher failed"):
            exp.analyze(make_result(10))
        assert exp.analysis is successful
        assert set(Widget.widgets) == live_widgets
    finally:
        if exp.analysis is not None:
            exp.analysis_plots.release()
    assert set(Widget.widgets) == before
