"""Independent Notebook flux picks retain explicit sources and native figures."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pytest
from ipympl.backend_nbagg import Canvas, Toolbar
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep.line_state import FluxPickAnalysis, FluxPickState
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepCfg, FluxDepResult
from zcu_tools.experiment.v2.twotone.fluxdep import FreqFluxCfg, FreqFluxResult
from zcu_tools.notebook.experiments import (
    FluxDepAnalysisRecord,
    FluxDepAnalyzer,
    FluxDepPickerOptions,
)
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.plots import NonPresentingHost, PlotHost, Plots

from tests.experiment.v2.onetone.flux_dep_support import make_result


def make_source() -> RunRecord[FluxDepCfg, FluxDepResult]:
    return RunRecord(cfg=None, result=make_result())


@pytest.fixture(autouse=True)
def suppress_notebook_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "zcu_tools.notebook.experiments.flux_dep.ipython_display.display",
        lambda _widget: None,
    )


def expect_analysis_plots(analyzer: FluxDepAnalyzer) -> Plots:
    plots = analyzer.analysis_plots
    assert plots is not None
    return plots


def completed_analyzer(
    host: PlotHost | None = None,
) -> tuple[FluxDepAnalyzer, FluxDepAnalysisRecord, Plots]:
    analyzer = FluxDepAnalyzer(host)
    record = analyzer.start(make_source(), FluxDepPickerOptions(-0.2, 0.3)).done()
    return analyzer, record, expect_analysis_plots(analyzer)


def test_done_publishes_source_options_numeric_result_and_named_figure(
    tmp_path,
) -> None:
    analyzer = FluxDepAnalyzer(NonPresentingHost())
    source = make_source()
    original_signals = source.result.signals.copy()
    control = analyzer.start(
        source, FluxDepPickerOptions(-0.2, 0.3, magnitude_only=True)
    )
    try:
        assert isinstance(control.widget, widgets.VBox)
        assert control.positions() == pytest.approx((-0.2, 0.3))
        assert control.record is None
        assert analyzer.analysis is None
        control.set_positions(-0.1, 0.4)
        control.conjugate_checkbox.value = True

        record = control.done()

        assert control.is_finished
        assert analyzer.analysis is record
        assert record.source is source
        assert record.options == FluxPickState(
            flux_half=-0.1, flux_int=0.4, conjugate=True, magnitude_only=True
        )
        assert record.result == FluxPickAnalysis(-0.1, 0.4, 1.0)
        assert control.record is record
        assert control.plots is analyzer.analysis_plots
        assert list(record.figures) == ["pick"]
        assert record.figures["pick"] is not control.figure
        record.figures["pick"].savefig(tmp_path / "onetone-pick.png")
        assert (tmp_path / "onetone-pick.png").stat().st_size > 0
        np.testing.assert_array_equal(source.result.signals, original_signals)
        with pytest.raises(RuntimeError, match="finished"):
            control.done()
    finally:
        if not control.is_finished:
            control.cancel()
        if analyzer.analysis_plots is not None:
            analyzer.analysis_plots.release()


def test_twotone_pick_keeps_captured_source_and_retains_figure(tmp_path: Path) -> None:
    data = make_result()
    source: RunRecord[FreqFluxCfg, FreqFluxResult] = RunRecord(
        cfg=None,
        result=FreqFluxResult(data.values, data.freqs, data.signals),
    )
    analyzer = FluxDepAnalyzer[FreqFluxCfg, FreqFluxResult](NonPresentingHost())
    first = analyzer.start(source, FluxDepPickerOptions(-0.2, 0.3))
    other: RunRecord[FreqFluxCfg, FreqFluxResult] = RunRecord(
        cfg=None,
        result=FreqFluxResult(data.values, data.freqs + 10.0, data.signals * 2),
    )
    second = analyzer.start(other, FluxDepPickerOptions(-0.1, 0.4))
    try:
        second.cancel()
        assert analyzer.analysis is None
        first.set_positions(-0.15, 0.35)
        record = first.done()
        assert analyzer.analysis is record
        assert record.source is source
        assert record.source.result is source.result
        assert record.options.flux_half == pytest.approx(-0.15)
        assert record.result.flux_period == pytest.approx(1.0)
        plots = analyzer.analysis_plots
        assert plots is not None
        plots.release()
        record.figures["pick"].savefig(tmp_path / "twotone-pick.png")
        assert (tmp_path / "twotone-pick.png").stat().st_size > 0
    finally:
        if not first.is_finished:
            first.cancel()
        if not second.is_finished:
            second.cancel()
        if analyzer.analysis_plots is not None:
            analyzer.analysis_plots.release()


def test_unexpected_numeric_failure_retires_preview_and_keeps_published_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = FluxDepAnalyzer(NonPresentingHost())
    old = analyzer.start(make_source(), FluxDepPickerOptions(-0.2, 0.3)).done()
    old_plots = analyzer.analysis_plots
    assert old_plots is not None
    before_new = set(widgets.Widget.widgets)
    control = analyzer.start(make_source(), FluxDepPickerOptions(-0.1, 0.4))

    def fail_analysis(*_args: object) -> FluxPickAnalysis:
        raise RuntimeError("numeric analysis failed")

    monkeypatch.setattr(
        "zcu_tools.notebook.experiments.flux_dep.analyze_flux_pick", fail_analysis
    )
    try:
        with pytest.raises(RuntimeError, match="numeric analysis failed"):
            control.done()
        assert control.is_finished
        assert control.record is None
        assert control.plots is None
        assert analyzer.analysis is old
        assert analyzer.analysis_plots is old_plots
        assert set(widgets.Widget.widgets) == before_new
    finally:
        if not control.is_finished:
            control.cancel()
        old_plots.release()


def test_invalid_done_and_cancel_leave_old_analysis_and_editable_control() -> None:
    analyzer = FluxDepAnalyzer(NonPresentingHost())
    first = analyzer.start(make_source(), FluxDepPickerOptions(-0.2, 0.3))
    old = first.done()
    old_plots = analyzer.analysis_plots
    assert old_plots is not None
    next_control = analyzer.start(make_source(), FluxDepPickerOptions(0.0, 0.0))
    try:
        with pytest.raises(ValueError, match="separated"):
            next_control.done()
        assert not next_control.is_finished
        assert analyzer.analysis is old
        assert analyzer.analysis_plots is old_plots
        next_control.set_positions(-0.1, 0.4)
        next_control.cancel_button.click()
        assert next_control.is_finished
        assert analyzer.analysis is old
        assert analyzer.analysis_plots is old_plots
        with pytest.raises(RuntimeError, match="finished"):
            next_control.done()
    finally:
        if not next_control.is_finished:
            next_control.cancel()
        old_plots.release()


def test_failed_start_and_preview_cleanup_preserve_both_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp, old_record, old_plots = completed_analyzer()
    before_new = set(widgets.Widget.widgets)
    release = NotebookPlotHost.release

    def fail_display(_widget: object) -> None:
        raise RuntimeError("frontend publisher failed")

    def fail_release(self: NotebookPlotHost, figure: Figure) -> None:
        release(self, figure)
        raise RuntimeError("preview cleanup failed")

    try:
        with monkeypatch.context() as patch:
            patch.setattr("IPython.display.display", fail_display)
            patch.setattr(NotebookPlotHost, "release", fail_release)
            with pytest.raises(BaseExceptionGroup) as failure:
                exp.start(make_source(), FluxDepPickerOptions(-0.1, 0.4))
        assert [str(error) for error in failure.value.exceptions] == [
            "frontend publisher failed",
            "preview cleanup failed",
        ]
        assert exp.analysis is old_record
        assert exp.analysis_plots is old_plots
        assert set(widgets.Widget.widgets) == before_new
    finally:
        old_plots.release()


@pytest.mark.parametrize("fail_on", [1, 2])
def test_failed_notebook_publication_cleans_up_widgets_and_keeps_record(
    monkeypatch: pytest.MonkeyPatch, fail_on: int
) -> None:
    exp = FluxDepAnalyzer()
    source = make_source()
    existing_widgets = set(widgets.Widget.widgets)
    calls = 0

    def fail_display(_widget: object) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise RuntimeError("frontend publisher failed")

    monkeypatch.setattr("IPython.display.display", fail_display)
    with pytest.raises(RuntimeError, match="frontend publisher failed"):
        exp.start(source, FluxDepPickerOptions(-0.2, 0.3))
    assert exp.analysis is None
    assert set(widgets.Widget.widgets) == existing_widgets


def test_failed_final_pick_publication_retires_preview_without_losing_old_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("IPython.display.display", lambda _widget: None)
    initial_widgets = set(widgets.Widget.widgets)
    exp, old_record, old_plots = completed_analyzer()
    before_new = set(widgets.Widget.widgets)
    control = exp.start(make_source(), FluxDepPickerOptions(-0.1, 0.4))

    def fail_final(_widget: object) -> None:
        raise RuntimeError("final canvas failed")

    monkeypatch.setattr("IPython.display.display", fail_final)
    try:
        with pytest.raises(RuntimeError, match="final canvas failed"):
            control.done()
        assert control.is_finished
        assert exp.analysis is old_record
        assert exp.analysis_plots is old_plots
        assert set(widgets.Widget.widgets) == before_new
    finally:
        old_plots.release()
    assert set(widgets.Widget.widgets) == initial_widgets


def test_failed_final_pick_and_preview_cleanup_preserve_both_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp, old_record, old_plots = completed_analyzer()
    before_new = set(widgets.Widget.widgets)
    control = exp.start(make_source(), FluxDepPickerOptions(-0.1, 0.4))
    release = NotebookPlotHost.release

    def fail_final(_widget: object) -> None:
        raise RuntimeError("final canvas failed")

    def fail_preview(self: NotebookPlotHost, figure: Figure) -> None:
        release(self, figure)
        if figure is control.figure:
            raise RuntimeError("preview release failed")

    try:
        with monkeypatch.context() as patch:
            patch.setattr("IPython.display.display", fail_final)
            patch.setattr(NotebookPlotHost, "release", fail_preview)
            with pytest.raises(BaseExceptionGroup) as failure:
                control.done()
        assert [str(error) for error in failure.value.exceptions] == [
            "final canvas failed",
            "preview release failed",
        ]
        assert control.is_finished
        assert control.record is None
        assert control.plots is None
        assert exp.analysis is old_record
        assert exp.analysis_plots is old_plots
        assert set(widgets.Widget.widgets) == before_new
    finally:
        if not control.is_finished:
            control.cancel()
        old_plots.release()


@pytest.mark.parametrize("terminal", ["done", "cancel"])
def test_terminal_preview_release_failure_closes_controls_and_retains_record(
    monkeypatch: pytest.MonkeyPatch, terminal: str
) -> None:
    monkeypatch.setattr("IPython.display.display", lambda _widget: None)
    initial_widgets = set(widgets.Widget.widgets)
    exp, old_record, old_plots = completed_analyzer()
    before_new = set(widgets.Widget.widgets)
    control = exp.start(make_source(), FluxDepPickerOptions(-0.1, 0.4))
    release = NotebookPlotHost.release

    def fail_preview(self: NotebookPlotHost, figure: Figure) -> None:
        release(self, figure)
        if figure is control.figure:
            raise RuntimeError("preview release failed")

    action = control.done if terminal == "done" else control.cancel
    try:
        with monkeypatch.context() as patch:
            patch.setattr(NotebookPlotHost, "release", fail_preview)
            with pytest.raises(RuntimeError, match="preview release failed"):
                action()
        assert control.is_finished
        assert exp.analysis is old_record
        assert exp.analysis_plots is old_plots
        assert set(widgets.Widget.widgets) == before_new
    finally:
        old_plots.release()
    assert set(widgets.Widget.widgets) == initial_widgets


@pytest.mark.parametrize("terminal", ["done", "cancel"])
@pytest.mark.parametrize("stage", ["before", "after"])
def test_toolbar_failure_before_preview_canvas_close_preserves_old_record(
    monkeypatch: pytest.MonkeyPatch, terminal: str, stage: str
) -> None:
    monkeypatch.setattr("IPython.display.display", lambda _widget: None)
    existing_widgets = set(widgets.Widget.widgets)
    exp, old_record, old_plots = completed_analyzer()
    control = exp.start(make_source(), FluxDepPickerOptions(-0.1, 0.4))
    canvas = control.figure.canvas
    assert isinstance(canvas, Canvas)
    toolbar = canvas.toolbar
    assert isinstance(toolbar, Toolbar)
    close = Toolbar.close

    def fail_closing_toolbar(self: Toolbar) -> None:
        if self is toolbar and stage == "before":
            raise RuntimeError("toolbar close failed before canvas close")
        close(self)
        if self is toolbar and stage == "after":
            raise RuntimeError("toolbar close failed before canvas close")

    action = control.done if terminal == "done" else control.cancel
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Toolbar, "close", fail_closing_toolbar)
            with pytest.raises(
                RuntimeError, match="toolbar close failed before canvas"
            ):
                action()
        assert control.is_finished
        assert canvas.comm is None
        assert control.figure.canvas is not canvas
        assert exp.analysis is old_record
        assert exp.analysis_plots is old_plots
    finally:
        old_plots.release()
    assert set(widgets.Widget.widgets) == existing_widgets


def test_default_notebook_host_drags_and_releases_preview_widgets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    displayed: list[object] = []
    monkeypatch.setattr("IPython.display.display", displayed.append)
    existing_widgets = set(widgets.Widget.widgets)
    exp = FluxDepAnalyzer()
    control = exp.start(make_source(), FluxDepPickerOptions(-0.2, 0.3))
    try:
        assert displayed == [control.widget, control.figure.canvas]
        canvas = control.figure.canvas
        control.half_button.click()
        canvas.draw()
        ax = control.figure.axes[0]
        pixel_x, pixel_y = ax.transData.transform((-0.1, 5.1))
        canvas.callbacks.process(
            "motion_notify_event",
            MouseEvent("motion_notify_event", canvas, pixel_x, pixel_y),
        )
        assert control.positions()[0] == pytest.approx(-0.1, abs=0.01)
        preview = control.figure
        control.done_button.click()
        assert control.is_finished
        assert exp.analysis is not None
        assert exp.analysis.result.flux_half == pytest.approx(-0.1, abs=0.01)
        assert preview.canvas is not canvas
        exp.analysis.figures["pick"].savefig(tmp_path / "interactive-pick.png")
        assert (tmp_path / "interactive-pick.png").stat().st_size > 0
    finally:
        if not control.is_finished:
            control.cancel()
        if exp.analysis is not None:
            expect_analysis_plots(exp).release()
    assert set(widgets.Widget.widgets) == existing_widgets
