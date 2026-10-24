"""Independent Notebook flux picks retain explicit sources and native figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pytest
from ipympl.backend_nbagg import Canvas, Toolbar
from matplotlib.backend_bases import MouseEvent
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickAnalysis,
    FluxPickInputs,
    FluxPickState,
)
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.records import RunRecord
from zcu_tools.notebook.flux_dep import (
    FluxDepAnalysisRecord,
    FluxDepAnalyzer,
    FluxDepPickerOptions,
)
from zcu_tools.plotting.plots import NonPresentingHost, PlotHost, Plots


@dataclass
class Spectrum:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def make_result() -> Spectrum:
    values = np.linspace(-0.5, 0.5, 9)
    freqs = np.linspace(4.8, 5.4, 7)
    signals = np.asarray(
        np.sin(values[:, None] * 7 + freqs[None, :] * 9)
        + 1j * np.cos(values[:, None] * 3 - freqs[None, :] * 7),
        dtype=np.complex128,
    )
    return Spectrum(values, freqs, signals)


def make_source() -> RunRecord[ConfigBase, Spectrum]:
    return RunRecord(cfg=None, result=make_result())


def picker_inputs(source: RunRecord[ConfigBase, Spectrum]) -> FluxPickInputs:
    data = source.result
    return FluxPickInputs(data.signals, data.values, data.freqs)


@pytest.fixture(autouse=True)
def suppress_notebook_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "IPython.display.display",
        lambda _widget: None,
    )


def expect_analysis_plots(analyzer: FluxDepAnalyzer[ConfigBase, Spectrum]) -> Plots:
    plots = analyzer.analysis_plots
    assert plots is not None
    return plots


def completed_analyzer(
    host: PlotHost | None = None,
) -> tuple[
    FluxDepAnalyzer[ConfigBase, Spectrum],
    FluxDepAnalysisRecord[ConfigBase, Spectrum],
    Plots,
]:
    analyzer = FluxDepAnalyzer[ConfigBase, Spectrum](host)
    record = analyzer.start(
        make_source(), picker_inputs(make_source()), FluxDepPickerOptions(-0.2, 0.3)
    ).done()
    return analyzer, record, expect_analysis_plots(analyzer)


def test_done_publishes_source_options_numeric_result_and_named_figure(
    tmp_path,
) -> None:
    analyzer = FluxDepAnalyzer[ConfigBase, Spectrum](NonPresentingHost())
    source = make_source()
    original_signals = source.result.signals.copy()
    control = analyzer.start(
        source,
        picker_inputs(source),
        FluxDepPickerOptions(-0.2, 0.3, magnitude_only=True),
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
    source: RunRecord[ConfigBase, Spectrum] = RunRecord(
        cfg=None,
        result=Spectrum(data.values, data.freqs, data.signals),
    )
    analyzer = FluxDepAnalyzer[ConfigBase, Spectrum](NonPresentingHost())
    first = analyzer.start(
        source, picker_inputs(source), FluxDepPickerOptions(-0.2, 0.3)
    )
    other: RunRecord[ConfigBase, Spectrum] = RunRecord(
        cfg=None,
        result=Spectrum(data.values, data.freqs + 10.0, data.signals * 2),
    )
    second = analyzer.start(
        other, picker_inputs(other), FluxDepPickerOptions(-0.1, 0.4)
    )
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


def test_invalid_done_and_cancel_leave_old_analysis_and_editable_control() -> None:
    analyzer = FluxDepAnalyzer[ConfigBase, Spectrum](NonPresentingHost())
    first = analyzer.start(
        make_source(), picker_inputs(make_source()), FluxDepPickerOptions(-0.2, 0.3)
    )
    old = first.done()
    old_plots = analyzer.analysis_plots
    assert old_plots is not None
    next_control = analyzer.start(
        make_source(), picker_inputs(make_source()), FluxDepPickerOptions(0.0, 0.0)
    )
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


@pytest.mark.parametrize("fail_on", [1, 2])
def test_failed_notebook_publication_cleans_up_widgets_and_keeps_record(
    monkeypatch: pytest.MonkeyPatch, fail_on: int
) -> None:
    exp = FluxDepAnalyzer[ConfigBase, Spectrum]()
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
        exp.start(source, picker_inputs(source), FluxDepPickerOptions(-0.2, 0.3))
    assert exp.analysis is None
    assert set(widgets.Widget.widgets) == existing_widgets


def test_failed_final_pick_publication_retires_preview_without_losing_old_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("IPython.display.display", lambda _widget: None)
    initial_widgets = set(widgets.Widget.widgets)
    exp, old_record, old_plots = completed_analyzer()
    before_new = set(widgets.Widget.widgets)
    control = exp.start(
        make_source(), picker_inputs(make_source()), FluxDepPickerOptions(-0.1, 0.4)
    )

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


@pytest.mark.parametrize("terminal", ["done", "cancel"])
@pytest.mark.parametrize("stage", ["before", "after"])
def test_toolbar_failure_before_preview_canvas_close_preserves_old_record(
    monkeypatch: pytest.MonkeyPatch, terminal: str, stage: str
) -> None:
    monkeypatch.setattr("IPython.display.display", lambda _widget: None)
    existing_widgets = set(widgets.Widget.widgets)
    exp, old_record, old_plots = completed_analyzer()
    control = exp.start(
        make_source(), picker_inputs(make_source()), FluxDepPickerOptions(-0.1, 0.4)
    )
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
    exp = FluxDepAnalyzer[ConfigBase, Spectrum]()
    control = exp.start(
        make_source(), picker_inputs(make_source()), FluxDepPickerOptions(-0.2, 0.3)
    )
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
