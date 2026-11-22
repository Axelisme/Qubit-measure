"""Read native Measure figures without leaving Data or saving first."""

import sys
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.backend_bases import RendererBase
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    PostAnalysis,
    PostAnalyzeRequest,
)
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.figure_export import DATA_PREVIEW_DPI, SAVE_DPI
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.plotting.plots import Plots

from tests._lab import synthetic_lab
from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    PaneParams,
    PaneSummary,
    PostSummary,
    SavingAdapter,
)
from tests.gui.app.measure.ui._qt_interactions import modal_action
from tests.program._profiles import load_soccfg


def _run(
    cfg: SavingCfg, *, context: RunContext, on_reading_draw: Artist | None = None
) -> RunRecord[SavingCfg, SavingResult]:
    record = SavingAdapter.run(cfg, context=context)
    _show_run_figure(context.plots, on_reading_draw=on_reading_draw)
    return record


def _show_run_figure(plots: Plots, *, on_reading_draw: Artist | None = None) -> None:
    figure, axes = plots.subplots(num="signal", layout="constrained")
    axes.set_facecolor("#eff9ff")
    axes.plot([0.0, 0.4, 0.8], [0.0, 0.8, 0.0], label="Mean I")
    axes.plot([0.0, 0.4, 0.8], [0.0, -0.7, 0.0], label="Mean Q")
    axes.set(xlabel="Time (µs)", ylabel="Amplitude (a.u.)", ylim=(-1, 1))
    axes.legend(loc="upper right")
    if on_reading_draw is not None:
        figure.add_artist(on_reading_draw)
    figure.show()


class _OnReadingDraw(Artist):
    """Run a callback only for the separate reading draw."""

    def __init__(self, callback: Callable[[], None]) -> None:
        super().__init__()
        self._callback = callback

    def draw(self, renderer: RendererBase) -> None:
        figure = self.get_figure()
        if (
            figure is not None
            and figure.canvas.is_saving()
            and not np.any(
                np.isclose(renderer.points_to_pixels(72), [DATA_PREVIEW_DPI, SAVE_DPI])
            )
        ):
            self._callback()


def _fail_draw() -> None:
    raise ValueError("Synthetic reading render failure")


class _ReadingDrawHook:
    callback: Callable[[], None] | None = None

    def draw_once(self) -> None:
        callback, self.callback = self.callback, None
        if callback is not None:
            callback()


@pytest.fixture
def reading_draw_hook() -> _ReadingDrawHook:
    return _ReadingDrawHook()


_LONG_MEMBERS = (
    "averaged_readout_demodulation_window_calibration_reference_in_phase",
    "averaged_readout_demodulation_window_calibration_reference_quadrature",
)


def _analyze(
    req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams], *, plots: Plots
) -> PaneSummary:
    for name in ("broken", "peak", *_LONG_MEMBERS, "signal"):
        figure, axes = plots.subplots(num=name, layout="constrained")
        if name == "broken":
            figure.add_artist(_OnReadingDraw(_fail_draw))
        if name == "signal":
            axes.set_facecolor("#fff2cf")
            axes.scatter([0, 0.4, 0.8], [1, 2, 1], label="Sweep samples")
            axes.axvline(
                0.4, color="tab:red", linestyle="--", label="Selected: 0.4000 µs"
            )
            axes.legend(loc="upper right")
        else:
            heat = axes.imshow([[0, 1], [2, 3]], extent=(0, 1, 4, 5), aspect="auto")
            figure.colorbar(heat, ax=axes, label="Mean IQ (a.u.)")
        axes.set(
            xlabel="Delay (µs)", ylabel="Frequency (MHz)", title=f"Analysis • {name}"
        )
    return PaneSummary()


def _post_analyze(
    req: PostAnalyzeRequest[
        RunRecord[SavingCfg, SavingResult], PaneSummary, PaneParams
    ],
    *,
    plots: Plots,
) -> PostSummary:
    _, axes = plots.subplots(num="signal", layout="constrained")
    axes.set_facecolor("#f4eafa")
    axes.plot([0, 1, 2], [0, 0.8, 0], label="Post trace")
    axes.set(xlabel="Time (µs)", ylabel="Amplitude (a.u.)")
    axes.legend(loc="upper right")
    return PostSummary()


@pytest.fixture
def reading_app(
    qapp: QApplication, tmp_path: Path, reading_draw_hook: _ReadingDrawHook
) -> Iterator[tuple[Controller, MainWindow]]:
    def run_with_draw_hook(
        cfg: SavingCfg, *, context: RunContext
    ) -> RunRecord[SavingCfg, SavingResult]:
        return _run(
            cfg,
            context=context,
            on_reading_draw=_OnReadingDraw(reading_draw_hook.draw_once),
        )

    registry = Registry()
    registry.register(
        "Figure reading",
        Adapter(
            spec=SavingAdapter.spec,
            make_form=SavingAdapter.make_form,
            run=run_with_draw_hook,
            guide=None,
        ),
    )
    registry.register(
        "Named figures",
        Adapter(
            spec=SavingAdapter.spec,
            make_form=SavingAdapter.make_form,
            run=run_with_draw_hook,
            guide=None,
            analysis=Analysis(fn=_analyze, options=lambda source, env: PaneParams()),
            post_analysis=PostAnalysis(fn=_post_analyze, options=None, writeback=None),
        ),
    )
    registry.register("Other tab", SavingAdapter)
    with _open_reading_app(qapp, tmp_path, registry) as app:
        yield app


@contextmanager
def _open_reading_app(
    qapp: QApplication, tmp_path: Path, registry: Registry
) -> Generator[tuple[Controller, MainWindow]]:
    previous_hook = sys.excepthook
    behavior = MeasureGuiBehavior(
        synthetic_lab(registry), clean=True, project_root=str(tmp_path)
    )
    assembly = behavior.assemble(None)
    ctrl, window = assembly.controller, assembly.window
    assert isinstance(ctrl, Controller) and isinstance(window, MainWindow)
    board = white_noise(load_soccfg(), std_adc=1.0, seed=17, routes=((0, 0),))
    try:
        behavior.before_show(assembly)
        ctrl.entry_control.new_entry(EntryRequest("reading"))
        ctrl.entry_control.new_point(
            PointCreateRequest("p", "Offline", CommandOrigin("gui"))
        )
        ctrl.session_services.soc_connection.install_soc(board.soc, board.soccfg)
        window.show()
        qapp.processEvents()
        yield ctrl, window
    finally:
        behavior.close(assembly)
        sys.excepthook = previous_hook
        board.devices.close_all_devices()
        qapp.processEvents()


@pytest.fixture
def reading_window(reading_app: tuple[Controller, MainWindow]) -> MainWindow:
    ctrl, window = reading_app
    ctrl.tab_control.new_tab("Figure reading")
    QApplication.processEvents()
    return window


@pytest.fixture
def failing_live_run(
    qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[MainWindow, Callable[[], None]]]:
    allow_failure = Event()

    def run(
        _cfg: SavingCfg, *, context: RunContext
    ) -> RunRecord[SavingCfg, SavingResult]:
        _show_run_figure(context.plots)
        if not allow_failure.wait(timeout=30):
            raise TimeoutError("Test did not release the synthetic Run")
        raise ValueError("Synthetic failure after live figure")

    registry = Registry()
    registry.register(
        "Failing live Run",
        Adapter(
            spec=SavingAdapter.spec,
            make_form=SavingAdapter.make_form,
            run=run,
            guide=None,
        ),
    )
    with _open_reading_app(qapp, tmp_path, registry) as (ctrl, window):
        ctrl.tab_control.new_tab("Failing live Run")
        qapp.processEvents()
        try:
            yield window, allow_failure.set
        finally:
            allow_failure.set()


def test_failed_live_run_clears_its_reading_without_leaving_data(
    failing_live_run: tuple[MainWindow, Callable[[], None]], tmp_path: Path
) -> None:
    window, release_run = failing_live_run
    flow = ComponentDriver(window)
    shots = tmp_path / "failed-live-run"
    flow.click("Run")
    qt_until(lambda: flow.legends().texts == ("Mean I", "Mean Q"), label="live figure")
    flow.screenshot(shots / "00-live-run.png")
    flow.select_tab("Data")
    flow.screenshot(shots / "01-live-preview.png")
    flow.click(object_name="dataFigureView/run/signal")
    assert flow.observe("Run — signal").visible
    assert flow.image_center_colour(object_name="dataFigureReadingImage") == "#eff9ff"
    flow.screenshot(shots / "02-live-reading.png")
    release_run()
    qt_until(
        lambda: any(d.title == "Run failed" for d in flow.dialogs()),
        label="failure delivered on the owner loop",
    )
    flow.screenshot(shots / "03-run-failed.png")
    assert flow.tabs("Data").current == "Data"
    assert not any(w.text == "Back to previews" for w in flow.visible().widgets)
    assert flow.observe("No figure yet").visible
    assert not any(w.text == "View figure" for w in flow.visible().widgets)


def _run_to_completion(flow: ComponentDriver) -> None:
    flow.select_tab("Run")
    flow.click("Run")
    qt_until(
        lambda: (
            flow.legends().texts == ("Mean I", "Mean Q")
            and any(w.text == "Run" and w.enabled for w in flow.visible().widgets)
        ),
        label="synthetic Run idle with figure",
    )


def test_data_reads_run_figure_in_place_and_returns_focus(
    reading_window: MainWindow, tmp_path: Path
) -> None:
    flow = ComponentDriver(reading_window)
    shots = tmp_path / "run-reading"
    flow.select_tab("Data")
    assert flow.observe("No figure yet").visible
    flow.screenshot(shots / "00-empty.png")
    _run_to_completion(flow)
    flow.screenshot(shots / "01-run.png")
    flow.select_tab("Data")
    flow.screenshot(shots / "02-overview.png")
    control = "dataFigureView/run/signal"
    flow.click(object_name=control)
    assert flow.observe("Run — signal").visible
    assert flow.observe("Save All").enabled
    assert flow.image_center_colour(object_name="dataFigureReadingImage") == "#eff9ff"
    flow.screenshot(shots / "03-reading.png")
    default_view = _reading_viewport(flow)
    flow.press_key(Qt.Key.Key_Space)
    assert flow.focused_input().object_name == control
    assert flow.observe(object_name="previewGalleryHeader").visible
    flow.screenshot(shots / "04-overview-return.png")
    flow.resize_window(860, 675)
    flow.screenshot(shots / "05-narrow-overview.png")
    flow.press_key(Qt.Key.Key_Space)
    assert flow.observe("Back to previews").visible
    flow.screenshot(shots / "06-narrow-reading.png")
    narrow_view = _reading_viewport(flow)
    _record_viewports(shots, default_view, narrow_view)
    _focus_reading_scroll(flow)
    _walk_narrow_image(flow, shots, start=7)
    flow.press_key(Qt.Key.Key_Escape)
    assert flow.focused_input().object_name == control
    flow.screenshot(shots / "11-narrow-return.png")
    assert flow.observe("○ NOT SAVED").visible
    flow.click("Save All")
    assert _save_completed(flow)
    assert flow.observe("✓ SAVED").visible
    flow.screenshot(shots / "12-saved.png")
    flow.click(object_name=control)
    flow.screenshot(shots / "13-before-tab-close.png")
    flow.click_tooltip("Close Tab")
    assert flow.observe("Use New Tab to start an experiment.").visible
    flow.screenshot(shots / "14-tab-closed.png")


def _save_completed(flow: ComponentDriver) -> bool:
    message = flow.status().text
    return message.startswith("Saved ") and " for run " in message


def _visible_bounds(
    flow: ComponentDriver, object_name: str
) -> tuple[int, int, int, int]:
    widget = flow.observe(object_name=object_name)
    assert widget.visible and widget.bounds is not None
    return widget.bounds


def _reading_viewport(flow: ComponentDriver) -> tuple[int, int]:
    viewport = flow.scroll_area(object_name="dataFigureReadingScroll").viewport_size
    assert viewport is not None
    return viewport


def _record_viewports(
    shots: Path, default_view: tuple[int, int], narrow_view: tuple[int, int]
) -> None:
    assert narrow_view[0] < default_view[0]
    (shots / "viewports.txt").write_text(
        f"reading viewport (width, height)\ndefault={default_view}\nnarrow={narrow_view}\n",
        encoding="utf-8",
    )


def _focus_reading_scroll(flow: ComponentDriver) -> None:
    flow.press_key(Qt.Key.Key_Tab)
    assert flow.focused_input().object_name == "dataFigureReadingScroll"


def _pan(flow: ComponentDriver, key: Qt.Key, times: int) -> None:
    for _ in range(times):
        flow.press_key(key)


def _walk_narrow_image(flow: ComponentDriver, shots: Path, *, start: int) -> None:
    assert flow.scroll_area(object_name="dataFigureReadingScroll").vertical_scrollable
    image_name = "dataFigureReadingImage"
    left, top, _, _ = _visible_bounds(flow, image_name)
    _pan(flow, Qt.Key.Key_Down, 11)
    assert _visible_bounds(flow, image_name)[1] < top
    flow.screenshot(shots / f"{start - 1:02d}b-narrow-middle-left.png")
    _pan(flow, Qt.Key.Key_PageDown, 2)
    flow.screenshot(shots / f"{start:02d}-narrow-bottom-left.png")
    _pan(flow, Qt.Key.Key_Right, 24)
    flow.screenshot(shots / f"{start + 1:02d}-narrow-bottom-middle.png")
    _pan(flow, Qt.Key.Key_Right, 60)
    assert _visible_bounds(flow, image_name)[0] < left
    flow.screenshot(shots / f"{start + 2:02d}-narrow-bottom-right.png")
    _pan(flow, Qt.Key.Key_Up, 11)
    flow.screenshot(shots / f"{start + 2:02d}b-narrow-middle-right.png")
    _pan(flow, Qt.Key.Key_PageUp, 2)
    flow.screenshot(shots / f"{start + 3:02d}-narrow-top-right.png")


def test_switching_experiment_tabs_dismisses_data_reading(
    reading_app: tuple[Controller, MainWindow], tmp_path: Path
) -> None:
    ctrl, window = reading_app
    ctrl.tab_control.new_tab("Other tab")
    ctrl.tab_control.new_tab("Figure reading")
    QApplication.processEvents()
    flow = ComponentDriver(window)
    shots = tmp_path / "tab-switch"
    _run_to_completion(flow)
    flow.screenshot(shots / "00-run.png")
    flow.select_tab("Data")
    flow.screenshot(shots / "01-data.png")
    flow.click(object_name="dataFigureView/run/signal")
    assert flow.observe("Run — signal").visible
    flow.screenshot(shots / "02-reading.png")
    flow.select_tab("Other tab")
    flow.screenshot(shots / "03-other-tab.png")
    flow.select_tab("Figure reading")
    flow.screenshot(shots / "04-returned.png")
    assert any(w.object_name == "previewGalleryHeader" for w in flow.visible().widgets)


@pytest.fixture
def named_window(reading_app: tuple[Controller, MainWindow]) -> MainWindow:
    ctrl, window = reading_app
    ctrl.tab_control.new_tab("Named figures")
    QApplication.processEvents()
    return window


@pytest.mark.parametrize("reopen", [False, True], ids=["leave", "replace"])
def test_reentrant_draw_cannot_present_after_its_reading_is_invalidated(
    named_window: MainWindow,
    reading_draw_hook: _ReadingDrawHook,
    tmp_path: Path,
    *,
    reopen: bool,
) -> None:
    flow = ComponentDriver(named_window)
    shots = tmp_path / "reentrant-reading"
    _run_to_completion(flow)
    if reopen:
        _analyze_to_completion(flow)
    flow.select_tab("Data")
    if reopen:
        flow.click("Next page")
    control = "dataFigureView/run/signal"
    flow.screenshot(shots / "00-before-draw.png")

    def during_draw() -> None:
        if reopen:
            flow.click(object_name="dataFigureView/analysis/signal")
            _focus_reading_scroll(flow)
            flow.screenshot(shots / "01-reopened-during-draw.png")
            _fail_draw()
        flow.select_tab("Guide")
        flow.screenshot(shots / "01-left-during-draw.png")
        flow.select_tab("Data")
        flow.screenshot(shots / "02-returned-during-draw.png")

    reading_draw_hook.callback = during_draw
    flow.click(object_name=control)
    flow.screenshot(shots / "03-original-draw-returned.png")
    if reopen:
        assert flow.observe("Analysis — signal").visible
        assert not any(
            w.text == "Unable to render this figure." for w in flow.visible().widgets
        )
        assert (
            flow.image_center_colour(object_name="dataFigureReadingImage") == "#fff2cf"
        )
        assert flow.focused_input().object_name == "dataFigureReadingScroll"
    else:
        assert flow.observe(object_name="previewGalleryHeader").visible
        assert flow.observe(object_name=control).visible


def _analyze_to_completion(flow: ComponentDriver) -> None:
    flow.select_tab("Analysis")
    flow.click("Analyze")
    qt_until(
        lambda: (
            any(w.text == "Analyze" and w.enabled for w in flow.visible().widgets)
            and flow.observe("1 / 2").visible
        ),
        label="synthetic Analyze idle with five figures",
    )


def test_data_reads_selected_stage_and_member_and_restores_gallery_page(
    named_window: MainWindow, tmp_path: Path
) -> None:
    flow = ComponentDriver(named_window)
    shots = tmp_path / "named-reading"
    _run_to_completion(flow)
    _analyze_to_completion(flow)
    flow.select_tab("Data")
    flow.screenshot(shots / "00-grouped.png")
    flow.click("Next page")
    flow.screenshot(shots / "01-page-two.png")
    control = "dataFigureView/analysis/signal"
    flow.click(object_name=control)
    assert flow.observe("Analysis — signal").visible
    assert flow.image_center_colour(object_name="dataFigureReadingImage") == "#fff2cf"
    flow.screenshot(shots / "02-analysis-signal.png")
    _focus_reading_scroll(flow)
    _pan(flow, Qt.Key.Key_Right, 60)
    flow.screenshot(shots / "03-analysis-legend.png")
    flow.resize_window(860, 675)
    _pan(flow, Qt.Key.Key_Right, 60)
    flow.screenshot(shots / "03b-narrow-analysis-legend.png")
    flow.resize_window(1280, 750)
    flow.screenshot(shots / "03c-default-width-restored.png")
    flow.click("Back to previews")
    assert flow.observe("2 / 2").visible
    assert flow.focused_input().object_name == control
    flow.screenshot(shots / "04-page-two-restored.png")
    flow.press_key(Qt.Key.Key_Space)
    flow.screenshot(shots / "05-keyboard-open.png")
    flow.select_tab("Guide")
    flow.screenshot(shots / "06-guide.png")
    flow.select_tab("Data")
    assert flow.observe("2 / 2").visible
    flow.screenshot(shots / "07-data-restored.png")
    flow.click(object_name="dataFigureView/run/signal")
    assert flow.observe("Run — signal").visible
    assert flow.image_center_colour(object_name="dataFigureReadingImage") == "#eff9ff"
    flow.screenshot(shots / "08-same-name-run.png")
    flow.select_tab("Post-Analysis")
    flow.click("Run Post-Analysis")
    qt_until(
        lambda: (
            flow.legends().texts == ("Post trace",)
            and any(
                w.text == "Run Post-Analysis" and w.enabled
                for w in flow.visible().widgets
            )
        ),
        label="synthetic Post-Analysis idle with figure",
    )
    flow.screenshot(shots / "09-post-analysis.png")
    flow.select_tab("Data")
    flow.screenshot(shots / "10-three-stages.png")
    flow.press_key(Qt.Key.Key_PageDown, object_name="previewGalleryScroll")
    flow.screenshot(shots / "11-post-stage.png")
    flow.click(object_name="dataFigureView/post_analysis/signal")
    assert flow.observe("Post-Analysis — signal").visible
    assert flow.image_center_colour(object_name="dataFigureReadingImage") == "#f4eafa"
    flow.screenshot(shots / "12-same-name-post.png")


def test_long_member_names_fit_in_gallery_and_reading_at_narrow_width(
    named_window: MainWindow, tmp_path: Path
) -> None:
    flow = ComponentDriver(named_window)
    shots = tmp_path / "long-member-reading"
    _run_to_completion(flow)
    _analyze_to_completion(flow)
    flow.select_tab("Data")
    flow.resize_window(860, 675)
    flow.screenshot(shots / "00-narrow-overview.png")
    flow.press_key(Qt.Key.Key_PageDown, object_name="previewGalleryScroll")
    for _ in range(3):
        flow.press_key(Qt.Key.Key_Down, object_name="previewGalleryScroll")
    flow.screenshot(shots / "01-analysis-group.png")
    captions = {w.text.replace("\n", ""): w for w in flow.visible().widgets}
    for name in _LONG_MEMBERS:
        assert captions[name].caption_fits and captions[name].fully_visible

    first, second = (f"dataFigureView/analysis/{name}" for name in _LONG_MEMBERS)
    flow.click(object_name="dataFigureView/analysis/peak")
    flow.screenshot(shots / "02-peak-open.png")
    short_caption_width, _ = _reading_viewport(flow)
    flow.press_key(Qt.Key.Key_Escape)
    assert flow.focused_input().object_name == "dataFigureView/analysis/peak"
    flow.screenshot(shots / "03-back-to-peak.png")
    flow.press_key(Qt.Key.Key_Tab)
    assert flow.focused_input().object_name == first
    flow.screenshot(shots / "04-first-member-focused.png")
    flow.press_key(Qt.Key.Key_Tab)
    assert flow.focused_input().object_name == second
    flow.screenshot(shots / "05-second-member-focused.png")
    flow.press_key(Qt.Key.Key_Space)
    flow.screenshot(shots / "06-quadrature-open.png")
    long_caption_width, _ = _reading_viewport(flow)
    assert long_caption_width == short_caption_width
    heading = f"Analysis — {_LONG_MEMBERS[1]}"
    caption = next(
        w for w in flow.visible().widgets if w.text.replace("\n", "") == heading
    )
    assert caption.caption_fits and caption.fully_visible
    flow.press_key(Qt.Key.Key_Escape)
    assert flow.focused_input().object_name == second
    flow.screenshot(shots / "07-preview-restored.png")


def test_data_render_error_does_not_block_save_and_load_clears_reading(
    named_window: MainWindow, tmp_path: Path, qapp: QApplication
) -> None:
    flow = ComponentDriver(named_window)
    shots = tmp_path / "failure-and-load"
    _run_to_completion(flow)
    _analyze_to_completion(flow)
    flow.select_tab("Data")
    flow.screenshot(shots / "00-overview.png")
    assert flow.observe("broken").caption_fits
    flow.click(object_name="dataFigureView/analysis/broken")
    assert flow.observe("Analysis — broken").visible
    assert flow.observe("Unable to render this figure.").visible
    assert flow.observe("Save All").enabled
    flow.screenshot(shots / "01-reading-error.png")
    flow.click("Save All")
    assert _save_completed(flow)
    flow.screenshot(shots / "02-saved-after-error.png")
    if any(w.text == "Back to previews" for w in flow.visible().widgets):
        flow.click("Back to previews")
    flow.click(object_name="dataFigureView/analysis/peak")
    assert flow.observe("Analysis — peak").visible
    flow.screenshot(shots / "03-other-member.png")
    default_view = _reading_viewport(flow)
    _focus_reading_scroll(flow)
    _pan(flow, Qt.Key.Key_Right, 60)
    flow.screenshot(shots / "04-heatmap-colourbar.png")
    flow.resize_window(860, 675)
    _pan(flow, Qt.Key.Key_Left, 60)
    flow.screenshot(shots / "05-narrow-heatmap.png")
    narrow_view = _reading_viewport(flow)
    _record_viewports(shots, default_view, narrow_view)
    _walk_narrow_image(flow, shots, start=6)

    def choose(dialog: RunLoadDialog) -> None:
        chooser = ComponentDriver(dialog)
        (row,) = chooser.table_rows(object_name="load_runs")
        chooser.screenshot(shots / "10-load-picker.png")
        chooser.select_table_row(
            row.columns[0], object_name="load_runs", double_click=True
        )

    modal_action(qapp, lambda: flow.click("Load Data"), [(RunLoadDialog, choose)])
    assert flow.status().text.startswith("Loaded data from ")
    assert flow.observe(object_name="previewGalleryHeader").visible
    assert not any(w.text == "Back to previews" for w in flow.visible().widgets)
    assert not any(w.text == "View figure" for w in flow.visible().widgets)
    flow.screenshot(shots / "11-loaded-figures-cleared.png")
