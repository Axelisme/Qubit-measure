"""Read native Measure figures without leaving Data or saving first."""

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.backend_bases import RendererBase
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import load
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


def _run(cfg: SavingCfg, *, context: RunContext) -> RunRecord[SavingCfg, SavingResult]:
    record = SavingAdapter.run(cfg, context=context)
    figure, axes = context.plots.subplots(num="signal", layout="constrained")
    axes.set_facecolor("#eff9ff")
    axes.plot([0.0, 0.4, 0.8], [0.0, 0.8, 0.0], label="Mean I")
    axes.plot([0.0, 0.4, 0.8], [0.0, -0.7, 0.0], label="Mean Q")
    axes.set(xlabel="Time (µs)", ylabel="Amplitude (a.u.)", ylim=(-1, 1))
    axes.legend(loc="upper right")
    figure.show()
    return record


class _ReadingDrawFailure(Artist):
    """Let native, thumbnail and Save draws work; fail the separate reading draw."""

    def draw(self, renderer: RendererBase) -> None:
        figure = self.get_figure()
        if (
            figure is not None
            and figure.canvas.is_saving()
            and not np.any(
                np.isclose(renderer.points_to_pixels(72), [DATA_PREVIEW_DPI, SAVE_DPI])
            )
        ):
            raise ValueError("Synthetic reading render failure")


def _analyze(
    req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams], *, plots: Plots
) -> PaneSummary:
    for name in ("broken", "peak", "band", "baseline", "signal"):
        figure, axes = plots.subplots(num=name, layout="constrained")
        if name == "broken":
            figure.add_artist(_ReadingDrawFailure())
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
    qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[Controller, MainWindow]]:
    registry = Registry()
    registry.register(
        "Figure reading",
        Adapter(
            spec=SavingAdapter.spec,
            make_form=SavingAdapter.make_form,
            run=_run,
            guide=None,
        ),
    )
    registry.register(
        "Named figures",
        Adapter(
            spec=SavingAdapter.spec,
            make_form=SavingAdapter.make_form,
            run=_run,
            guide=None,
            analysis=Analysis(fn=_analyze, options=lambda source, env: PaneParams()),
            post_analysis=PostAnalysis(fn=_post_analyze, options=None, writeback=None),
        ),
    )
    registry.register("Other tab", SavingAdapter)
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
    database = tmp_path / "Database" / "reading"
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
    default_view = flow.observe(object_name="dataFigureReadingViewport").bounds
    flow.press_key(Qt.Key.Key_Space)
    assert flow.focused_input().object_name == control
    assert flow.observe(object_name="previewGalleryHeader").visible
    flow.screenshot(shots / "04-overview-return.png")
    flow.resize_window(860, 675)
    flow.screenshot(shots / "05-narrow-overview.png")
    flow.press_key(Qt.Key.Key_Space)
    assert flow.observe("Back to previews").visible
    flow.screenshot(shots / "06-narrow-reading.png")
    narrow_view = flow.observe(object_name="dataFigureReadingViewport").bounds
    (shots / "viewports.txt").write_text(
        f"viewport global bounds (x, y, width, height)\ndefault={default_view}\nnarrow={narrow_view}\n",
        encoding="utf-8",
    )
    _scroll_to_far_edge(flow, shots)
    flow.press_key(Qt.Key.Key_Escape)
    assert flow.focused_input().object_name == control
    flow.screenshot(shots / "10-narrow-return.png")
    assert not list(database.glob("runs/*/data.h5"))
    flow.click("Save All")
    flow.screenshot(shots / "11-saved.png")
    assert sorted(path.suffix for path in database.rglob("*") if path.is_file()) == [
        ".h5",
        ".hdf5",
    ]
    (source,) = database.glob("runs/*/data.h5")
    restored = SavingAdapter.restore(load(source)).record
    np.testing.assert_array_equal(restored.result.signal, [1 + 2j, 3 - 4j, 5 + 6j])
    flow.click(object_name=control)
    flow.screenshot(shots / "12-before-tab-close.png")
    flow.click_tooltip("Close Tab")
    assert flow.observe("Use New Tab to start an experiment.").visible
    flow.screenshot(shots / "13-tab-closed.png")


def _scroll_to_far_edge(flow: ComponentDriver, shots: Path) -> None:
    assert flow.scroll_area(object_name="dataFigureReadingScroll").vertical_scrollable
    flow.press_key(Qt.Key.Key_Tab)
    assert flow.focused_input().object_name == "dataFigureReadingScroll"
    for _ in range(2):
        flow.press_key(Qt.Key.Key_PageDown)
    flow.screenshot(shots / "07-narrow-bottom-left.png")
    for _ in range(60):
        flow.press_key(Qt.Key.Key_Right)
    flow.screenshot(shots / "08-narrow-bottom-right.png")
    for _ in range(2):
        flow.press_key(Qt.Key.Key_PageUp)
    flow.screenshot(shots / "09-narrow-top-right.png")


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
    flow.press_key(Qt.Key.Key_Tab)
    for _ in range(60):
        flow.press_key(Qt.Key.Key_Right)
    flow.screenshot(shots / "03-analysis-legend.png")
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
    (native,) = (tmp_path / "Database").glob("**/data.h5")
    assert {path.name for path in (native.parent / "figures").glob("*.png")} == {
        f"analysis_results_{name}.png"
        for name in ("broken", "peak", "band", "baseline", "signal")
    }
    flow.screenshot(shots / "02-saved-after-error.png")
    if any(w.text == "Back to previews" for w in flow.visible().widgets):
        flow.click("Back to previews")
    flow.click(object_name="dataFigureView/analysis/peak")
    assert flow.observe("Analysis — peak").visible
    flow.screenshot(shots / "03-other-member.png")
    flow.press_key(Qt.Key.Key_Tab)
    for _ in range(60):
        flow.press_key(Qt.Key.Key_Right)
    flow.screenshot(shots / "04-heatmap-colourbar.png")

    def choose(dialog: RunLoadDialog) -> None:
        chooser = ComponentDriver(dialog)
        (row,) = chooser.table_rows(object_name="load_runs")
        chooser.screenshot(shots / "05-load-picker.png")
        chooser.select_table_row(
            row.columns[0], object_name="load_runs", double_click=True
        )

    modal_action(qapp, lambda: flow.click("Load Data"), [(RunLoadDialog, choose)])
    assert flow.status().text.startswith("Loaded data from ")
    assert flow.observe(object_name="previewGalleryHeader").visible
    assert not any(w.text == "Back to previews" for w in flow.visible().widgets)
    assert not any(w.text == "View figure" for w in flow.visible().widgets)
    flow.screenshot(shots / "06-loaded-figures-cleared.png")
