"""User-visible experiment pane switching and named figure presentation contracts."""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    PostAnalysis,
    PostAnalyzeRequest,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.interactive_frontend import InteractiveFrontend
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.plotting.figures import FigureCollection
from zcu_tools.plotting.plots import Plots

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._app_launch import launched_measure_app
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    PaneParams,
    PaneSummary,
    PostSummary,
)
from tests.gui.app.measure.ui._figure_presentation_support import gallery


class StaticInteractive(InteractiveFrontend):
    """Synthetic presentation with no worker or editing actions to quiesce."""

    def __init__(self) -> None:
        super().__init__()
        self._figure = Figure()
        axes = self._figure.subplots()
        axes.plot([0, 1, 2], [1, 0, 1], label="Interactive preview")
        axes.set_xlabel("Time")
        axes.legend()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Interactive preview"))
        layout.addWidget(FigureCanvasQTAgg(self._figure))

    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def preview_active(self) -> bool:
        return False

    def set_read_only(self, *, read_only: bool) -> None:
        pass

    def cancel_preview(self) -> None:
        pass

    def teardown(self) -> None:
        pass


def LayoutAdapter_draw_figures(plots: Plots, count: int, prefix: str) -> None:
    for index in range(1, count + 1):
        name = f"{prefix} tone {index}"
        plots.subplots(name)

        def draw(figure: Figure, name: str = name, height: int = index) -> None:
            axes = figure.axes[0]
            axes.plot([0, 1, 2], [0, height, 0], label=name)
            axes.set_xlabel("Time")
            axes.legend()

        plots.draw(name, draw)


def PresentationAdapter_run(cfg: SavingCfg, *, context: RunContext) -> SavingResult:
    record = AnalysisSavingAdapter.run(cfg, context=context)
    context.plots.subplots("signal")

    def draw(figure: Figure) -> None:
        axes = figure.axes[0]
        axes.plot([0, 1, 2], [0, 1, 0], label="Signal")
        axes.set_xlabel("Time")
        axes.legend()

    context.plots.draw("signal", draw)
    return record


def PresentationAdapter_analyze(
    req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams], *, plots: Plots
) -> PaneSummary:
    for name, values in (("fit", [0, 1, 0]), ("residual", [0, -1, 0])):
        _, axes = plots.subplots(name)
        axes.plot([0, 1, 2], values, label=name)
        axes.set_xlabel("Time")
        axes.legend()
    return PaneSummary()


def PresentationAdapter_post_analyze(
    req: PostAnalyzeRequest[
        RunRecord[SavingCfg, SavingResult], PaneSummary, PaneParams
    ],
    *,
    plots: Plots,
) -> PostSummary:
    for name, values in (("summary", [1, 0, 1]), ("detail", [-1, 0, -1])):
        _, axes = plots.subplots(name)
        axes.plot([0, 1, 2], values, label=name)
        axes.set_xlabel("Time")
        axes.legend()
    return PostSummary()


PresentationAdapter = Adapter(
    spec=AnalysisSavingAdapter.spec,
    make_form=AnalysisSavingAdapter.make_form,
    run=PresentationAdapter_run,
    guide=None,
    analysis=Analysis(
        fn=PresentationAdapter_analyze,
        options=lambda source, env: PaneParams(),
        writeback=AnalysisSavingAdapter.get_writeback_items,
    ),
    post_analysis=PostAnalysis(
        fn=PresentationAdapter_post_analyze, options=None, writeback=None
    ),
)


@pytest.fixture
def presentation_window(qapp, tmp_path: Path) -> Iterator[QWidget]:
    registry = Registry()
    registry.register("presentation", PresentationAdapter)
    with launched_measure_app(qapp, tmp_path, registry=registry) as app:
        # Public fixture bootstrap only; the scenario below uses visible controls.
        app.controller.setup_control.new_entry(EntryRequest("presentation"))
        app.controller.setup_control.new_point(
            PointCreateRequest("p", "Offline presentation", CommandOrigin("gui"))
        )
        app.controller.new_tab("presentation")
        app.window.resize(1280, 800)
        yield app.window


@pytest.fixture
def layout_window(qapp, tmp_path: Path) -> Iterator[QWidget]:
    layout_counts = [3, 3]

    def LayoutAdapter_run(cfg: SavingCfg, *, context: RunContext) -> SavingResult:
        record = AnalysisSavingAdapter.run(cfg, context=context)
        layout_counts[0] += 1
        LayoutAdapter_draw_figures(context.plots, layout_counts[0], "Run")
        return record

    def LayoutAdapter_analyze(
        req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams],
        *,
        plots: Plots,
    ) -> PaneSummary:
        layout_counts[1] += 1
        LayoutAdapter_draw_figures(plots, layout_counts[1], "Analysis")
        return PaneSummary()

    LayoutAdapter = Adapter(
        spec=AnalysisSavingAdapter.spec,
        make_form=AnalysisSavingAdapter.make_form,
        run=LayoutAdapter_run,
        guide=None,
        analysis=Analysis(
            fn=LayoutAdapter_analyze,
            options=lambda source, env: PaneParams(),
            writeback=AnalysisSavingAdapter.get_writeback_items,
        ),
        post_analysis=AnalysisSavingAdapter.post_analysis,
    )

    registry = Registry()
    registry.register("layout", LayoutAdapter)
    with launched_measure_app(qapp, tmp_path, registry=registry) as app:
        app.controller.setup_control.new_entry(EntryRequest("layout"))
        app.controller.setup_control.new_point(
            PointCreateRequest("p", "Offline layout", CommandOrigin("gui"))
        )
        app.controller.new_tab("layout")
        app.window.resize(1440, 900)
        yield app.window


@pytest.mark.parametrize("pane", ["Run", "Analysis", "Data"])
def test_user_sees_four_figures_then_pages_through_five(
    layout_window: QWidget, pane: str, tmp_path: Path
) -> None:
    flow = ComponentDriver(layout_window)
    shots = tmp_path / f"layout-{pane}"
    flow.click("Close")
    flow.screenshot(shots / "01-ready.png")
    flow.click("Run")
    qt_until(
        lambda: any(w.text == "Run" and w.enabled for w in flow.visible().widgets),
        label="synthetic run settled",
    )
    action = "Run"
    if pane == "Analysis":
        flow.select_tab("Analysis")
        action = "Analyze"
        flow.click(action)
        qt_until(
            lambda: any(w.text == action and w.enabled for w in flow.visible().widgets),
            label="synthetic analysis settled",
        )
    prefix = "Run" if pane == "Data" else pane
    if pane == "Data":
        flow.select_tab("Data")
    flow.screenshot(shots / "02-four.png")
    if pane == "Data":
        assert {f"Run tone {i}" for i in range(1, 5)} <= {
            widget.text for widget in flow.visible().widgets
        }
    else:
        assert flow.legends().texts == tuple(f"{prefix} tone {i}" for i in range(1, 5))
    assert not any(
        w.kind == "QComboBox" and w.text == f"{prefix} tone 1"
        for w in flow.visible().widgets
    )
    if pane == "Data":
        flow.select_tab("Run")
    flow.click(action)
    qt_until(
        lambda: any(w.text == action and w.enabled for w in flow.visible().widgets),
        label="five-figure operation settled",
    )
    if pane == "Data":
        flow.select_tab("Data")
    flow.screenshot(shots / "03-five.png")
    if pane != "Data":
        assert flow.legends().texts == tuple(f"{prefix} tone {i}" for i in range(1, 5))
    assert flow.observe("1 / 2").visible
    flow.click("Next page")
    assert flow.observe("2 / 2").visible
    if pane == "Data":
        assert flow.observe("Run tone 5").visible
    else:
        assert flow.legends().texts == (f"{prefix} tone 5",)
    flow.screenshot(shots / "04-second-page.png")
    flow.click("Previous page")
    assert flow.observe("1 / 2").visible
    flow.screenshot(shots / "05-first-page.png")


@pytest.fixture(params=["mounted", "unmounted"])
def interactive_presentation(
    presentation_window: QWidget, request
) -> tuple[QWidget, str]:
    """Prepare public mounted/unmounted states; no user-driven lifecycle is claimed."""
    (tab,) = presentation_window.findChildren(ExpTabWidget)
    tab.mount_interactive_widget(StaticInteractive())
    if request.param == "unmounted":
        tab.unmount_interactive_widgets(StaticInteractive)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return presentation_window, request.param


def test_interactive_presentation_and_placeholder_survive_user_pane_switches(
    interactive_presentation: tuple[QWidget, str], tmp_path: Path
) -> None:
    window, state = interactive_presentation
    flow = ComponentDriver(window)
    shots = tmp_path / state
    flow.click("Close")
    flow.select_tab("Analysis")
    expected = ("Interactive preview",) if state == "mounted" else ()
    assert flow.legends().texts == expected
    if state == "unmounted":
        assert flow.observe("(no plot yet)").visible
    flow.screenshot(shots / "01-analysis.png")
    flow.select_tab("Data")
    assert flow.observe(object_name="dataFigurePreviewGallery").visible
    flow.screenshot(shots / "02-data.png")
    flow.select_tab("Guide")
    assert flow.legends().texts == ()
    flow.screenshot(shots / "03-guide.png")
    flow.select_tab("Analysis")
    assert flow.legends().texts == expected
    if state == "unmounted":
        assert flow.observe("(no plot yet)").visible
    flow.screenshot(shots / "04-analysis-return.png")


def test_user_switches_named_figures_gallery_and_left_panel(
    presentation_window: QWidget, tmp_path: Path
) -> None:
    flow = ComponentDriver(presentation_window)
    shots = tmp_path / "presentation"
    flow.click("Close")
    flow.screenshot(shots / "01-ready.png")
    flow.click("Run")
    # The synthetic adapter completes its draw and returns, restoring the idle button.
    qt_until(
        lambda: (
            flow.legends().texts == ("Signal",)
            and any(
                w.text == "Run" and w.enabled and w.visible
                for w in flow.visible().widgets
            )
        ),
        label="offline Run presentation",
    )
    flow.screenshot(shots / "02-run.png")
    flow.select_tab("Analysis")
    flow.click("Analyze")
    # Both figures are committed by this finite offline analysis.
    qt_until(
        lambda: (
            flow.legends().texts == ("fit", "residual")
            and flow.observe("Analyze").enabled
        ),
        label="offline analysis presentation",
    )
    flow.screenshot(shots / "03-analysis-fit.png")
    assert flow.legends().texts == ("fit", "residual")
    flow.screenshot(shots / "04-analysis-together.png")
    flow.select_tab("Post-Analysis")
    flow.click("Run Post-Analysis")
    qt_until(
        lambda: (
            flow.legends().texts == ("summary", "detail")
            and flow.observe("Run Post-Analysis").enabled
        ),
        label="offline post-analysis presentation",
    )
    flow.screenshot(shots / "05-post-summary.png")
    assert flow.legends().texts == ("summary", "detail")
    flow.screenshot(shots / "06-post-together.png")
    flow.select_tab("Data")
    assert flow.observe(object_name="dataFigurePreviewGallery").visible
    assert flow.observe("Save All").enabled
    flow.screenshot(shots / "07-data.png")
    flow.click_tooltip("Collapse left panel")
    assert not any(w.text == "Save All" for w in flow.visible().widgets)
    flow.screenshot(shots / "08-data-collapsed.png")
    flow.click_tooltip("Expand left panel")
    assert flow.observe("Save All").visible
    flow.screenshot(shots / "09-data-expanded.png")
    flow.select_tab("Guide")
    assert not flow.observe(object_name="dataFigurePreviewGallery").visible
    assert flow.legends().texts == ()
    flow.screenshot(shots / "10-guide.png")
    flow.select_tab("Analysis")
    assert flow.legends().texts == ("fit", "residual")
    flow.screenshot(shots / "11-analysis-retained.png")
    flow.select_tab("Post-Analysis")
    assert flow.legends().texts == (
        "summary",
        "detail",
    )
    flow.screenshot(shots / "12-post-retained.png")


@pytest.mark.parametrize("failed_pane", ["run", "analysis", "post_analysis"])
def test_data_activation_uses_public_reader_and_isolates_each_failure(
    presentation_window: QWidget, monkeypatch: pytest.MonkeyPatch, failed_pane: str
) -> None:
    (tab,) = presentation_window.findChildren(ExpTabWidget)
    preview = gallery(tab)
    figures = {pane: Figure() for pane in ("run", "analysis", "post_analysis")}
    reads: list[str] = []
    events: list[tuple[str, bool]] = []

    def read(pane: str) -> Figure:
        reads.append(pane)
        if pane == failed_pane:
            raise ArithmeticError("overridden reader failed")
        return figures[pane]

    update = preview.update_figures
    reflow = preview.refresh_layout

    def record_update(
        run: Figure | None, analysis: Figure | None, post: Figure | None
    ) -> None:
        events.append(("refresh", preview.isVisible()))
        update(run, analysis, post)

    def record_reflow() -> None:
        events.append(("reflow", preview.isVisible()))
        reflow()

    monkeypatch.setattr(tab, "get_current_figure_for_pane", read)
    monkeypatch.setattr(preview, "update_figures", record_update)
    monkeypatch.setattr(preview, "refresh_layout", record_reflow)
    tab.select_pane("data")

    assert reads == ["run", "analysis", "post_analysis"]
    assert events[0] == ("refresh", False)
    assert events[-1] == ("reflow", True)
    for pane in figures:
        assert preview.card_state(pane) == (
            "empty" if pane == failed_pane else "available"
        )


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_named_collection_keeps_figures_saveable_after_clear(
    presentation_window: QWidget, pane: str
) -> None:
    (tab,) = presentation_window.findChildren(ExpTabWidget)
    collection = FigureCollection()
    first = Figure()
    second = Figure()
    collection.adopt("first", first)
    collection.adopt("second", second)
    for index in range(3):
        collection.adopt(f"extra-{index}", Figure())
    collection.seal()
    show = (
        tab.show_analysis_figures
        if pane == "analysis"
        else tab.show_post_analysis_figures
    )
    show(collection)
    assert tab.get_current_figure_for_pane(pane) is first
    tab.clear_all_figures()
    assert tab.get_current_figure_for_pane(pane) is None
    for figure in (first, second):
        output = BytesIO()
        figure.savefig(output, format="png")
        assert output.getvalue().startswith(b"\x89PNG")
