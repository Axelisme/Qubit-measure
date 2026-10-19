"""User-visible experiment pane switching and named figure presentation contracts."""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QComboBox, QWidget
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    PostAnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.persistence_types import (
    AppPersistedState,
    PersistedSession,
    PersistedTab,
)
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
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


class PresentationAdapter(AnalysisSavingAdapter):
    """Offline adapter with two independently selectable figures in each result pane."""

    def run(
        self, req: RunRequest, cfg: SavingCfg, *, context: RunContext
    ) -> RunRecord[SavingCfg, SavingResult]:
        record = super().run(req, cfg, context=context)
        context.plots.subplots("signal")

        def draw(figure: Figure) -> None:
            axes = figure.axes[0]
            axes.plot([0, 1, 2], [0, 1, 0], label="Signal")
            axes.set_xlabel("Time")
            axes.legend()

        context.plots.draw("signal", draw)
        return record

    def analyze(
        self,
        req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams],
        *,
        plots: Plots,
    ) -> PaneSummary:
        for name, values in (("fit", [0, 1, 0]), ("residual", [0, -1, 0])):
            _, axes = plots.subplots(name)
            axes.plot([0, 1, 2], values, label=name)
            axes.set_xlabel("Time")
            axes.legend()
        return PaneSummary()

    def post_analyze(
        self,
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
        app.controller.restore_persisted_state(
            AppPersistedState(
                session=PersistedSession(
                    tabs=(PersistedTab(adapter_name="presentation", form_input={}),),
                    active_tab_index=0,
                )
            )
        )
        app.window.resize(1280, 800)
        yield app.window


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
        lambda: flow.legends().texts == ("fit",) and flow.observe("Analyze").enabled,
        label="offline analysis presentation",
    )
    flow.screenshot(shots / "03-analysis-fit.png")
    flow.choose_option("fit", "residual")
    assert flow.legends().texts == ("residual",)
    flow.screenshot(shots / "04-analysis-residual.png")
    flow.select_tab("Post-Analysis")
    flow.click("Run Post-Analysis")
    qt_until(
        lambda: (
            flow.legends().texts == ("summary",)
            and flow.observe("Run Post-Analysis").enabled
        ),
        label="offline post-analysis presentation",
    )
    flow.screenshot(shots / "05-post-summary.png")
    flow.choose_option("summary", "detail")
    assert flow.legends().texts == ("detail",)
    flow.screenshot(shots / "06-post-detail.png")
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
    assert flow.legends().texts == ("residual",)
    flow.screenshot(shots / "11-analysis-retained.png")
    flow.select_tab("Post-Analysis")
    assert flow.legends().texts == ("detail",)
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


@pytest.mark.parametrize(
    "pane,label",
    [("analysis", "Analysis figure"), ("post_analysis", "Post-analysis figure")],
)
def test_named_collection_identity_blocks_selector_repopulation_and_keeps_figures_saveable(
    presentation_window: QWidget, pane: str, label: str
) -> None:
    (tab,) = presentation_window.findChildren(ExpTabWidget)
    selector = next(
        w for w in tab.findChildren(QComboBox) if w.accessibleName() == label
    )
    collection = FigureCollection()
    first = Figure()
    second = Figure()
    collection.adopt("first", first)
    collection.adopt("second", second)
    collection.seal()
    changes: list[int] = []
    selector.currentIndexChanged.connect(changes.append)
    show = (
        tab.show_analysis_figures
        if pane == "analysis"
        else tab.show_post_analysis_figures
    )
    show(collection)
    assert changes == []
    selector.setCurrentIndex(1)
    assert changes == [1]
    assert tab.get_current_figure_for_pane(pane) is second
    show(collection)
    assert changes == [1]
    assert selector.currentText() == "second"
    assert tab.get_current_figure_for_pane(pane) is second
    tab.clear_all_figures()
    assert changes == [1]
    assert tab.get_current_figure_for_pane(pane) is None
    for figure in (first, second):
        output = BytesIO()
        figure.savefig(output, format="png")
        assert output.getvalue().startswith(b"\x89PNG")
