"""Workflow figure paging through visible controls and public presentation seams."""

from pathlib import Path

import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget
from zcu_tools.experiment.workflows import WorkflowPlots
from zcu_tools.gui.app.measure.ui.workflow_figures import WorkflowFigures
from zcu_tools.gui.plotting import get_figure_container
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.plotting.plots import Plots

from tests.gui._user_input import ComponentDriver


@pytest.fixture
def workflow_figures(qapp):
    window = QWidget()
    owner = QtOwnerScheduler()
    view = WorkflowFigures(owner, lambda: plots, window)
    plots = WorkflowPlots(view)
    QVBoxLayout(window).addWidget(view)
    window.resize(1100, 760)
    window.show()
    try:
        yield window, view, plots
    finally:
        plots.finish()
        plots.release()
        view.dispose()
        window.close()
        window.deleteLater()
        owner.deleteLater()
        qapp.processEvents()


def add_figure(plots: Plots, index: int) -> Figure:
    """Produce one synthetic native figure through the framework plotting API."""
    figure = Figure()
    plots.adopt(f"Tone {index}", figure)

    def draw(figure: Figure) -> None:
        axes = figure.subplots()
        axes.plot([0, 1, 2], [0, index, 0], label=f"Tone {index}")
        axes.set_xlabel("Time")
        axes.legend()

    plots.draw(f"Tone {index}", draw)
    QApplication.processEvents()
    return figure


def test_follow_latest_selects_new_figure_across_pages(workflow_figures, tmp_path):
    window, view, plots = workflow_figures
    flow = ComponentDriver(window)
    for index in range(1, 6):
        figure = Figure()
        plots.adopt(f"Tone {index}", figure)
        view.present(figure)
        QApplication.processEvents()
        flow.screenshot(tmp_path / f"follow-new-{index}.png")
        assert view.get_current_figure() is figure
    assert flow.observe("Follow latest").checked
    assert flow.observe("2 / 2").visible


def test_follow_latest_tracks_live_refresh_but_not_final_flush(
    workflow_figures, tmp_path
):
    window, view, plots = workflow_figures
    flow = ComponentDriver(window)
    figures = [add_figure(plots, index) for index in range(1, 6)]
    flow.click("Previous page")
    flow.screenshot(tmp_path / "live-before-refresh.png")
    plots.refresh("Tone 5")
    QApplication.processEvents()
    flow.screenshot(tmp_path / "live-second-page.png")
    assert flow.observe("2 / 2").visible
    assert view.get_current_figure() is figures[4]
    plots.refresh("Tone 2")
    QApplication.processEvents()
    flow.screenshot(tmp_path / "live-first-page.png")
    assert flow.observe("1 / 2").visible
    assert view.get_current_figure() is figures[1]
    assert flow.observe("Follow latest").checked
    plots.finish()
    QApplication.processEvents()
    flow.screenshot(tmp_path / "final-keeps-focus.png")
    assert flow.observe("1 / 2").visible
    assert view.get_current_figure() is figures[1]


@pytest.mark.parametrize("tile", [1, 4])
def test_canvas_click_pins_focus_until_follow_is_enabled(
    workflow_figures, tmp_path, tile
):
    window, view, plots = workflow_figures
    flow = ComponentDriver(window)
    figures = [add_figure(plots, index) for index in range(1, 6)]
    if tile < 4:
        flow.click("Previous page")
    flow.screenshot(tmp_path / "follow-checked.png")
    flow.click(object_name=f"figures.tile.workflow.{tile}")
    flow.screenshot(tmp_path / "follow-unchecked.png")
    assert not flow.observe("Follow latest").checked
    assert view.get_current_figure() is figures[tile]
    add_figure(plots, 6)
    plots.refresh("Tone 3")
    QApplication.processEvents()
    flow.screenshot(tmp_path / "pinned-after-new-and-refresh.png")
    assert view.get_current_figure() is figures[tile]
    flow.click("Follow latest")
    plots.refresh("Tone 5")
    QApplication.processEvents()
    flow.screenshot(tmp_path / "following-again.png")
    assert flow.observe("Follow latest").checked
    assert flow.observe("2 / 2").visible
    assert view.get_current_figure() is figures[4]


def test_workflow_tiles_pages_selection_identity_and_disposal(
    workflow_figures, tmp_path: Path
):
    window, view, plots = workflow_figures
    flow = ComponentDriver(window)
    flow.click("Follow latest")
    figures = [add_figure(plots, index) for index in range(1, 4)]
    assert flow.legends().texts == ("Tone 1", "Tone 2", "Tone 3")
    assert not any(
        widget.object_name == "figures.pager.workflow"
        for widget in flow.visible().widgets
    )
    flow.screenshot(tmp_path / "three.png")

    operation = plots.start_operation()
    figures.extend(add_figure(operation, index) for index in (4, 5))
    operation.finish()
    plots.finish()
    QApplication.processEvents()
    assert flow.legends().texts == ("Tone 1", "Tone 2", "Tone 3", "Tone 4")
    assert flow.observe("1 / 2").visible
    assert not flow.observe("Previous page").enabled
    flow.screenshot(tmp_path / "five-page-1.png")
    flow.click(object_name="figures.tile.workflow.2")
    assert view.get_current_figure() is figures[2]
    flow.screenshot(tmp_path / "selected-tile.png")
    flow.click("Next page")
    assert flow.legends().texts == ("Tone 5",)
    assert flow.observe("2 / 2").visible
    assert not flow.observe("Next page").enabled
    assert view.get_current_figure() is figures[4]
    flow.screenshot(tmp_path / "five-page-2.png")

    canvas = figures[4].canvas
    view.present(figures[4])
    QApplication.processEvents()
    assert figures[4].canvas is canvas
    assert flow.observe("2 / 2").visible
    assert view.get_current_figure() is figures[4]
    flow.screenshot(tmp_path / "retained.png")
    flow.click("Previous page")
    assert view.get_current_figure() is figures[0]
    assert flow.observe("1 / 2").visible
    flow.screenshot(tmp_path / "returned.png")
    view.dispose()
    QApplication.processEvents()
    assert view.get_current_figure() is None
    assert flow.plots().axes == ()
    assert not flow.observe(object_name="figures.pager.workflow").visible
    assert all(get_figure_container(figure) is None for figure in figures)
    assert all(figure.axes for figure in figures)
    flow.screenshot(tmp_path / "disposed.png")
