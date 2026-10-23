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


def test_workflow_tiles_pages_selection_identity_and_disposal(
    workflow_figures, tmp_path: Path
):
    window, view, plots = workflow_figures
    flow = ComponentDriver(window)
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
