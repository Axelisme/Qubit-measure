"""Named pane paging through visible controls and the public Figure reader."""

from pathlib import Path
from typing import Literal

import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.ui.experiment_figure_area import ExperimentFigureArea
from zcu_tools.plotting.figures import FigureCollection

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure.ui._preview_gallery_support import fake_renderer


def figures(count: int) -> FigureCollection:
    """Build synthetic named, saveable Figures without a lab dependency."""
    collection = FigureCollection()
    for index in range(1, count + 1):
        figure = Figure()
        axes = figure.subplots()
        axes.plot([0, 1, 2], [0, index, 0], label=f"Tone {index}")
        axes.set_xlabel("Time")
        axes.legend()
        collection.adopt(f"Tone {index}", figure)
    collection.seal()
    return collection


@pytest.fixture
def area(qapp):
    widget = ExperimentFigureArea(
        "paging",
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True),
        lambda pane: widget.get_current_figure_for_pane(pane),
        preview_renderer=fake_renderer,
    )
    widget.resize(1100, 760)
    widget.show()
    yield widget
    widget.clear_all_figures()
    widget.close()
    widget.deleteLater()


def present(
    area: ExperimentFigureArea,
    pane: Literal["run", "analysis", "post_analysis"],
    collection: FigureCollection,
) -> None:
    """Bootstrap named canvases through the public attachment and presentation seams."""
    if pane == "run":
        if not all(
            area.get_run_container().contains_figure(figure)
            for figure in collection.values()
        ):
            area.prepare_run_container()
        for figure in collection.values():
            if not area.get_run_container().contains_figure(figure):
                area.show_run_figure(figure)
        area.present_figures(pane, collection)
    elif pane == "analysis":
        area.show_analysis_figures(collection)
    else:
        area.show_post_analysis_figures(collection)
    area.select_pane(pane)
    QApplication.processEvents()


@pytest.mark.parametrize("pane", ["run", "analysis", "post_analysis"])
@pytest.mark.parametrize("count", [5, 6])
def test_paging_selection_identity_and_clear(
    area, pane: Literal["run", "analysis", "post_analysis"], count: int, tmp_path: Path
):
    flow = ComponentDriver(area)
    pager = f"figures.pager.{pane}"
    present(area, pane, figures(3))
    assert flow.legends().texts == ("Tone 1", "Tone 2", "Tone 3")
    assert not any(w.object_name == pager for w in flow.visible().widgets)
    flow.screenshot(tmp_path / "three.png")

    collection = figures(count)
    present(area, pane, collection)
    assert flow.legends().texts == tuple(f"Tone {i}" for i in range(1, 5))
    assert flow.observe("1 / 2").visible
    assert not flow.observe("Previous page").enabled
    assert area.get_current_figure_for_pane(pane) is collection["Tone 1"]
    flow.screenshot(tmp_path / f"{count}-page-1.png")
    flow.click("Next page")
    assert flow.legends().texts == tuple(f"Tone {i}" for i in range(5, count + 1))
    assert flow.observe("2 / 2").visible
    assert not flow.observe("Next page").enabled
    assert area.get_current_figure_for_pane(pane) is collection["Tone 5"]
    flow.screenshot(tmp_path / f"{count}-page-2.png")
    # Canvas object names are public interaction targets, not private state.
    flow.click(object_name=f"figures.tile.{pane}.{count - 1}")
    assert area.get_current_figure_for_pane(pane) is collection[f"Tone {count}"]
    present(area, pane, collection)
    assert flow.observe("2 / 2").visible
    assert area.get_current_figure_for_pane(pane) is collection[f"Tone {count}"]
    flow.screenshot(tmp_path / "retained.png")
    flow.click("Previous page")
    assert area.get_current_figure_for_pane(pane) is collection["Tone 1"]
    assert flow.observe("1 / 2").visible
    flow.screenshot(tmp_path / "returned.png")
    area.clear_all_figures()
    assert not flow.observe(object_name=pager).visible
    assert area.get_current_figure_for_pane(pane) is None
    flow.screenshot(tmp_path / "cleared.png")


@pytest.mark.parametrize("pane", ["run", "analysis", "post_analysis"])
def test_tiled_heatmap_ylabels_fit_after_resize_and_paging(
    area, pane: Literal["run", "analysis", "post_analysis"], tmp_path: Path
):
    collection = FigureCollection()
    for index in range(1, 6):
        figure = Figure()
        heatmap, line = figure.subplots(ncols=2)
        heatmap.pcolormesh([0, 1, 2], [7000, 7001, 7002], [[1, 2], [3, 4]])
        heatmap.set_xlabel("DC value (A)")
        heatmap.set_ylabel("Frequency (MHz)")
        line.plot([7000, 7001, 7002], [0, index, 0], label=f"Tone {index}")
        line.set_xlabel("Frequency (MHz)")
        line.legend()
        collection.adopt(f"Tone {index}", figure)
    collection.seal()
    flow = ComponentDriver(area)
    # The same native figure remains untouched in ordinary single-canvas mode.
    area.prepare_run_container()
    area.show_run_figure(collection["Tone 1"])
    original_positions = tuple(
        ax.get_position().bounds for ax in collection["Tone 1"].axes
    )
    flow.screenshot(tmp_path / "single.png")
    assert all(
        b.canvas_left <= b.ylabel_left <= b.axes_left for b in flow.plot_label_bounds()
    )
    assert (
        tuple(ax.get_position().bounds for ax in collection["Tone 1"].axes)
        == original_positions
    )

    present(area, pane, collection)
    flow.screenshot(tmp_path / "tiled.png")
    bounds = flow.plot_label_bounds()
    assert len(bounds) == 4
    assert all(b.canvas_left <= b.ylabel_left <= b.axes_left for b in bounds), bounds
    assert all(b.ylabel_right <= b.canvas_right for b in bounds), bounds
    flow.click("Next page")
    flow.screenshot(tmp_path / "last-page.png")
    bounds = flow.plot_label_bounds()
    assert len(bounds) == 1
    assert bounds[0].canvas_name == f"figures.tile.{pane}.4"
    assert bounds[0].canvas_left <= bounds[0].ylabel_left <= bounds[0].axes_left
    flow.click("Previous page")
    flow.screenshot(tmp_path / "returned-labels.png")
    assert all(
        b.canvas_left <= b.ylabel_left <= b.axes_left for b in flow.plot_label_bounds()
    )


def test_data_named_previews_page_and_reset(area, tmp_path: Path):
    flow = ComponentDriver(area)
    present(area, "run", figures(3))
    area.select_pane("data")
    QApplication.processEvents()
    assert not any(
        w.object_name == "figures.pager.data.run" for w in flow.visible().widgets
    )
    assert {"Tone 1", "Tone 2", "Tone 3"} <= {w.text for w in flow.visible().widgets}
    flow.screenshot(tmp_path / "data-three.png")
    collection = figures(5)
    present(area, "run", collection)
    area.select_pane("data")
    QApplication.processEvents()
    assert {f"Tone {i}" for i in range(1, 5)} <= {
        w.text for w in flow.visible().widgets
    }
    assert flow.observe("1 / 2").visible
    flow.screenshot(tmp_path / "data-page-1.png")
    flow.click("Next page")
    assert flow.observe("2 / 2").visible
    assert flow.observe("Tone 5").visible
    flow.screenshot(tmp_path / "data-page-2.png")
    area.select_pane("run")
    area.select_pane("data")
    QApplication.processEvents()
    assert flow.observe("2 / 2").visible
    area.clear_all_figures()
    assert not flow.observe(object_name="figures.pager.data.run").visible
    flow.screenshot(tmp_path / "data-cleared.png")
