from collections.abc import Callable, Iterator
from io import BytesIO
from typing import cast

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ipympl.backend_nbagg import Canvas, Toolbar
from ipywidgets import Widget
from matplotlib.figure import Figure
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.plotting.plots import Plots


def widget_ids() -> set[str]:
    # Snapshot membership without serializing unrelated widgets' transient state.
    return set(cast("dict[str, Widget]", Widget.widgets))


@pytest.fixture(scope="module", autouse=True)
def module_widget_registry_guard() -> Iterator[None]:
    before = widget_ids()
    yield
    assert widget_ids() == before


@pytest.fixture(autouse=True)
def widget_registry_guard() -> Iterator[None]:
    before = widget_ids()
    yield
    assert widget_ids() == before


@pytest.fixture
def notebook_plots(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Callable[[], Plots], list[Canvas]]]:
    published: list[Canvas] = []
    operations: list[Plots] = []
    monkeypatch.setattr(IPython.display, "display", published.append)

    def create() -> Plots:
        plots = Plots(NotebookPlotHost())
        operations.append(plots)
        return plots

    yield create, published
    for plots in operations:
        plots.finish(present=False)
        plots.release()


def test_live_and_ordinary_figures_present_once_without_pyplot_registration(
    notebook_plots: tuple[Callable[[], Plots], list[Canvas]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[list[float]] = []
    draw = Canvas.draw

    def record_draw(canvas: Canvas) -> None:
        draw(canvas)
        rendered.append(
            np.asarray(canvas.figure.axes[0].lines[0].get_ydata(), dtype=float).tolist()
        )

    monkeypatch.setattr(Canvas, "draw", record_draw)
    create, published = notebook_plots
    before = plt.get_fignums()
    backend = plt.get_backend()
    plots = create()
    ordinary, axes = plots.subplots("ordinary")
    axes.plot([1, 2], [3, 4])
    assert published == []

    live = plots.liveplot_1d("live", "x", "y")
    assert published == [plots["live"].canvas]
    live.update(np.array([1.0, 2.0]), np.array([5.0, 6.0]))
    plots.finish()
    plots.finish()

    assert published == [plots["live"].canvas, ordinary.canvas]
    np.testing.assert_array_equal(plots["live"].axes[0].lines[0].get_ydata(), [5, 6])
    assert rendered[-1] == [5.0, 6.0]
    assert plt.get_fignums() == before
    assert plt.get_backend() == backend


def test_release_closes_widgets_and_preserves_other_operation_and_savefig(
    notebook_plots: tuple[Callable[[], Plots], list[Canvas]],
) -> None:
    create, published = notebook_plots
    first, second = create(), create()
    first.subplots("fit")
    first.finish()
    second.subplots("fit")
    second.finish()
    old_canvas, current_canvas = published
    old_toolbar = old_canvas.toolbar
    current_toolbar = current_canvas.toolbar
    assert isinstance(old_toolbar, Toolbar)
    assert isinstance(current_toolbar, Toolbar)
    assert old_canvas.comm is not None
    assert old_toolbar.comm is not None

    first.release()
    first.release()
    assert old_canvas.comm is None
    assert old_toolbar.comm is None
    assert current_canvas.comm is not None
    assert current_toolbar.comm is not None
    output = BytesIO()
    first["fit"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_failed_display_propagates_and_diagnostic_release_closes_widgets(
    notebook_plots: tuple[Callable[[], Plots], list[Canvas]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create, published = notebook_plots

    def fail_display(canvas: Canvas) -> None:
        published.append(canvas)
        raise RuntimeError("publisher unavailable")

    monkeypatch.setattr(IPython.display, "display", fail_display)
    plots = create()
    with pytest.raises(RuntimeError, match="publisher unavailable"):
        plots.liveplot_1d("live", "x", "y")
    canvas = published[0]
    toolbar = canvas.toolbar
    assert isinstance(toolbar, Toolbar)
    plots.finish(present=False)
    plots.release()
    assert canvas.comm is None
    assert toolbar.comm is None
    output = BytesIO()
    plots["live"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_host_rejects_other_manager_without_moving_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[Canvas] = []
    monkeypatch.setattr(IPython.display, "display", published.append)
    first, second = NotebookPlotHost(), NotebookPlotHost()
    figure = Figure()
    first.present(figure)
    try:
        canvas = figure.canvas
        first.present(figure)
        with pytest.raises(ValueError, match="already has a presentation manager"):
            second.present(figure)
        second.release(figure)
        with pytest.raises(ValueError, match="not presented by this host"):
            second.refresh(figure)
        assert figure.canvas is canvas
        assert published == [canvas]
        assert published[0].comm is not None
    finally:
        first.release(figure)
