from collections.abc import Callable, Iterator
from io import BytesIO
from typing import cast

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ipympl.backend_nbagg import Canvas, FigureManager, Toolbar
from ipywidgets import Widget
from matplotlib.figure import Figure
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.twotone.time_domain.t1 import (
    T1AnalyzeOptions,
    T1Cfg,
    T1Exp,
    T1Result,
)
from zcu_tools.notebook import NotebookAdapter
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


def test_initial_frame_is_ready_before_widget_publication(
    notebook_plots: tuple[Callable[[], Plots], list[Canvas]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames: list[bytes] = []
    frames_at_publication: list[tuple[bytes, ...]] = []
    send_binary = Canvas.send_binary
    create, published = notebook_plots

    def record_frame(canvas: Canvas, data: bytes) -> None:
        send_binary(canvas, data)
        frames.append(data)

    def publish(canvas: Canvas) -> None:
        frames_at_publication.append(tuple(frames))
        published.append(canvas)

    monkeypatch.setattr(Canvas, "send_binary", record_frame)
    monkeypatch.setattr(IPython.display, "display", publish)
    plots = create()
    figure, axes = plots.subplots("fit")
    axes.plot([0, 1, 2], [0, 1, 0])
    plots.finish()

    assert published == [figure.canvas]
    assert frames_at_publication[0]
    assert frames_at_publication[0][-1].startswith(b"\x89PNG")


def test_live_and_ordinary_figures_present_once_without_pyplot_registration(
    notebook_plots: tuple[Callable[[], Plots], list[Canvas]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[tuple[Canvas, list[float]]] = []
    send_binary = Canvas.send_binary

    def record_frame(canvas: Canvas, data: bytes) -> None:
        send_binary(canvas, data)
        rendered.append(
            (
                canvas,
                np.asarray(
                    canvas.figure.axes[0].lines[0].get_ydata(), dtype=float
                ).tolist(),
            )
        )

    monkeypatch.setattr(Canvas, "send_binary", record_frame)
    create, published = notebook_plots
    before = plt.get_fignums()
    backend = plt.get_backend()
    plots = create()
    ordinary, axes = plots.subplots("ordinary")
    axes.plot([1, 2], [3, 4])
    assert published == []

    live = plots.liveplot_1d("live", "x", "y")
    assert published == [plots["live"].canvas]
    rendered.clear()
    live.update(np.array([1.0, 2.0]), np.array([5.0, 6.0]))
    assert rendered
    assert rendered[-1] == (plots["live"].canvas, [5.0, 6.0])
    plots.finish()
    plots.finish()

    assert published == [plots["live"].canvas, ordinary.canvas]
    np.testing.assert_array_equal(plots["live"].axes[0].lines[0].get_ydata(), [5, 6])
    live_frames = [data for canvas, data in rendered if canvas is plots["live"].canvas]
    assert live_frames[-1] == [5.0, 6.0]
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


@pytest.mark.parametrize("stage", ["before", "after"])
def test_toolbar_release_error_still_closes_canvas_and_keeps_figure_savable(
    notebook_plots: tuple[Callable[[], Plots], list[Canvas]],
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    create, published = notebook_plots
    plots = create()
    figure, _axes = plots.subplots("fit")
    plots.finish()
    canvas = published[0]
    toolbar = canvas.toolbar
    assert isinstance(toolbar, Toolbar)
    original_close = Toolbar.close

    def fail_toolbar_close(self: Toolbar) -> None:
        if stage == "before":
            raise RuntimeError("toolbar close failed")
        original_close(self)
        raise RuntimeError("toolbar close failed")

    with monkeypatch.context() as patch:
        patch.setattr(Toolbar, "close", fail_toolbar_close)
        with pytest.raises(ExceptionGroup, match="Failed to release plot") as exc:
            plots.release()
    assert len(exc.value.exceptions) == 1
    assert str(exc.value.exceptions[0]) == "toolbar close failed"
    assert canvas.comm is None
    assert canvas.layout.comm is None
    assert toolbar.comm is None
    assert toolbar.layout.comm is None
    assert figure.canvas is not canvas
    assert figure.canvas.manager is None
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_initialization_failure_releases_widgets_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[Canvas] = []
    failed: list[Canvas] = []
    monkeypatch.setattr(IPython.display, "display", published.append)
    host = NotebookPlotHost()
    figure = Figure()
    figure.subplots().plot([0, 1], [0, 1])
    error = RuntimeError("initial draw failed")

    def fail_draw(canvas: Canvas) -> None:
        failed.append(canvas)
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(Canvas, "draw", fail_draw)
        with pytest.raises(RuntimeError, match="initial draw failed") as exc:
            host.present(figure)
    assert exc.value is error
    canvas = failed[0]
    assert published == []
    assert canvas.comm is None
    assert canvas.layout.comm is None
    assert figure.canvas is not canvas
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")

    try:
        host.present(figure)
        assert published == [figure.canvas]
        assert published[0].comm is not None
    finally:
        host.release(figure)


def test_notebook_adapter_uses_default_widget_host_and_retains_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    displayed: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    times = np.linspace(0.0, 80.0, 81)
    source = RunRecord[T1Cfg, T1Result](
        cfg=None,
        result=T1Result(times, np.exp(-times / 20.0).astype(np.complex128)),
    )
    adapter = NotebookAdapter(T1Exp())
    try:
        successful = adapter.analyze(T1AnalyzeOptions(), source=source)
        figure = successful.figures["fit"]
        assert displayed == [figure.canvas]
        assert figure.canvas.manager is not None
        live_widgets = widget_ids()

        def fail_display(widget: object) -> None:
            raise RuntimeError("publisher failed")

        monkeypatch.setattr(IPython.display, "display", fail_display)
        with pytest.raises(RuntimeError, match="publisher failed"):
            adapter.analyze(T1AnalyzeOptions(skip=1), source=source)
        assert adapter.analysis is successful
        assert widget_ids() == live_widgets
    finally:
        if adapter.analysis_presentation is not None:
            adapter.analysis_presentation.release()


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


@pytest.mark.parametrize("stage", ["canvas", "toolbar"])
def test_manager_failure_releases_acquired_widgets_and_restores_savefig(
    stage: str,
    notebook_plots: tuple[Callable[[], Plots], list[Canvas]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create, published = notebook_plots
    other = create()
    other.subplots("retained")
    other.finish()
    retained = published[0]
    acquired: list[Canvas] = []
    error = RuntimeError("manager construction failed")

    def fail_manager(canvas: Canvas, num: int) -> FigureManager:
        acquired.append(canvas)
        if stage == "toolbar":
            FigureManager(canvas, num)
        raise error

    monkeypatch.setattr("ipympl.backend_nbagg.FigureManager", fail_manager)
    plots = create()
    with pytest.raises(RuntimeError, match="manager construction failed") as exc:
        plots.liveplot_1d("live", "x", "y")
    assert exc.value is error
    canvas = acquired[0]
    assert canvas.comm is None
    assert canvas.layout.comm is None
    if stage == "toolbar":
        assert isinstance(canvas.toolbar, Toolbar)
        assert canvas.toolbar.comm is None
        assert canvas.toolbar.layout.comm is None
    assert plots["live"].canvas is not canvas
    assert plots["live"].canvas.manager is None
    plots.finish(present=False)
    plots.release()
    output = BytesIO()
    plots["live"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")
    assert retained.comm is not None
    assert other["retained"].canvas is retained


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
