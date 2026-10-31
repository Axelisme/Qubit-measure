from __future__ import annotations

import gc
import os
import threading
from collections.abc import Callable, Iterator
from io import BytesIO
from pathlib import Path
from weakref import ref

import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from qtpy.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.plotting import FigureContainer, get_figure_container
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.ports import BackgroundError
from zcu_tools.plotting.plots import Plots

from tests._plotting import create_trace
from tests.gui._user_input import ComponentDriver


@pytest.fixture
def hosts(qapp):
    owner = QtOwnerScheduler()
    stacks = [QStackedWidget(), QStackedWidget()]
    containers = []
    for stack in stacks:
        placeholder = QLabel("empty")
        stack.addWidget(placeholder)
        containers.append(FigureContainer(stack, placeholder))
    try:
        yield stacks, containers, [QtPlotHost(c, owner) for c in containers]
    finally:
        for container, stack in zip(containers, stacks, strict=True):
            container.clear_dynamic_canvases()
            stack.deleteLater()
        owner.deleteLater()
        qapp.processEvents()


def _run_worker(qapp: QApplication, work: Callable[[], int]) -> int:
    background = BackgroundRunner()
    loop, timer = QEventLoop(), QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    outcomes: list[int | BackgroundError] = []

    def completed(value: int | BackgroundError) -> None:
        outcomes.append(value)
        loop.quit()

    background.submit(work, on_done=completed, on_error=completed, run_in_pool=False)
    try:
        timer.start(5000)
        loop.exec()
        assert len(outcomes) == 1
        result = outcomes[0]
        if not isinstance(result, int):
            raise result
        return result
    finally:
        timer.stop()
        assert background.quiesce()
        background.deleteLater()
        qapp.processEvents()


def test_figure_show_survives_qt_canvas_swap(qapp, hosts) -> None:
    stacks, containers, adapters = hosts
    plots = Plots(adapters[0])
    figure, axes = plots.subplots(num="explicit")
    axes.plot([0, 1], [2, 3])
    assert stacks[0].count() == 1
    figure.show()
    canvas = figure.canvas
    assert get_figure_container(figure) is containers[0]
    assert stacks[0].count() == 2
    figure.show()
    assert figure.canvas is canvas
    assert stacks[0].count() == 2
    plots.finish()
    with pytest.raises(RuntimeError, match="finished"):
        figure.show()
    plots.release()
    with pytest.raises(RuntimeError, match="operation.*finished"):
        figure.show()
    adapters[1].present(figure)
    assert get_figure_container(figure) is containers[1]
    with pytest.raises(RuntimeError, match="operation.*finished"):
        figure.show()
    adapters[1].release(figure)


def _native_callbacks(
    plots: Plots, axes: Axes
) -> tuple[Callable[[str], None], Callable[[NDArray[np.float64]], None]]:
    owner_id = threading.get_ident()
    (curve,) = axes.plot([], [], "o-")
    annotation = axes.text(0.5, 0.9, "", transform=axes.transAxes, ha="center")

    @plots.on_plot_thread
    def set_caption(text: str) -> tuple[Artist, ...]:
        assert threading.get_ident() == owner_id
        axes.set_title(text)
        return (axes.title,)

    @plots.on_plot_thread
    def push(values: NDArray[np.float64]) -> tuple[Artist, ...]:
        assert threading.get_ident() == owner_id
        curve.set_data([0.0, 1.0, 2.0], values)
        axes.relim()
        axes.autoscale_view()
        annotation.set_text(f"Worker peak {values[1]:g} mV")
        return (curve, annotation, axes)

    return set_caption, push


def _finish_actions(
    plots: Plots, figure: Figure, container: FigureContainer, status: QLabel
) -> tuple[Callable[[], None], Callable[[], None], Callable[[], None]]:
    def finish_and_clear() -> None:
        plots.finish()
        container.clear_dynamic_canvases()
        status.setText("Finished and cleared")

    def try_finished_show() -> None:
        try:
            figure.show()
        except RuntimeError as error:
            status.setText(str(error))

    def finish() -> None:
        plots.finish()
        plots.release()
        status.setText("Released")

    return finish_and_clear, try_finished_show, finish


@pytest.fixture
def explicit_show_driver(qapp) -> Iterator[ComponentDriver]:
    window = QWidget()
    layout = QVBoxLayout(window)
    status = QLabel("Ready")
    layout.addWidget(status)
    stack = QStackedWidget()
    placeholder = QLabel("No figure shown")
    stack.addWidget(placeholder)
    layout.addWidget(stack, stretch=1)
    container = FigureContainer(stack, placeholder)
    owner = QtOwnerScheduler()
    host = QtPlotHost(container, owner)
    plots = Plots(host)
    figure, axes = plots.subplots(
        num="explicit",
        subplot_kw={"xlabel": "Frequency (MHz)", "ylabel": "Signal (mV)"},
    )
    scope = LifetimeScope(window)
    set_caption, push = _native_callbacks(plots, axes)

    def show() -> None:
        set_caption("Native decorated update")
        figure.show()
        status.setText("Shown")

    def update(peak: float) -> None:
        def work() -> int:
            push(np.array([1.0, peak, 2.0]))
            return threading.get_ident()

        _run_worker(qapp, work)
        if get_figure_container(figure) is not None:
            host.refresh(figure, final=True)
        status.setText("Worker complete")

    def legacy() -> None:
        create_trace(plots, "legacy", "Time (us)", "Legacy signal").update(
            (np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        )
        host.refresh(plots["legacy"], final=True)
        status.setText("Legacy shown")

    finish_and_clear, try_finished_show, finish = _finish_actions(
        plots, figure, container, status
    )
    for caption, action in (
        ("Show figure", show),
        ("Prime before show", lambda: update(3.0)),
        ("Update from worker", lambda: update(5.0)),
        ("Legacy live", legacy),
        ("Finish and clear", finish_and_clear),
        ("Try finished show", try_finished_show),
        ("Finish and release", finish),
    ):
        button = QPushButton(caption)
        layout.addWidget(button)
        scope.connect(button.clicked, action)
    window.show()
    try:
        yield ComponentDriver(window)
    finally:
        plots.finish(present=False)
        plots.release()
        scope.close()
        container.clear_dynamic_canvases()
        window.close()
        window.deleteLater()
        owner.deleteLater()
        qapp.processEvents()


def test_component_driver_explicit_show_worker_update_and_legacy_live(
    explicit_show_driver: ComponentDriver, tmp_path: Path
) -> None:
    driver = explicit_show_driver
    driver.resize_window(800, 650)
    evidence = Path(os.environ.get("PLOT_SHOW_EVIDENCE", str(tmp_path)))
    assert driver.plots().axes == ()
    driver.screenshot(evidence / "01-unshown.png")
    driver.click("Prime before show")
    assert driver.observe("Worker complete").visible
    assert driver.plots().axes == ()
    driver.screenshot(evidence / "01b-updated-before-show.png")
    driver.click("Show figure")
    assert driver.observe("Shown").visible
    assert driver.plots().axes[0].xlabel == "Frequency (MHz)"
    assert driver.plots().axes[0].visible_curves == 1
    assert "Worker peak 3 mV" in driver.plots().axes[0].annotations
    driver.screenshot(evidence / "02-shown.png")
    driver.click("Update from worker")
    assert driver.observe("Worker complete").visible
    assert driver.plots().axes[0].visible_curves == 1
    assert "Worker peak 5 mV" in driver.plots().axes[0].annotations
    driver.screenshot(evidence / "04-worker-update.png")
    driver.click("Legacy live")
    assert driver.observe("Legacy shown").visible
    assert driver.plots().axes[0].ylabel == "Legacy signal"
    assert driver.plots().axes[0].visible_curves == 1
    driver.screenshot(evidence / "05-legacy-live.png")
    driver.click("Finish and clear")
    assert driver.observe("Finished and cleared").visible
    assert driver.plots().axes == ()
    driver.screenshot(evidence / "06-cleared.png")
    driver.click("Try finished show")
    assert driver.observe("Plot operation has finished").visible
    assert driver.plots().axes == ()
    driver.screenshot(evidence / "07-finished-show-rejected.png")
    driver.click("Finish and release")
    assert driver.observe("Released").visible
    assert driver.plots().axes == ()
    driver.screenshot(evidence / "08-released.png")
    driver.click("Try finished show")
    assert driver.observe("Plot operation has finished").visible
    assert driver.plots().axes == ()
    driver.screenshot(evidence / "09-released-show-rejected.png")


def test_worker_live_artists_run_on_owner_and_regular_figures_wait(
    qapp, hosts, monkeypatch
) -> None:
    stacks, containers, adapters = hosts
    owner_id = threading.get_ident()
    artist_threads: list[int] = []
    update_threads: list[int] = []
    original_plot = Axes.plot
    original_set_data = Line2D.set_data

    def observed_plot(self, *args, **kwargs):
        artist_threads.append(threading.get_ident())
        return original_plot(self, *args, **kwargs)

    plots = Plots(adapters[0])

    def observed_set_data(self, *args):
        if "measurement" in plots and self in plots["measurement"].axes[0].lines:
            update_threads.append(threading.get_ident())
        return original_set_data(self, *args)

    monkeypatch.setattr(Axes, "plot", observed_plot)
    monkeypatch.setattr(Line2D, "set_data", observed_set_data)

    def work() -> int:
        normal, _ = plots.subplots("fit")
        assert adapters[0].call(lambda: get_figure_container(normal)) is None
        viewer = create_trace(plots, "measurement", "Time", "Signal")
        assert adapters[0].call(lambda: stacks[0].count()) == 2
        viewer.update((np.array([0.0, 1.0]), np.array([3.0, 4.0])), refresh=False)
        plots.finish()
        return threading.get_ident()

    assert _run_worker(qapp, work) != owner_id
    assert artist_threads and set(artist_threads) == {owner_id}
    assert update_threads and set(update_threads) == {owner_id}
    assert stacks[0].count() == 3
    assert get_figure_container(plots["fit"]) is containers[0]
    np.testing.assert_array_equal(
        plots["measurement"].axes[0].lines[0].get_ydata(), [3, 4]
    )
    plots.release()
    assert stacks[0].count() == 1
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    output = BytesIO()
    plots["measurement"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_two_plot_hosts_keep_ownership_and_release_independent(hosts) -> None:
    stacks, containers, adapters = hosts
    first, second = Plots(adapters[0]), Plots(adapters[1])
    create_trace(first, "live", "x", "y")
    create_trace(second, "live", "x", "y")
    figure = first["live"]
    canvas = figure.canvas
    with pytest.raises(ValueError, match="another container"):
        adapters[1].present(figure)
    with pytest.raises(ValueError, match="another container"):
        adapters[1].release(figure)
    assert figure.canvas is canvas
    assert get_figure_container(figure) is containers[0]
    first.finish()
    second.finish()
    first.release()
    assert stacks[0].count() == 1
    assert stacks[1].count() == 2
    assert get_figure_container(second["live"]) is containers[1]
    second.release()
    assert stacks[1].count() == 1


def test_clearing_container_keeps_figures_saveable_and_representable(
    qapp, hosts
) -> None:
    stacks, containers, adapters = hosts
    plots = Plots(adapters[0])
    first, ax = plots.subplots("first")
    ax.plot([0, 1], [2, 3])
    second, ax = plots.subplots("second")
    ax.plot([0, 1], [4, 5])
    plots.finish()
    other = Plots(adapters[1])
    create_trace(other, "live", "x", "y")
    other.finish()
    assert get_figure_container(first) is containers[0]
    assert get_figure_container(second) is containers[0]

    containers[0].clear_dynamic_canvases()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    assert stacks[0].count() == 1
    assert stacks[0].currentIndex() == 0
    assert stacks[1].count() == 2
    assert get_figure_container(other["live"]) is containers[1]
    for figure in (first, second):
        assert get_figure_container(figure) is None
        output = BytesIO()
        figure.savefig(output, format="png")
        assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")

    adapters[0].present(first)
    adapters[0].refresh(first, final=True)
    assert get_figure_container(first) is containers[0]
    assert stacks[0].currentWidget() is first.canvas
    plots.release()
    other.release()
    assert stacks[0].count() == stacks[1].count() == 1


@pytest.mark.parametrize(
    "release", [False, True], ids=["before-release", "after-release"]
)
def test_clearing_finished_container_preserves_show_lifetime(
    hosts, release: bool
) -> None:
    _stacks, containers, adapters = hosts
    plots = Plots(adapters[0])
    figure, axes = plots.subplots(num="finished")
    axes.plot([0, 1], [2, 3])
    plots.finish()
    containers[0].clear_dynamic_canvases()
    try:
        if release:
            plots.release()
        with pytest.raises(RuntimeError, match="operation.*finished"):
            figure.show()
    finally:
        plots.release()
    operation = ref(plots)
    del plots
    gc.collect()
    assert operation() is None
    with pytest.raises(RuntimeError, match="operation.*finished"):
        figure.show()


def test_clearing_container_preserves_caller_replaced_canvas(qapp, hosts) -> None:
    stacks, containers, adapters = hosts
    plots = Plots(adapters[0])
    figure, ax = plots.subplots("replaced")
    ax.plot([0, 1], [2, 3])
    plots.finish()
    replacement_canvas = FigureCanvasAgg(figure)

    containers[0].clear_dynamic_canvases()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    assert figure.canvas is replacement_canvas
    assert get_figure_container(figure) is None
    assert stacks[0].count() == 1
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    plots.release()


def test_worker_draw_presents_on_first_success_and_keeps_other_figures_pending(
    qapp: QApplication,
    hosts: tuple[list[QStackedWidget], list[FigureContainer], list[QtPlotHost]],
    tmp_path: Path,
) -> None:
    stacks, containers, adapters = hosts
    plots = Plots(adapters[0])
    owner_id = threading.get_ident()
    output = tmp_path / "summary.png"

    def draw(figure: Figure) -> None:
        assert threading.get_ident() == owner_id
        assert get_figure_container(figure) is None
        figure.axes[0].plot([0.0, 1.0], [2.0, 3.0])

    def redraw(figure: Figure) -> None:
        assert threading.get_ident() == owner_id
        assert get_figure_container(figure) is containers[0]
        figure.clear()
        figure.subplots().plot([0.0, 1.0], [4.0, 5.0])

    def work() -> int:
        figure, _ = plots.subplots("summary")
        pending, _ = plots.subplots("pending")
        plots.draw("summary", draw)
        assert adapters[0].call(lambda: get_figure_container(figure)) is containers[0]
        plots.draw("summary", redraw)
        assert adapters[0].call(lambda: stacks[0].count()) == 2
        assert adapters[0].call(lambda: get_figure_container(pending)) is None
        plots.save("summary", output)
        plots.finish()
        return threading.get_ident()

    assert _run_worker(qapp, work) != owner_id
    assert stacks[0].count() == 3
    assert get_figure_container(plots["pending"]) is containers[0]
    np.testing.assert_array_equal(plots["summary"].axes[0].lines[0].get_ydata(), [4, 5])
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    plots.release()
    assert stacks[0].count() == 1


def test_worker_failed_draw_does_not_present_or_rollback(
    qapp: QApplication,
    hosts: tuple[list[QStackedWidget], list[FigureContainer], list[QtPlotHost]],
) -> None:
    stacks, _containers, adapters = hosts
    plots = Plots(adapters[0])
    owner_id = threading.get_ident()
    cause = ValueError("partial drawing")

    def fail(figure: Figure) -> None:
        assert threading.get_ident() == owner_id
        figure.axes[0].set_title("partial")
        raise cause

    def work() -> int:
        figure, _ = plots.subplots("diagnostic")
        with pytest.raises(ValueError, match="partial drawing") as caught:
            plots.draw("diagnostic", fail)
        assert caught.value is cause
        assert adapters[0].call(lambda: get_figure_container(figure)) is None
        assert adapters[0].call(lambda: stacks[0].count()) == 1
        plots.finish(present=False)
        return threading.get_ident()

    assert _run_worker(qapp, work) != owner_id
    assert plots["diagnostic"].axes[0].get_title() == "partial"
    assert get_figure_container(plots["diagnostic"]) is None
    plots.release()


def test_qt_diagnostic_finish_does_not_attach_regular_figure(hosts) -> None:
    stacks, _containers, adapters = hosts
    plots = Plots(adapters[0])
    figure, _ = plots.subplots("diagnostic")
    plots.finish(present=False)
    assert stacks[0].count() == 1
    assert get_figure_container(figure) is None
    plots.release()
    assert isinstance(figure, Figure)
