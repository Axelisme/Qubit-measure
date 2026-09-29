from __future__ import annotations

import threading
from collections.abc import Callable
from io import BytesIO

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from qtpy.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
from qtpy.QtWidgets import QApplication, QLabel, QStackedWidget
from zcu_tools.gui.plotting import FigureContainer, get_figure_container
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.plotting.plots import Plots


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
    outcomes: list[int | Exception] = []

    def completed(value: int | Exception) -> None:
        outcomes.append(value)
        loop.quit()

    background.submit(work, on_done=completed, on_error=completed, run_in_pool=False)
    try:
        timer.start(5000)
        loop.exec()
        assert len(outcomes) == 1
        result = outcomes[0]
        if isinstance(result, Exception):
            raise result
        return result
    finally:
        timer.stop()
        assert background.quiesce()
        background.deleteLater()
        qapp.processEvents()


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
        viewer = plots.liveplot_1d("measurement", "Time", "Signal")
        assert adapters[0].call(lambda: stacks[0].count()) == 2
        viewer.update(np.array([0.0, 1.0]), np.array([3.0, 4.0]), refresh=False)
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
    first.liveplot_1d("live", "x", "y")
    second.liveplot_1d("live", "x", "y")
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
    other.liveplot_1d("live", "x", "y")
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


def test_qt_diagnostic_finish_does_not_attach_regular_figure(hosts) -> None:
    stacks, _containers, adapters = hosts
    plots = Plots(adapters[0])
    figure, _ = plots.subplots("diagnostic")
    plots.finish(present=False)
    assert stacks[0].count() == 1
    assert get_figure_container(figure) is None
    plots.release()
    assert isinstance(figure, Figure)
