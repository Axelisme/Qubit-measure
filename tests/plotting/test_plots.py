from __future__ import annotations

import gc
from collections.abc import Callable
from io import BytesIO
from typing import TypeVar

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.plotting.plots import NonPresentingHost, Plots

_T = TypeVar("_T")


class RecordingHost(NonPresentingHost):
    def __init__(self) -> None:
        self.presented: list[Figure] = []
        self.refreshed: list[tuple[Figure, bool]] = []
        self.released: list[Figure] = []
        self.fail_release: Figure | None = None
        self.before_call: Callable[[], None] | None = None

    def call(self, callback: Callable[[], _T]) -> _T:
        if self.before_call is not None:
            self.before_call()
        return callback()

    def present(self, figure: Figure) -> None:
        self.presented.append(figure)

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        self.refreshed.append((figure, final))

    def release(self, figure: Figure) -> None:
        self.released.append(figure)
        if figure is self.fail_release:
            raise OSError("release failed")


@pytest.fixture(autouse=True)
def collect_plot_cycles():
    yield
    gc.collect()


def test_live_updates_and_native_save_work_without_presentation() -> None:
    plots = Plots(NonPresentingHost())
    viewer = plots.liveplot_1d("measurement", "Time", "Signal", title="T1")
    xs, ys = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    viewer.update(xs, ys, "last frame", refresh=False)
    xs[:] = 99
    ys[:] = 99
    figures = plots.finish()
    line = figures["measurement"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [0, 1])
    np.testing.assert_array_equal(line.get_ydata(), [2, 3])
    assert figures["measurement"].axes[0].get_title() == "last frame"
    plots.release()
    output = BytesIO()
    figures["measurement"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(RuntimeError, match="finished"):
        viewer.update(xs, ys)


def test_normal_figures_wait_for_finish_and_last_live_frame_refreshes() -> None:
    host = RecordingHost()
    plots = Plots(host)
    regular, _ = plots.subplots("fit")
    assert host.presented == []
    viewer = plots.liveplot_1d("measurement", "x", "y")
    live = plots["measurement"]
    assert host.presented == [live]
    viewer.update(np.array([1.0]), np.array([2.0]), refresh=False)
    assert host.refreshed == []
    with pytest.raises(RuntimeError, match="Finish"):
        plots.release()
    result = plots.finish()
    assert result is plots
    assert host.presented == [live, regular]
    assert host.refreshed == [(live, True)]
    assert plots.finish() is result
    assert host.presented == [live, regular]
    plots.release()
    plots.release()
    assert host.released == [regular, live]


def test_data_is_copied_before_host_dispatch() -> None:
    host = RecordingHost()
    plots = Plots(host)
    viewer = plots.liveplot_1d("live", "x", "y")
    xs, ys = np.array([0.0, 1.0]), np.array([2.0, 3.0])

    def mutate_source() -> None:
        xs[:] = 10
        ys[:] = 20

    host.before_call = mutate_source
    viewer.update(xs, ys)
    line = plots["live"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [0, 1])
    np.testing.assert_array_equal(line.get_ydata(), [2, 3])
    plots.finish()


@pytest.mark.parametrize(
    "bad_data",
    [np.array([1.0]), np.ones((3, 2)), np.ones((2, 2), dtype=complex)],
)
def test_invalid_update_keeps_every_existing_line(bad_data) -> None:
    plots = Plots(NonPresentingHost())
    viewer = plots.liveplot_1d("live", "x", "y", num_lines=2)
    xs = np.array([0.0, 1.0])
    viewer.update(xs, np.array([[2.0, 3.0], [4.0, 5.0]]))
    with pytest.raises(ValueError, match="real-valued|line count"):
        viewer.update(xs, bad_data)
    lines = plots["live"].axes[0].lines
    np.testing.assert_array_equal(lines[0].get_ydata(), [2, 3])
    np.testing.assert_array_equal(lines[1].get_ydata(), [4, 5])
    plots.finish()


def test_diagnostic_finish_does_not_present_ordinary_figures() -> None:
    host = RecordingHost()
    plots = Plots(host)
    figure, _ = plots.subplots("diagnostic")
    assert plots.finish(present=False)["diagnostic"] is figure
    assert host.presented == []
    with pytest.raises(RuntimeError, match="sealed"):
        plots.subplots("late")
    plots.release()


def test_release_attempts_every_figure_and_retains_failed_item_for_retry() -> None:
    host = RecordingHost()
    plots = Plots(host)
    first, _ = plots.subplots("first")
    second, _ = plots.subplots("second")
    plots.finish()
    host.fail_release = first
    with pytest.raises(ExceptionGroup, match="Failed to release") as captured:
        plots.release()
    assert [str(error) for error in captured.value.exceptions] == ["release failed"]
    assert host.released == [first, second]
    assert plots["first"] is first
    host.fail_release = None
    plots.release()
    assert host.released == [first, second, first]


def test_failed_live_presentation_can_finish_diagnostics_and_release() -> None:
    class FailingHost(RecordingHost):
        def present(self, figure: Figure) -> None:
            raise OSError("present failed")

    host = FailingHost()
    plots = Plots(host)
    with pytest.raises(OSError, match="present failed"):
        plots.liveplot_1d("failed", "x", "y")
    figures = plots.finish(present=False)
    assert list(figures) == ["failed"]
    assert host.refreshed == []
    plots.release()
    assert host.released == [figures["failed"]]


def test_live_name_conflict_does_not_present_another_figure() -> None:
    host = RecordingHost()
    plots = Plots(host)
    figure, _ = plots.subplots("same")
    with pytest.raises(ValueError, match="already belongs"):
        plots.liveplot_1d("same", "x", "y")
    assert host.presented == []
    assert plots["same"] is figure
    plots.finish()
