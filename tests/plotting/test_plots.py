from __future__ import annotations

import gc
from collections.abc import Callable
from io import BytesIO
from typing import TypeVar

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.plotting.figures import FigureCollection
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests._plotting import create_trace

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
    viewer = create_trace(plots, "measurement", "Time", "Signal", title="T1")
    xs, ys = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    viewer.update((xs, ys), refresh=False)
    xs[:] = 99
    ys[:] = 99
    figures = plots.finish()
    line = figures["measurement"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [0, 1])
    np.testing.assert_array_equal(line.get_ydata(), [2, 3])
    assert figures["measurement"].axes[0].get_title() == "T1"
    plots.release()
    output = BytesIO()
    figures["measurement"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(RuntimeError, match="finished"):
        viewer.update((xs, ys))


def test_normal_figures_wait_for_finish_and_last_live_frame_refreshes() -> None:
    host = RecordingHost()
    plots = Plots(host)
    regular, _ = plots.subplots("fit")
    assert host.presented == []
    viewer = create_trace(plots, "measurement", "x", "y")
    live = plots["measurement"]
    assert host.presented == [live]
    viewer.update((np.array([1.0]), np.array([2.0])), refresh=False)
    assert host.refreshed == []
    with pytest.raises(RuntimeError, match="Finish"):
        plots.release()
    result = plots.finish()
    assert result["fit"] is regular
    assert host.presented == [live, regular]
    assert host.refreshed == [(live, True)]
    assert plots.finish() is result
    assert host.presented == [live, regular]
    plots.release()
    plots.release()
    assert host.released == [regular, live]


def test_finished_figures_keep_ownership_without_the_presentation_handle() -> None:
    plots = Plots(NonPresentingHost())
    figure, axes = plots.subplots("fit")
    axes.plot([0.0, 1.0], [2.0, 3.0])

    figures = plots.finish()
    assert figures is not plots
    plots.release()
    del plots
    gc.collect()

    assert list(figures) == ["fit"]
    assert figures["fit"] is figure
    other = FigureCollection()
    with pytest.raises(ValueError, match="owned by another"):
        other.adopt("reclaimed", figure)
    output = BytesIO()
    figures["fit"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


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
        create_trace(plots, "failed", "x", "y")
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
        create_trace(plots, "same", "x", "y")
    assert host.presented == []
    assert plots["same"] is figure
    plots.finish()


def test_live_axes_reject_foreign_reused_and_removed_axes_before_mutation() -> None:
    host = RecordingHost()
    plots = Plots(host)
    figure, _ = plots.subplots("workflow", ncols=3)
    other, other_ax = plots.subplots("other")
    first, second, removed = figure.axes
    figure.delaxes(removed)
    with pytest.raises(ValueError, match="named figure"):
        create_trace(plots, "workflow", axes=other_ax)
    with pytest.raises(ValueError, match="named figure"):
        create_trace(plots, "workflow", axes=removed)
    with pytest.raises(ValueError, match="distinct"):
        plots.create_live("workflow", lambda *_: None, axes=(first, first), count=2)
    with pytest.raises(ValueError, match="named figure"):
        plots.create_live("workflow", lambda *_: None, axes=(second, other_ax), count=2)
    assert not first.lines and not second.images and not other.axes[0].lines
    assert host.presented == []
    create_trace(plots, "workflow", axes=first)
    with pytest.raises(ValueError, match="already belong"):
        create_trace(plots, "workflow", axes=first)
    assert len(first.lines) == 1
    plots.finish()
    plots.release()


def test_typed_plots_share_one_named_frame_and_preserve_it_after_release() -> None:
    host = RecordingHost()
    plots = Plots(host)
    figure, _ = plots.subplots("workflow", ncols=2)
    handles = [create_trace(plots, "workflow", axes=ax) for ax in figure.axes]
    assert host.presented == [figure]
    for handle in handles:
        handle.update((np.array([1.0]), np.array([2.0])), refresh=False)
    plots.refresh("workflow")
    assert host.refreshed == [(figure, False)]
    figures = plots.finish()
    plots.release()
    assert figures["workflow"] is figure
    assert host.released == [figure]
    assert host.refreshed == [(figure, False), (figure, True)]


def test_fork_shares_host_but_has_independent_membership_and_lifetime() -> None:
    host = RecordingHost()
    parent = Plots(host)
    create_trace(parent, "measurement", "x", "y")
    child = parent.fork()
    viewer = create_trace(child, "measurement", "x", "y")
    create_trace(child, "operation/nested", "x", "y")
    assert child["measurement"] is not parent["measurement"]
    assert host.presented == [
        parent["measurement"],
        child["measurement"],
        child["operation/nested"],
    ]
    child.finish()
    with pytest.raises(RuntimeError, match="finished"):
        viewer.update((np.array([0.0]), np.array([1.0])))
    child.release()
    create_trace(parent, "trend", "x", "y")
    with pytest.raises(ValueError, match="another figure"):
        create_trace(parent, "measurement", "x", "y")
    assert host.released == [child["measurement"], child["operation/nested"]]
    parent.finish()
    parent.release()
