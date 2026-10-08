from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.plotting.plots import NonPresentingHost, Plots

_T = TypeVar("_T")


class OwnerHost(NonPresentingHost):
    """Observe synchronous owner dispatch without replacing plotting behavior."""

    def __init__(self) -> None:
        self.in_call = False
        self.refreshed: list[Figure] = []

    def call(self, callback: Callable[[], _T]) -> _T:
        self.in_call = True
        try:
            return callback()
        finally:
            self.in_call = False

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        assert self.in_call
        self.refreshed.append(figure)


def test_draw_shares_template_axes_and_refreshes_after_callback() -> None:
    host = OwnerHost()
    plots = Plots(host)
    viewer = plots.liveplot_1d("frequency", "MHz", "Signal")
    figure = plots["frequency"]

    def references(target: Figure) -> None:
        assert target is figure
        assert host.in_call
        assert host.refreshed == []
        target.axes[0].axvline(5000.0, label="expected")
        target.axes[0].axvline(4990.0, label="previous")

    plots.draw("frequency", references)
    assert host.refreshed == [figure]
    assert not host.in_call
    viewer.update(np.array([4990.0, 5000.0]), np.array([0.2, 0.8]))
    lines = figure.axes[0].lines
    np.testing.assert_array_equal(lines[0].get_ydata(), [0.2, 0.8])
    assert [line.get_label() for line in lines[1:]] == ["expected", "previous"]
    plots.finish()
    plots.release()


def test_draw_rebuilds_same_named_figure_for_successive_live_templates() -> None:
    host = OwnerHost()
    plots = Plots(host)
    previous = plots.liveplot_1d("decay", "Time", "Signal")
    figure = plots["decay"]
    xs = np.array([0.0, 1.0])
    previous.update(xs, xs)

    def rebuild(target: Figure) -> None:
        assert host.in_call
        target.clear()
        target.subplots()

    for attempt in range(3):
        plots.draw("decay", rebuild)
        current = plots.liveplot_1d("decay", "Time", "Signal", axes=figure.axes[0])
        current.update(xs, xs + attempt)
        assert plots["decay"] is figure
        assert len(figure.axes) == 1
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), xs + attempt)
        with pytest.raises(RuntimeError, match="removed from.*axes"):
            previous.update(xs, xs)
        previous = current
    plots.finish()
    plots.release()


def test_draw_failure_preserves_cause_without_refresh_or_rollback() -> None:
    host = OwnerHost()
    plots = Plots(host)
    figure, _ = plots.subplots("diagnostic")
    error = ValueError("bad fit overlay")

    def fail(target: Figure) -> None:
        assert host.in_call
        target.axes[0].set_title("partial diagnostic")
        raise error

    with pytest.raises(ValueError, match="bad fit overlay") as caught:
        plots.draw("diagnostic", fail)
    assert caught.value is error
    assert host.refreshed == []
    assert figure.axes[0].get_title() == "partial diagnostic"
    plots.finish(present=False)
    plots.release()


@pytest.mark.parametrize("finished", [False, True])
def test_save_uses_owner_without_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finished: bool
) -> None:
    host = OwnerHost()
    plots = Plots(host)
    plots.liveplot_1d("decay", "Time", "Signal")
    figure = plots["decay"]
    if finished:
        plots.finish()
    host.refreshed.clear()
    native_savefig = Figure.savefig
    saved: list[Figure] = []

    def observe_savefig(target: Figure, path: str | Path) -> None:
        assert host.in_call
        saved.append(target)
        native_savefig(target, path)

    monkeypatch.setattr(Figure, "savefig", observe_savefig)
    path = tmp_path / "decay.png"
    plots.save("decay", path)
    assert saved == [figure]
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert host.refreshed == []
    assert not host.in_call
    plots.finish()
    plots.release()


def test_save_io_error_propagates_without_refresh(tmp_path: Path) -> None:
    host = OwnerHost()
    plots = Plots(host)
    plots.subplots("decay")
    with pytest.raises(FileNotFoundError):
        plots.save("decay", tmp_path / "missing" / "decay.png")
    assert host.refreshed == []
    plots.finish(present=False)
    plots.release()


def test_unknown_names_and_finished_draw_do_not_execute_callback(
    tmp_path: Path,
) -> None:
    plots = Plots(NonPresentingHost())
    calls: list[Figure] = []
    with pytest.raises(KeyError, match="missing"):
        plots.draw("missing", calls.append)
    with pytest.raises(KeyError, match="missing"):
        plots.save("missing", tmp_path / "missing.png")
    plots.subplots("retained")
    plots.finish(present=False)
    with pytest.raises(RuntimeError, match="finished"):
        plots.draw("retained", calls.append)
    assert calls == []
    plots.release()


@pytest.mark.parametrize("kind", ["point", "line"])
@pytest.mark.parametrize("removal", ["marker", "cla", "axes", "figure"])
def test_marker_reports_draw_removal(kind: str, removal: str) -> None:
    host = OwnerHost()
    plots = Plots(host)
    if kind == "point":
        heatmap = plots.liveplot_2d("live", "x", "y")

        def mark() -> None:
            heatmap.mark_point(0.5, 0.75)
    else:
        combined = plots.liveplot_2d_with_line("live", "x", "y")

        def mark() -> None:
            combined.mark_line(0.5)

    mark()

    def remove(target: Figure) -> None:
        assert host.in_call
        axes = target.axes[0 if kind == "point" else 1]
        if removal == "marker":
            if kind == "point":
                marker = next(
                    item
                    for item in axes.collections
                    if item.get_label() == "Best Point"
                )
                marker.remove()
            else:
                axes.lines[-1].remove()
        elif removal == "cla":
            axes.cla()
        elif removal == "axes":
            target.delaxes(axes)
        else:
            target.clear()

    plots.draw("live", remove)
    host.refreshed.clear()
    with pytest.raises(RuntimeError, match="removed from.*axes"):
        mark()
    assert host.refreshed == []
    plots.finish()
    plots.release()


@pytest.mark.parametrize("kind", ["point", "line"])
def test_marker_keeps_valid_overlay_after_draw(kind: str) -> None:
    host = OwnerHost()
    plots = Plots(host)
    if kind == "point":
        heatmap = plots.liveplot_2d("live", "x", "y")

        def mark(value: float) -> None:
            heatmap.mark_point(value, value)
    else:
        combined = plots.liveplot_2d_with_line("live", "x", "y")

        def mark(value: float) -> None:
            combined.mark_line(value)

    mark(0.25)

    def overlay(target: Figure) -> None:
        axes = target.axes[0 if kind == "point" else 1]
        axes.set_title("overlay")
        axes.text(0.0, 0.0, "reference")

    plots.draw("live", overlay)
    mark(0.75)
    figures = plots.finish()
    axes = figures["live"].axes[0 if kind == "point" else 1]
    if kind == "point":
        marker = next(
            item for item in axes.collections if item.get_label() == "Best Point"
        )
        assert np.array_equal(marker.get_offsets(), [[0.75, 0.75]])
    else:
        assert np.array_equal(axes.lines[-1].get_xdata(), [0.75])
    plots.release()


def _remove_template_artists(target: Figure, template: str, removal: str) -> None:
    axes = target.axes[1 if template == "scan_line" else 0]
    if removal == "clear_figure":
        target.clear()
    elif removal == "remove_axes":
        target.delaxes(axes)
    elif removal == "detach_axes":
        axes.remove()
    elif removal == "clear_axes":
        axes.cla()
    elif template in ("line", "scan_line"):
        axes.lines[0].remove()
    elif template == "scatter":
        axes.collections[0].remove()
    else:
        axes.images[0].remove()


@pytest.mark.parametrize(
    "template",
    ["line", "scatter", "heatmap", "nonuniform", "heatmap_line", "scan_line"],
)
@pytest.mark.parametrize(
    "removal",
    ["clear_axes", "remove_artist", "clear_figure", "remove_axes", "detach_axes"],
)
def test_template_update_reports_artists_removed_by_draw(
    template: str, removal: str
) -> None:
    host = OwnerHost()
    plots = Plots(host)
    xs, ys = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    grid = np.ones((2, 2))
    if template == "line":
        line = plots.liveplot_1d("live", "x", "y", num_lines=2)

        def update() -> None:
            line.update(xs, np.ones((2, 2)))
    elif template == "scatter":
        scatter = plots.liveplot_scatter("live", "x", "y")

        def update() -> None:
            scatter.update(xs, ys, colors=ys)
    elif template in ("heatmap", "nonuniform"):
        heatmap = plots.liveplot_2d("live", "x", "y", uniform=template == "heatmap")

        def update() -> None:
            heatmap.update(xs, ys, grid)
    else:
        combined = plots.liveplot_2d_with_line("live", "x", "y")

        def update() -> None:
            combined.update(xs, ys, grid)

    update()

    def remove(target: Figure) -> None:
        assert host.in_call
        _remove_template_artists(target, template, removal)

    plots.draw("live", remove)
    host.refreshed.clear()
    with pytest.raises(RuntimeError, match="removed from.*axes"):
        update()
    assert host.refreshed == []
    plots.finish()
    plots.release()
