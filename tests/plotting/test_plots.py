from __future__ import annotations

import gc
from collections.abc import Callable
from io import BytesIO
from typing import Any, TypeVar, cast

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from zcu_tools.plotting.figures import FigureCollection
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


def test_scatter_copies_phase_colors_and_retains_native_figures_after_release() -> None:
    host = RecordingHost()
    plots = Plots(host)
    viewer = plots.liveplot_scatter("samples", "Flux", "SNR")
    figure = plots["samples"]
    assert host.presented == [figure]
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([4.0, 5.0, np.nan])
    colors = np.array([1.0, 2.0, 3.0])

    def mutate_source() -> None:
        assert np.asarray(figure.axes[0].collections[0].get_offsets()).size == 0
        xs[:] = 10
        ys[:] = 20
        colors[:] = 30

    host.before_call = mutate_source
    viewer.update(xs, ys, colors=colors, title="phases", refresh=False)
    host.before_call = None
    scatter = figure.axes[0].collections[0]
    np.testing.assert_allclose(
        np.asarray(scatter.get_offsets()), [[1, 4], [2, 5], [3, np.nan]]
    )
    np.testing.assert_array_equal(scatter.get_array(), [1, 2, 3])
    assert scatter.get_clim() == (1, 3)
    assert figure.axes[0].get_title() == "phases"
    assert host.refreshed == []
    figures = plots.finish()
    assert host.refreshed == [(figure, True)]
    plots.release()
    assert host.released == [figure]
    assert figures["samples"] is figure
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(RuntimeError, match="finished"):
        viewer.update(xs, ys, colors=colors)


@pytest.mark.parametrize(
    ("xs", "ys", "colors"),
    [
        ([], [], []),
        ([[1.0]], [2.0], [0.0]),
        ([1.0], [2.0, 3.0], [0.0]),
        ([1.0], [2.0], [0.0, 1.0]),
        ([1j], [2.0], [0.0]),
        ([1.0], [2j], [0.0]),
        ([1.0], [2.0], [1j]),
    ],
)
def test_invalid_scatter_update_preserves_points_and_colors(xs, ys, colors) -> None:
    host = RecordingHost()
    plots = Plots(host)
    viewer = plots.liveplot_scatter("samples", "x", "y")
    viewer.update(np.array([1.0]), np.array([2.0]), colors=np.array([3.0]))
    scatter = plots["samples"].axes[0].collections[0]
    with pytest.raises(ValueError, match="Scatter"):
        viewer.update(np.asarray(xs), np.asarray(ys), colors=np.asarray(colors))
    np.testing.assert_array_equal(scatter.get_offsets(), [[1, 2]])
    np.testing.assert_array_equal(scatter.get_array(), [3])
    assert len(host.refreshed) == 1
    plots.finish()
    plots.release()


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


def test_live_axes_configuration_runs_on_owner_before_presenting() -> None:
    class OwnerHost(RecordingHost):
        in_call = False

        def call(self, callback: Callable[[], _T]) -> _T:
            self.in_call = True
            try:
                return callback()
            finally:
                self.in_call = False

    host = OwnerHost()
    plots = Plots(host)
    configured: list[Axes] = []

    def configure(axes: Axes) -> None:
        assert host.in_call
        assert host.presented == []
        configured.append(axes)
        axes.set_xticks([0.0, 1.0], ["I-I", "X-X"], rotation=30)
        axes.lines[0].set_marker("x")
        axes.lines[0].set_linestyle("None")

    viewer = plots.liveplot_1d("gates", "Gate", "Signal", configure_axes=configure)
    axes = plots["gates"].axes[0]
    assert configured == [axes]
    assert host.presented == [plots["gates"]]
    viewer.update(np.array([0.0, 1.0]), np.array([0.2, 0.8]))
    assert configured == [axes]
    assert [tick.get_text() for tick in axes.get_xticklabels()] == ["I-I", "X-X"]
    assert axes.lines[0].get_marker() == "x"
    assert axes.lines[0].get_linestyle() == "None"
    np.testing.assert_array_equal(axes.lines[0].get_ydata(), [0.2, 0.8])
    plots.finish()
    plots.release()


def test_live_axes_configuration_failure_is_not_presented() -> None:
    host = RecordingHost()
    plots = Plots(host)

    def configure(axes: Axes) -> None:
        axes.set_title("incomplete")
        raise ValueError("invalid configuration")

    with pytest.raises(ValueError, match="invalid configuration"):
        plots.liveplot_1d("failed", "x", "y", configure_axes=configure)
    assert host.presented == []
    figures = plots.finish(present=False)
    assert figures["failed"].axes[0].get_title() == "incomplete"
    plots.release()
    assert host.released == [figures["failed"]]


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


@pytest.mark.parametrize("uniform", [True, False])
def test_heatmap_fixed_color_limits_survive_updates(uniform: bool) -> None:
    plots = Plots(NonPresentingHost())
    try:
        viewer = plots.liveplot_2d(
            "population", "x", "y", uniform=uniform, clim=(0.0, 1.0)
        )
        xs, ys = np.array([0.0, 1.0]), np.array([2.0, 3.0])
        for value in (0.2, 0.8):
            viewer.update(xs, ys, np.full((2, 2), value))
            image = plots["population"].axes[0].images[0]
            assert image.get_clim() == (0.0, 1.0)
            np.testing.assert_allclose(np.asarray(image.get_array()), value)
    finally:
        plots.finish(present=False)
        plots.release()


@pytest.mark.parametrize("uniform", [True, False])
def test_plain_heatmap_preserves_data_and_native_figure_across_lifecycle(
    uniform: bool,
) -> None:
    host = RecordingHost()
    plots = Plots(host)
    viewer = plots.liveplot_2d("scan", "Time", "Phase", uniform=uniform)
    figure = plots["scan"]
    assert host.presented == [figure]
    (axes,) = figure.axes
    xs, ys = np.array([0.0, 1.0]), np.array([10.0, 20.0])
    data = np.array([[1.0, 2.0], [3.0, np.nan]])

    def mutate_source() -> None:
        xs[:] = 99
        ys[:] = 99
        data[:] = 99

    host.before_call = mutate_source
    viewer.update(xs, ys, data, "partial scan", refresh=False)
    expected = np.array([[1.0, 3.0], [2.0, np.nan]])
    np.testing.assert_allclose(
        np.asarray(axes.images[0].get_array()), expected, equal_nan=True
    )
    extent = (-0.5, 1.5, 5.0, 25.0) if uniform else (0.0, 1.0, 10.0, 20.0)
    assert axes.images[0].get_extent() == pytest.approx(extent)
    assert axes.get_title() == "partial scan"
    assert axes.get_xlabel() == "Time"
    assert axes.get_ylabel() == "Phase"
    assert host.refreshed == []

    dispatched: list[bool] = []
    host.before_call = lambda: dispatched.append(True)
    for bad_xs, bad_data in (
        (xs, np.ones((1, 3))),
        (np.array([], dtype=float), data),
        (xs, np.ones((2, 2), dtype=complex)),
    ):
        with pytest.raises(ValueError, match="real-valued|non-empty|shape"):
            viewer.update(bad_xs, ys, bad_data)
    assert dispatched == []
    np.testing.assert_allclose(
        np.asarray(axes.images[0].get_array()), expected, equal_nan=True
    )
    with pytest.raises(ValueError, match="already belongs"):
        plots.liveplot_2d("scan", "x", "y")
    assert host.presented == [figure]

    host.before_call = None
    viewer.update(np.array([0.0, 1.0]), np.array([10.0, 20.0]), expected.T)
    assert host.refreshed == [(figure, False)]
    assert plots.finish()["scan"] is figure
    assert host.refreshed == [(figure, False), (figure, True)]
    plots.release()
    assert host.released == [figure]
    with pytest.raises(RuntimeError, match="finished"):
        viewer.update(xs, ys, data)
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_heatmap_line_plot_keeps_partial_scan_lines_and_native_figure() -> None:
    plots = Plots(NonPresentingHost())
    viewer = plots.liveplot_2d_with_line(
        "measurement", "Flux device value", "Frequency (MHz)", num_lines=2
    )
    values = np.array([0.0, 1.0, 2.0])
    freqs = np.array([10.0, 20.0])
    data = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]])
    viewer.update(values, freqs, data, "last completed scan")
    values[:] = 99
    freqs[:] = 99
    data[:] = 99
    figure = plots.finish()["measurement"]
    heatmap, recent = figure.axes
    np.testing.assert_allclose(
        np.asarray(heatmap.images[0].get_array()),
        [[1.0, 3.0, np.nan], [2.0, 4.0, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_array_equal(recent.lines[0].get_ydata(), [1.0, 2.0])
    np.testing.assert_array_equal(recent.lines[1].get_ydata(), [3.0, 4.0])
    np.testing.assert_array_equal(recent.lines[1].get_xdata(), [10.0, 20.0])
    assert heatmap.get_title() == "last completed scan"
    plots.release()
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(RuntimeError, match="finished"):
        viewer.update(values, freqs, data)


def test_2d_update_copies_all_arrays_before_owner_dispatch_and_refreshes_on_finish() -> (
    None
):
    host = RecordingHost()
    plots = Plots(host)
    viewer = plots.liveplot_2d_with_line("scan", "x", "y")
    figure = plots["scan"]
    assert host.presented == [figure]
    xs, ys = np.array([0.0, 1.0]), np.array([10.0, 20.0])
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    def mutate_source() -> None:
        xs[:] = 99
        ys[:] = 99
        data[:] = 99

    host.before_call = mutate_source
    viewer.update(xs, ys, data, refresh=False)
    heatmap, recent = figure.axes
    np.testing.assert_array_equal(
        np.asarray(heatmap.images[0].get_array()), [[1.0, 3.0], [2.0, 4.0]]
    )
    np.testing.assert_array_equal(recent.lines[0].get_xdata(), [10.0, 20.0])
    np.testing.assert_array_equal(recent.lines[0].get_ydata(), [3.0, 4.0])
    assert heatmap.images[0].get_extent() == pytest.approx((-0.5, 1.5, 5.0, 25.0))
    assert host.refreshed == []
    assert plots.finish()["scan"] is figure
    assert host.refreshed == [(figure, True)]
    plots.release()
    assert host.released == [figure]


def test_nonuniform_2d_column_lines_and_invalid_update_preserve_artists() -> None:
    host = RecordingHost()
    plots = Plots(host)
    viewer = plots.liveplot_2d_with_line(
        "nonuniform", "x", "y", line_axis=0, num_lines=2, uniform=False
    )
    xs, ys = np.array([0.0, 1.0, 5.0]), np.array([10.0, 13.0])
    data = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, np.nan]])
    viewer.update(xs, ys, data)
    figure = plots["nonuniform"]
    heatmap, recent = figure.axes
    original = np.asarray(heatmap.images[0].get_array()).copy()
    np.testing.assert_array_equal(recent.lines[0].get_ydata(), [1.0, 3.0, 5.0])
    np.testing.assert_allclose(
        np.asarray(recent.lines[1].get_ydata()), [2.0, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_array_equal(recent.lines[1].get_xdata(), xs)

    # At x=3.5 the nearest nonuniform column is x=5; equal-width cells
    # would still display the x=1 column. Observe the rendered heatmap.
    canvas = figure.canvas
    assert isinstance(canvas, FigureCanvasAgg)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())

    def color_at(x: float) -> tuple[int, ...]:
        px, py = heatmap.transData.transform((x, 10.0))
        return tuple(int(v) for v in rgba[rgba.shape[0] - 1 - int(py), int(px), :3])

    assert color_at(2.0) != color_at(3.5)
    assert color_at(3.5) == color_at(4.6)

    attempted_calls: list[bool] = []
    host.before_call = lambda: attempted_calls.append(True)
    for bad_xs, bad_data in (
        (xs, np.ones((1, 3))),
        (np.array([], dtype=float), data),
        (xs, np.ones((3, 2), dtype=complex)),
    ):
        with pytest.raises(ValueError, match="real-valued|non-empty|shape"):
            viewer.update(bad_xs, ys, bad_data)
    assert attempted_calls == []
    np.testing.assert_allclose(
        np.asarray(heatmap.images[0].get_array()), original, equal_nan=True
    )
    np.testing.assert_array_equal(recent.lines[0].get_ydata(), [1.0, 3.0, 5.0])
    np.testing.assert_allclose(
        np.asarray(recent.lines[1].get_ydata()), [2.0, np.nan, np.nan], equal_nan=True
    )
    plots.finish()
    plots.release()
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_invalid_2d_factory_options_or_duplicate_name_never_present() -> None:
    host = RecordingHost()
    plots = Plots(host)
    with pytest.raises(ValueError, match="Line count"):
        plots.liveplot_2d_with_line("scan", "x", "y", num_lines=0)
    with pytest.raises(ValueError, match="line_axis"):
        plots.liveplot_2d_with_line("scan", "x", "y", line_axis=cast(Any, 2))
    assert list(plots) == []
    existing, _ = plots.subplots("scan")
    with pytest.raises(ValueError, match="already belongs"):
        plots.liveplot_2d_with_line("scan", "x", "y")
    assert host.presented == []
    assert plots["scan"] is existing
    plots.finish()


def test_typed_plots_share_one_named_frame_and_preserve_it_after_release() -> None:
    host = RecordingHost()
    plots = Plots(host)
    figure, _ = plots.subplots("workflow", ncols=5)
    axes = figure.axes
    line = plots.liveplot_1d("workflow", "flux", "value", axes=axes[0])
    samples = plots.liveplot_scatter("workflow", "flux", "snr", axes=axes[4])
    heatmap = plots.liveplot_2d("workflow", "flux", "time", axes=axes[1])
    scan = plots.liveplot_2d_with_line(
        "workflow", "flux", "frequency", axes=(axes[2], axes[3])
    )
    xs = np.array([1.0, 2.0])
    ys = np.array([3.0, 4.0, 5.0])
    data = np.arange(6, dtype=float).reshape(2, 3)
    line.update(xs, np.array([8.0, 9.0]), refresh=False)
    heatmap.update(xs, ys, data, refresh=False)
    scan.update(xs, ys, data, refresh=False)
    samples.update(xs, np.array([6.0, 7.0]), colors=np.array([0.0, 1.0]), refresh=False)
    assert list(plots) == ["workflow"]
    assert host.presented == [figure]
    assert host.refreshed == []
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), [8.0, 9.0])
    np.testing.assert_array_equal(axes[1].images[0].get_array(), data.T)
    np.testing.assert_array_equal(axes[2].images[0].get_array(), data.T)
    np.testing.assert_array_equal(axes[3].lines[0].get_ydata(), data[-1])
    np.testing.assert_array_equal(
        axes[4].collections[0].get_offsets(), [[1.0, 6.0], [2.0, 7.0]]
    )
    heatmap.mark_point(1.0, 4.0)
    scan.mark_line(3.5)
    np.testing.assert_array_equal(axes[1].collections[0].get_offsets(), [[1.0, 4.0]])
    np.testing.assert_array_equal(axes[3].lines[-1].get_xdata(), [3.5])
    heatmap.mark_point(np.nan, np.nan)
    scan.mark_line(np.nan)
    assert len(axes[1].collections) == 1
    assert len(axes[3].lines) == 2
    assert np.isnan(axes[1].collections[0].get_offsets()).all()
    assert np.isnan(axes[3].lines[-1].get_xdata()).all()
    plots.refresh("workflow")
    assert host.refreshed == [(figure, False)]
    named = plots.finish()
    plots.release()
    assert host.refreshed == [(figure, False), (figure, True)]
    assert host.released == [figure]
    assert named["workflow"] is figure
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")
    with pytest.raises(RuntimeError, match="finished"):
        plots.refresh("workflow")


def test_live_axes_reject_foreign_reused_and_removed_axes_before_mutation() -> None:
    host = RecordingHost()
    plots = Plots(host)
    figure, _ = plots.subplots("workflow", ncols=3)
    other, other_ax = plots.subplots("other")
    first, second, removed = figure.axes
    figure.delaxes(removed)
    with pytest.raises(ValueError, match="named figure"):
        plots.liveplot_1d("workflow", "x", "y", axes=other_ax)
    with pytest.raises(ValueError, match="named figure"):
        plots.liveplot_2d("workflow", "x", "y", axes=removed)
    with pytest.raises(ValueError, match="distinct"):
        plots.liveplot_2d_with_line("workflow", "x", "y", axes=(first, first))
    with pytest.raises(ValueError, match="named figure"):
        plots.liveplot_2d_with_line("workflow", "x", "y", axes=(second, other_ax))
    assert not first.lines and not second.images and not other.axes[0].lines
    assert host.presented == []
    plots.liveplot_1d("workflow", "x", "y", axes=first)
    with pytest.raises(ValueError, match="already belong"):
        plots.liveplot_2d("workflow", "x", "y", axes=first)
    with pytest.raises(ValueError, match="already belong"):
        plots.liveplot_scatter("workflow", "x", "y", axes=first)
    with pytest.raises(ValueError, match="named figure"):
        plots.liveplot_scatter("workflow", "x", "y", axes=other_ax)
    assert len(first.lines) == 1
    assert not first.images
    plots.finish()


@pytest.mark.parametrize("close_explicitly", [False, True])
def test_movie_captures_updated_frame_and_closes_on_owner(
    monkeypatch: pytest.MonkeyPatch, close_explicitly: bool
) -> None:
    class OwnerHost(RecordingHost):
        on_owner = False

        def call(self, callback: Callable[[], _T]) -> _T:
            previous = self.on_owner
            self.on_owner = True
            try:
                return super().call(callback)
            finally:
                self.on_owner = previous

    host = OwnerHost()
    frames: list[bytes] = []
    closed: list[bool] = []
    attached: list[Figure] = []

    class Writer:
        def __init__(self, *, fps: int) -> None:
            assert host.on_owner
            self.figure: Figure | None = None

        @classmethod
        def isAvailable(cls) -> bool:
            return True

        def setup(self, figure: Figure, filename: str, dpi: int) -> None:
            assert host.on_owner
            self.figure = figure
            attached.append(figure)

        def grab_frame(self) -> None:
            assert host.on_owner
            assert self.figure is not None
            output = BytesIO()
            self.figure.savefig(output, format="png")
            frames.append(output.getvalue())

        def finish(self) -> None:
            assert host.on_owner
            closed.append(True)

    monkeypatch.setattr("matplotlib.animation.FFMpegWriter", Writer)
    plots = Plots(host)
    viewer = plots.liveplot_1d("workflow", "x", "y")
    recorder = plots.record_animation("workflow", "unused.mp4")
    recorder.grab_frame()
    viewer.update(np.array([1.0, 2.0]), np.array([4.0, 9.0]), refresh=False)
    recorder.grab_frame()
    assert len(frames) == 2 and frames[0] != frames[1]
    assert attached == [plots["workflow"]]
    with pytest.raises(ValueError, match="already has"):
        plots.record_animation("workflow", "duplicate.mp4")
    if close_explicitly:
        recorder.finish()
        with pytest.raises(RuntimeError, match="finished"):
            recorder.grab_frame()
    named = plots.finish()
    recorder.finish()
    assert closed == [True]
    assert named["workflow"] is attached[0]
    assert host.released == []
    with pytest.raises(RuntimeError, match="finished"):
        recorder.grab_frame()
    plots.release()
    assert host.released == attached
