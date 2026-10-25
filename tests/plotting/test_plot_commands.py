from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
from matplotlib import rc_context
from matplotlib.figure import Figure
from PIL import Image
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests._plotting import create_trace

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
    viewer = create_trace(plots, "frequency", "MHz", "Signal")
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
    viewer.update((np.array([4990.0, 5000.0]), np.array([0.2, 0.8])))
    lines = figure.axes[0].lines
    np.testing.assert_array_equal(lines[0].get_ydata(), [0.2, 0.8])
    assert [line.get_label() for line in lines[1:]] == ["expected", "previous"]
    plots.finish()
    plots.release()


def test_draw_rebuilds_same_named_figure_for_successive_live_templates() -> None:
    host = OwnerHost()
    plots = Plots(host)
    previous = create_trace(plots, "decay", "Time", "Signal")
    figure = plots["decay"]
    xs = np.array([0.0, 1.0])
    previous.update((xs, xs))

    def rebuild(target: Figure) -> None:
        assert host.in_call
        target.clear()
        target.subplots()

    for attempt in range(3):
        plots.draw("decay", rebuild)
        current = create_trace(plots, "decay", "Time", "Signal", axes=figure.axes[0])
        current.update((xs, xs + attempt))
        assert plots["decay"] is figure
        assert len(figure.axes) == 1
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), xs + attempt)
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
    create_trace(plots, "decay", "Time", "Signal")
    figure = plots["decay"]
    if finished:
        plots.finish()
    host.refreshed.clear()
    native_savefig = Figure.savefig
    saved: list[Figure] = []

    def observe_savefig(target: Figure, path: str | Path, *, dpi: float) -> None:
        assert host.in_call
        saved.append(target)
        native_savefig(target, path, dpi=dpi)

    monkeypatch.setattr(Figure, "savefig", observe_savefig)
    path = tmp_path / "decay.png"
    plots.save("decay", path)
    assert saved == [figure]
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert host.refreshed == []
    assert not host.in_call
    plots.finish()
    plots.release()


@pytest.mark.parametrize("tile_size", [(2.52, 2.0), (12.0, 9.0)])
def test_save_png_size_is_independent_of_display(
    tmp_path: Path, tile_size: tuple[float, float]
) -> None:
    plots = Plots(NonPresentingHost())
    figure, axis = plots.subplots("trace", figsize=tile_size, dpi=137)
    axis.plot([0, 1], [0, 1], label="fit")
    axis.set(xlabel="Delay (us)", ylabel="Measured signal (a.u.)")
    axis.legend()
    canvas = figure.canvas
    original_position = axis.get_position().frozen()
    original_layout = figure.get_layout_engine()

    path = tmp_path / "trace.png"
    plots.save("trace", path)

    with Image.open(path) as image:
        assert image.size == (640, 480)
    assert tuple(figure.get_size_inches()) == tile_size
    assert figure.dpi == 137
    assert figure.canvas is canvas
    assert figure.get_layout_engine() is original_layout
    np.testing.assert_allclose(axis.get_position().bounds, original_position.bounds)
    assert axis.get_in_layout()
    plots.finish(present=False)
    plots.release()


def test_save_multi_panel_png_uses_explicit_output_geometry(tmp_path: Path) -> None:
    plots = Plots(NonPresentingHost())
    figure, _ = plots.subplots("map", ncols=2, layout="constrained")
    axis = figure.axes[0]
    image = axis.imshow([[1, 2], [3, 4]])
    figure.colorbar(image, ax=axis, label="SNR")
    figure.axes[1].set(xlabel="Pulse length (us)", ylabel="Signal (a.u.)")
    original_layout = figure.get_layout_engine()
    figure.set_size_inches(2.52, 2.0, forward=False)
    path = tmp_path / "map.png"

    with rc_context({"savefig.bbox": "tight"}):
        plots.save("map", path, figsize=(10.0, 4.8), dpi=150)

    with Image.open(path) as output:
        assert output.size == (1500, 720)
    assert tuple(figure.get_size_inches()) == (2.52, 2.0)
    assert figure.get_layout_engine() is original_layout
    plots.finish(present=False)
    plots.release()


def test_save_io_error_propagates_without_refresh(tmp_path: Path) -> None:
    host = OwnerHost()
    plots = Plots(host)
    figure, axis = plots.subplots("decay", figsize=(2.52, 2.0), dpi=137)
    original_layout = figure.get_layout_engine()
    original_position = axis.get_position().frozen()
    with pytest.raises(FileNotFoundError):
        plots.save("decay", tmp_path / "missing" / "decay.png")
    assert tuple(figure.get_size_inches()) == (2.52, 2.0)
    assert figure.dpi == 137
    assert figure.get_layout_engine() is original_layout
    np.testing.assert_allclose(axis.get_position().bounds, original_position.bounds)
    assert axis.get_in_layout()
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
