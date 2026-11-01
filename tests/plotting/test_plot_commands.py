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
    figure, axis = plots.subplots(num="trace", figsize=tile_size, dpi=137)
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
    figure, _ = plots.subplots(num="map", ncols=2, layout="constrained")
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
    figure, axis = plots.subplots(num="decay", figsize=(2.52, 2.0), dpi=137)
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


def test_save_unknown_name_raises_key_error(tmp_path: Path) -> None:
    plots = Plots(NonPresentingHost())
    with pytest.raises(KeyError, match="missing"):
        plots.save("missing", tmp_path / "missing.png")
    plots.finish(present=False)
    plots.release()
