"""Active liveplot backend selection: registered (ContextVar → default) → name."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from zcu_tools.liveplot.backend import (
    FallbackBackend,
    JupyterBackend,
    LivePlotBackend,
    active_backend,
    set_default_liveplot_backend,
    set_liveplot_backend,
)


class _StubBackend(LivePlotBackend):
    def make_plot_frame(
        self, n_row: int, n_col: int, plot_instant: bool = False, **kwargs: Any
    ) -> tuple[Figure, list[list[Axes]]]:
        raise NotImplementedError

    def instant_plot(self, fig: Figure) -> None:
        raise NotImplementedError

    def refresh_figure(self, fig: Figure) -> None:
        raise NotImplementedError

    def close_figure(self, fig: Figure) -> None:
        raise NotImplementedError


def test_registered_backend_takes_priority():
    stub = _StubBackend()
    with set_liveplot_backend(stub):
        assert active_backend() is stub


def test_set_liveplot_backend_restores_after_block():
    before = active_backend()
    with set_liveplot_backend(_StubBackend()):
        pass
    # outside the block, selection falls back again (not the stub)
    assert not isinstance(active_backend(), _StubBackend)
    assert type(active_backend()) is type(before)


def test_nested_registration_restores_outer():
    outer, inner = _StubBackend(), _StubBackend()
    with set_liveplot_backend(outer):
        assert active_backend() is outer
        with set_liveplot_backend(inner):
            assert active_backend() is inner
        assert active_backend() is outer


def test_process_default_used_when_nothing_registered():
    stub = _StubBackend()
    try:
        set_default_liveplot_backend(stub)
        assert active_backend() is stub
    finally:
        set_default_liveplot_backend(None)


def test_registered_overrides_process_default():
    default_stub, registered_stub = _StubBackend(), _StubBackend()
    try:
        set_default_liveplot_backend(default_stub)
        with set_liveplot_backend(registered_stub):
            assert active_backend() is registered_stub
    finally:
        set_default_liveplot_backend(None)


def test_name_fallback_when_unregistered():
    # No registration, no default: pick by matplotlib backend name. Under the
    # test process (Agg / module:// gui backend) this resolves to FallbackBackend.
    set_default_liveplot_backend(None)
    backend = active_backend()
    assert isinstance(backend, (FallbackBackend, JupyterBackend))
