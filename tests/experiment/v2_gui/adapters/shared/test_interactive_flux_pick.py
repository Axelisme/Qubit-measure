"""Tests for the shared interactive flux-pick session (no Qt, no host widget).

A fake InteractiveHost (a real matplotlib Figure + a synchronous run_background)
stands in for the GUI; the session is driven through its InteractiveSession
methods and its FluxPickResult is asserted.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from zcu_tools.experiment.v2_gui.adapters.shared import (
    FluxPickParams,
    FluxPickResult,
    build_flux_pick_session,
)
from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


class _FakeHost:
    """Minimal InteractiveHost: a real Figure + counters; run_background runs the
    compute synchronously so the test can assert the applied result."""

    def __init__(self, figure: Figure) -> None:
        self.figure = figure
        self.redraws = 0
        self.bg_calls = 0

    def redraw(self) -> None:
        self.redraws += 1

    def run_background(self, compute, on_done) -> None:
        self.bg_calls += 1
        on_done(compute())


def _run_result(n_dev: int = 60, n_freq: int = 30):
    devs = np.linspace(-5.0, 5.0, n_dev).astype(np.float64)
    freqs = np.linspace(4.0, 5.0, n_freq).astype(np.float64)
    sig = np.zeros((n_dev, n_freq), dtype=np.complex128)
    sig += np.exp(-(devs[:, None] ** 2) / (2 * 1.0**2))
    return SimpleNamespace(signals=sig, values=devs, freqs=freqs)


def _make_session(md: MetaDict | None = None, force_magnitude: bool = True):
    fig = Figure()
    FigureCanvasAgg(fig)
    host = _FakeHost(fig)
    req = AnalyzeRequest(
        run_result=_run_result(),
        analyze_params=FluxPickParams(force_magnitude=force_magnitude),
        md=md if md is not None else MetaDict(),
        ml=ModuleLibrary(),
        predictor=None,
    )
    session = build_flux_pick_session(req, host, force_magnitude=force_magnitude)
    return session, host


def test_actions_are_auto_align_and_swap():
    session, _ = _make_session()
    assert session.actions() == [("auto_align", "Auto Align"), ("swap", "Swap Lines")]


def test_pointer_events_repaint():
    session, host = _make_session()
    before = host.redraws
    session.on_press(0.0)
    session.on_move(1.0)
    session.on_release(1.0, 4.5)
    assert host.redraws > before


def test_swap_action_swaps_and_repaints():
    session, host = _make_session()
    half0, int0 = session.finish().flx_half, session.finish().flx_int
    r0 = host.redraws
    session.invoke_action("swap")
    res = session.finish()
    assert (res.flx_half, res.flx_int) == (int0, half0)
    assert host.redraws > r0


def test_auto_align_runs_off_main_via_host():
    session, host = _make_session()
    session.invoke_action("auto_align")
    assert host.bg_calls == 1  # the heavy step went through run_background
    res = session.finish()
    assert -5.0 <= res.flx_half <= 5.0
    assert -5.0 <= res.flx_int <= 5.0


def test_unknown_action_raises():
    import pytest

    session, _ = _make_session()
    with pytest.raises(ValueError, match="unknown action"):
        session.invoke_action("nope")


def test_finish_returns_flux_pick_result_with_period_and_figure():
    session, host = _make_session()
    res = session.finish()
    assert isinstance(res, FluxPickResult)
    assert res.flx_period == 2 * abs(res.flx_int - res.flx_half)
    assert res.figure is host.figure
    # to_summary_dict drops the Figure, keeps the scalars the agent reads
    summary = res.to_summary_dict()
    assert set(summary) == {"flx_half", "flx_int", "flx_period"}


def test_seeds_from_metadict():
    md = MetaDict()
    md.flx_half = 0.0
    md.flx_int = 2.0
    session, _ = _make_session(md=md)
    res = session.finish()
    # seeded near the supplied half/int (folded toward the spectrum centre)
    assert -5.0 <= res.flx_half <= 5.0
    assert -5.0 <= res.flx_int <= 5.0
