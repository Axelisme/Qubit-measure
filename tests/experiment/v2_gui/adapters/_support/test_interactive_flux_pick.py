"""Tests for the shared interactive flux-pick session (no Qt, no host widget).

A fake InteractiveHost (a real matplotlib Figure + a synchronous run_background)
stands in for the GUI; the session is driven through its InteractiveSession
methods and its FluxPickResult is asserted.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from zcu_tools.experiment.v2_gui.adapters._support import (
    FluxPickParams,
    FluxPickResult,
    build_flux_pick_session,
)
from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.app.main.adapter.types import ButtonControl, ToggleControl
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


class _FakeHost:
    """Minimal InteractiveHost: a real Figure + counters; run_background runs the
    compute synchronously so the test can assert the applied result."""

    def __init__(self, figure: Figure) -> None:
        self.figure = figure
        self.redraws = 0
        self.bg_calls = 0
        self._pending_on_done: list[tuple[object, object]] = []

    def redraw(self) -> None:
        self.redraws += 1

    def run_background(
        self, compute: Callable[[], object], on_done: Callable[[object], None]
    ) -> None:
        self.bg_calls += 1
        # Run synchronously for most tests
        on_done(compute())

    def run_background_deferred(
        self, compute: Callable[[], object], on_done: Callable[[object], None]
    ) -> None:
        """Capture for late-completion test: don't call on_done immediately."""
        self.bg_calls += 1
        self._pending_on_done.append((compute, on_done))

    def flush_pending(self) -> None:
        for compute, on_done in list(self._pending_on_done):
            on_done(compute())
        self._pending_on_done.clear()


def _run_result(n_dev: int = 60, n_freq: int = 30):
    devs = np.linspace(-5.0, 5.0, n_dev).astype(np.float64)
    freqs = np.linspace(4.0, 5.0, n_freq).astype(np.float64)
    sig = np.zeros((n_dev, n_freq), dtype=np.complex128)
    sig += np.exp(-(devs[:, None] ** 2) / (2 * 1.0**2))
    return SimpleNamespace(signals=sig, values=devs, freqs=freqs)


def _make_session(md: MetaDict | None = None, force_magnitude: bool = True, host=None):
    fig = Figure()
    FigureCanvasAgg(fig)
    if host is None:
        host = _FakeHost(fig)
    else:
        host.figure = fig
    req = AnalyzeRequest(
        run_result=_run_result(),
        analyze_params=FluxPickParams(),
        md=md if md is not None else MetaDict(),
        ml=ModuleLibrary(),
        predictor=None,
    )
    session = build_flux_pick_session(req, host, force_magnitude=force_magnitude)
    return session, host


def test_controls_are_conjugate_auto_align_swap_in_order():
    session, _ = _make_session()
    controls = session.controls()
    assert isinstance(controls, tuple)
    assert len(controls) == 3
    assert isinstance(controls[0], ToggleControl)
    assert controls[0].label == "Conjugate Line"
    assert controls[0].initial is False
    assert isinstance(controls[1], ButtonControl)
    assert controls[1].label == "Auto Align"
    assert isinstance(controls[2], ButtonControl)
    assert controls[2].label == "Swap Lines"
    # Stable keys unique and non-empty
    keys = [c.key for c in controls]
    assert len(keys) == len(set(keys))
    assert all(k for k in keys)


def test_controls_are_ordered_and_immutable():
    session, _ = _make_session()
    c1 = session.controls()
    c2 = session.controls()
    # Should be equal tuple each call; not required to be same object but ordered
    assert c1 == c2
    assert isinstance(c1, tuple)


def test_conjugate_initial_is_false_and_no_redraw():
    session, host = _make_session()
    toggle = session.controls()[0]
    assert isinstance(toggle, ToggleControl)
    assert toggle.initial is False
    # Initialization must not have triggered a callback or redraw
    # (host.redraws is 0 after construction)
    before = host.redraws
    # Invoke toggle on_change directly — it should not redraw
    toggle.on_change(True)
    assert host.redraws == before


def test_conjugate_toggle_enables_coupled_drag():
    session, _ = _make_session()
    controls = session.controls()
    toggle = controls[0]
    assert isinstance(toggle, ToggleControl)
    # Initially not conjugate: dragging one line moves only that line
    half0, int0 = session._picker.positions()  # type: ignore[attr-defined]
    session.on_press(half0)
    session.on_move(half0 + 0.6)
    half1, int1 = session._picker.positions()  # type: ignore[attr-defined]
    assert abs((half1 - half0) - 0.6) < 1e-9
    assert abs(int1 - int0) < 1e-9
    session.on_release(half1, 4.5)
    # picker stays picked after release; clear before next interaction
    session._picker.clear_selection()  # type: ignore[attr-defined]
    # Enable conjugate and drag again — both move equally
    toggle.on_change(True)
    half_cur, int_cur = session._picker.positions()  # type: ignore[attr-defined]
    session.on_press(half_cur)
    dx = 0.4
    session.on_move(half_cur + dx)
    half2, int2 = session._picker.positions()  # type: ignore[attr-defined]
    assert abs((half2 - half_cur) - dx) < 1e-9
    assert abs((int2 - int_cur) - dx) < 1e-9
    session.on_release(half2, 4.5)
    session._picker.clear_selection()  # type: ignore[attr-defined]
    # Disable conjugate again
    toggle.on_change(False)
    half_cur2, int_cur2 = session._picker.positions()  # type: ignore[attr-defined]
    session.on_press(half_cur2)
    session.on_move(half_cur2 + 0.3)
    half3, int3 = session._picker.positions()  # type: ignore[attr-defined]
    assert abs((half3 - half_cur2) - 0.3) < 1e-9
    assert abs(int3 - int_cur2) < 1e-9


def test_pointer_events_repaint():
    session, host = _make_session()
    before = host.redraws
    session.on_press(0.0)
    session.on_move(1.0)
    session.on_release(1.0, 4.5)
    assert host.redraws > before


def test_swap_control_swaps_and_repaints():
    session, host = _make_session()
    # Need two finish calls? But finish caches; so capture positions before swap without finishing
    half0, int0 = session._picker.positions()  # type: ignore[attr-defined]
    r0 = host.redraws
    controls = session.controls()
    swap = controls[2]
    assert isinstance(swap, ButtonControl)
    swap.on_trigger()
    half1, int1 = session._picker.positions()  # type: ignore[attr-defined]
    assert (half1, int1) == (int0, half0)
    assert host.redraws > r0


def test_auto_align_runs_off_main_via_host():
    session, host = _make_session()
    controls = session.controls()
    auto = controls[1]
    assert isinstance(auto, ButtonControl)
    auto.on_trigger()
    assert host.bg_calls == 1  # the heavy step went through run_background
    res = session.finish()
    assert -5.0 <= res.flx_half <= 5.0
    assert -5.0 <= res.flx_int <= 5.0


def test_auto_align_applies_positions_and_repaints():
    # Use deferred host to verify that the background result is applied
    fig = Figure()
    FigureCanvasAgg(fig)
    host = _FakeHost(fig)
    # Replace run_background with deferred for this session after creation?
    # We'll patch after creation: session's host is the same object
    session, _ = _make_session(host=host)
    controls = session.controls()
    auto = controls[1]
    assert isinstance(auto, ButtonControl)
    # Capture positions before
    half0, int0 = session._picker.positions()  # type: ignore[attr-defined]
    host.run_background = host.run_background_deferred  # type: ignore[method-assign]
    auto.on_trigger()
    assert host.bg_calls == 1
    # Not yet applied
    half_mid, int_mid = session._picker.positions()  # type: ignore[attr-defined]
    assert (half_mid, int_mid) == (half0, int0)
    host.flush_pending()
    half1, int1 = session._picker.positions()  # type: ignore[attr-defined]
    # After flush should have moved (auto align does a search, but at least in range)
    assert -5.0 <= half1 <= 5.0
    assert -5.0 <= int1 <= 5.0
    assert host.redraws > 0


def test_finish_returns_flux_pick_result_with_period_and_figure():
    session, host = _make_session()
    res = session.finish()
    assert isinstance(res, FluxPickResult)
    assert res.flx_period == 2 * abs(res.flx_int - res.flx_half)
    assert res.figure is host.figure
    # to_summary_dict drops the Figure, keeps the scalars the agent reads
    summary = res.to_summary_dict()
    assert set(summary) == {"flx_half", "flx_int", "flx_period"}


def test_finish_caches_and_ignores_subsequent_input():
    session, host = _make_session()
    half0, int0 = session._picker.positions()  # type: ignore[attr-defined]
    res1 = session.finish()
    # After finish, further pointer input must not modify picker
    session.on_press(half0)
    session.on_move(half0 + 1.0)
    half1, int1 = session._picker.positions()  # type: ignore[attr-defined]
    assert (half1, int1) == (half0, int0)
    res2 = session.finish()
    assert res2 is res1  # cached


def test_late_background_completion_ignored_after_finish():
    fig = Figure()
    FigureCanvasAgg(fig)
    host = _FakeHost(fig)
    session, _ = _make_session(host=host)
    # Make auto align deferred
    host.run_background = host.run_background_deferred  # type: ignore[method-assign]
    controls = session.controls()
    auto = controls[1]
    assert isinstance(auto, ButtonControl)
    auto.on_trigger()
    # Now finish before background completes
    res_before = session.finish()
    half_before, int_before = res_before.flx_half, res_before.flx_int
    # Flush late completion — should be ignored
    host.flush_pending()
    res_after = session.finish()
    # Picker positions must not have changed, result cached
    assert res_after is res_before
    assert res_after.flx_half == half_before
    assert res_after.flx_int == int_before
    half_cur, int_cur = session._picker.positions()  # type: ignore[attr-defined]
    assert half_cur == half_before
    assert int_cur == int_before


def test_seeds_from_metadict():
    md = MetaDict()
    md.flx_half = 0.0
    md.flx_int = 2.0
    session, _ = _make_session(md=md)
    res = session.finish()
    # seeded near the supplied half/int (folded toward the spectrum centre)
    assert -5.0 <= res.flx_half <= 5.0
    assert -5.0 <= res.flx_int <= 5.0


def test_both_force_magnitude_variants_share_same_control_surface():
    s1, _ = _make_session(force_magnitude=True)
    s2, _ = _make_session(force_magnitude=False)
    c1 = s1.controls()
    c2 = s2.controls()
    assert [type(c).__name__ for c in c1] == [type(c).__name__ for c in c2]
    assert [c.label for c in c1] == [c.label for c in c2]
    assert [c.key for c in c1] == [c.key for c in c2]
