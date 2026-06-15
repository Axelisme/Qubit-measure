"""Tests for PredictorCurveCanvas — window-shift logic and marker interaction."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.session.ui.predictor_canvas import _compute_xlim

# ---------------------------------------------------------------------------
# Pure-function: _compute_xlim window-shift logic
# ---------------------------------------------------------------------------


def _value_to_flux(v: float) -> float:
    """Simple identity-affine for tests: flux = value."""
    return v


def _flux_to_value(f: float) -> float:
    """Inverse of _value_to_flux."""
    return f


def test_compute_xlim_marker_inside_window():
    """When marker is inside the flux window, xlim maps to the window directly."""
    # flux_window [0.4, 1.1] → value [0.4, 1.1] (identity affine)
    x_lo, x_hi = _compute_xlim((0.4, 1.1), 0.7, _flux_to_value)
    assert x_lo == pytest.approx(0.4)
    assert x_hi == pytest.approx(1.1)


def test_compute_xlim_marker_below_window():
    """When marker is below the window, shift left (window width preserved)."""
    # Window width = 0.7; marker at 0.2 → should be included at left edge.
    width = 1.1 - 0.4
    x_lo, x_hi = _compute_xlim((0.4, 1.1), 0.2, _flux_to_value)
    assert x_lo == pytest.approx(0.2)
    assert (x_hi - x_lo) == pytest.approx(width)


def test_compute_xlim_marker_above_window():
    """When marker is above the window, shift right (window width preserved)."""
    width = 1.1 - 0.4
    x_lo, x_hi = _compute_xlim((0.4, 1.1), 1.5, _flux_to_value)
    assert x_hi == pytest.approx(1.5)
    assert (x_hi - x_lo) == pytest.approx(width)


def test_compute_xlim_marker_exactly_at_boundary():
    """Marker exactly at the lower boundary is still considered inside."""
    x_lo, x_hi = _compute_xlim((0.4, 1.1), 0.4, _flux_to_value)
    assert x_lo == pytest.approx(0.4)
    assert x_hi == pytest.approx(1.1)


def test_compute_xlim_window_width_preserved_far_outside():
    """Window width stays constant regardless of how far outside the marker is."""
    width = 1.1 - 0.4
    x_lo, x_hi = _compute_xlim((0.4, 1.1), 5.0, _flux_to_value)
    assert (x_hi - x_lo) == pytest.approx(width)
    assert x_hi == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Headless canvas smoke tests (decoupled render_curves API)
# ---------------------------------------------------------------------------


@pytest.fixture()
def canvas(qapp):
    from zcu_tools.gui.session.ui.predictor_canvas import PredictorCurveCanvas

    return PredictorCurveCanvas(figsize=(4.0, 3.0))


def _make_render_kwargs(n_transitions: int = 4, n: int = 20) -> dict:
    """Build keyword args for the decoupled render_curves API."""
    values = np.linspace(0.0, 1.0, n)
    labels = tuple(f"0→{i + 1}" for i in range(n_transitions))
    series = np.random.default_rng(42).uniform(100, 5000, size=(n_transitions, n))
    return dict(
        values=values,
        labels=labels,
        series=series,
        ylabel="Frequency (MHz)",
        highlight=(0, 1),
        marker_value=0.5,
        flux_window=(0.4, 1.1),
        value_to_flux=_value_to_flux,
        flux_to_value=_flux_to_value,
    )


def test_canvas_render_curves_no_error(canvas):
    """render_curves must not raise on valid inputs."""
    canvas.render_curves(**_make_render_kwargs())
    assert len(canvas.figure.get_axes()) > 0


def test_canvas_primary_xaxis_label_device_value_no_unit(canvas):
    """The primary x-axis is labelled 'Device value' with no Ampere/'(A)' unit."""
    canvas.render_curves(**_make_render_kwargs())
    ax = canvas._get_ax()
    assert ax is not None
    assert ax.get_xlabel() == "Device value"


def test_canvas_render_creates_four_curves(canvas):
    """render_curves must draw exactly four data lines (one per transition)."""
    canvas.render_curves(**_make_render_kwargs(n_transitions=4))
    ax = canvas._get_ax()
    assert ax is not None
    data_lines = [ln for ln in ax.lines if not ln.get_label().startswith("_")]
    assert len(data_lines) == 4


def test_canvas_set_marker_no_error(canvas):
    """set_marker must not raise even before render_curves."""
    canvas.set_marker(0.5)  # no-op — no axes yet


def test_canvas_set_marker_after_render(canvas):
    """set_marker after render_curves moves the marker line."""
    canvas.render_curves(**_make_render_kwargs())
    canvas.set_marker(0.3)
    assert canvas._marker_value == pytest.approx(0.3)


def test_canvas_clear_no_error(canvas):
    """clear must not raise and must remove the marker state."""
    canvas.render_curves(**_make_render_kwargs())
    canvas.clear()
    assert canvas._marker_line is None
    assert canvas._marker_value is None


class _FakeAxesEvent:
    """Lightweight in-axes event for driving the canvas handlers headlessly."""

    def __init__(self, ax, x: float) -> None:
        self.inaxes = ax
        self.xdata = x


def test_canvas_follow_engages_on_first_click(canvas):
    """A first click near the marker engages follow mode (no lock callback yet)."""
    canvas.render_curves(**_make_render_kwargs())
    ax = canvas._get_ax()
    assert ax is not None

    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)

    canvas._on_press(_FakeAxesEvent(ax, 0.5))  # click on the marker → engage
    assert canvas._following is True
    assert locks == []  # engaging click does not lock


def test_canvas_follow_callback_fires_on_motion_without_button(canvas):
    """While following, motion (button NOT held) fires on_follow with the cursor x."""
    canvas.render_curves(**_make_render_kwargs())
    ax = canvas._get_ax()
    assert ax is not None

    received: list[float] = []
    canvas.bind_callbacks(on_follow=received.append, on_lock=lambda _v: None)

    canvas._on_press(_FakeAxesEvent(ax, 0.5))  # engage follow
    canvas._on_move(_FakeAxesEvent(ax, 0.6))  # cursor moves while following
    canvas._on_move(_FakeAxesEvent(ax, 0.65))

    assert received == pytest.approx([0.6, 0.65])


def test_canvas_no_follow_motion_before_engage(canvas):
    """Motion without an engaging click must NOT fire on_follow."""
    canvas.render_curves(**_make_render_kwargs())
    ax = canvas._get_ax()
    assert ax is not None

    received: list[float] = []
    canvas.bind_callbacks(on_follow=received.append, on_lock=lambda _v: None)

    canvas._on_move(_FakeAxesEvent(ax, 0.6))  # not following yet
    assert received == []


def test_canvas_second_click_locks_and_disengages(canvas):
    """A second click disengages follow and fires on_lock at the current marker."""
    canvas.render_curves(**_make_render_kwargs())
    ax = canvas._get_ax()
    assert ax is not None

    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)

    canvas._on_press(_FakeAxesEvent(ax, 0.5))  # engage
    canvas._on_move(_FakeAxesEvent(ax, 0.7))  # follow → marker at 0.7
    canvas._on_press(_FakeAxesEvent(ax, 0.7))  # second click → lock + disengage

    assert canvas._following is False
    assert len(locks) == 1
    assert locks[0] == pytest.approx(0.7)


def test_canvas_motion_after_lock_does_not_follow(canvas):
    """After locking, further motion must NOT fire on_follow."""
    canvas.render_curves(**_make_render_kwargs())
    ax = canvas._get_ax()
    assert ax is not None

    received: list[float] = []
    canvas.bind_callbacks(on_follow=received.append, on_lock=lambda _v: None)

    canvas._on_press(_FakeAxesEvent(ax, 0.5))  # engage
    canvas._on_move(_FakeAxesEvent(ax, 0.7))  # follow
    canvas._on_press(_FakeAxesEvent(ax, 0.7))  # lock
    received.clear()
    canvas._on_move(_FakeAxesEvent(ax, 0.9))  # motion after lock → ignored

    assert received == []


# ---------------------------------------------------------------------------
# Auto-untrack: leaving the axes while following disengages without a click
# ---------------------------------------------------------------------------


class _FakeLeaveEvent:
    """Lightweight leave event (no xdata / inaxes) for driving _on_axes_leave."""

    # axes_leave_event / figure_leave_event carry no meaningful xdata.
    pass


def test_canvas_leave_event_while_following_disengages(canvas):
    """axes_leave_event while following disengages and fires on_lock at last position."""
    canvas.render_curves(**_make_render_kwargs())
    ax = canvas._get_ax()
    assert ax is not None

    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)

    canvas._on_press(_FakeAxesEvent(ax, 0.5))  # engage follow
    canvas._on_move(_FakeAxesEvent(ax, 0.65))  # move → marker at 0.65
    canvas._on_axes_leave(_FakeLeaveEvent())  # cursor leaves axes

    assert canvas._following is False
    assert len(locks) == 1
    assert locks[0] == pytest.approx(0.65)  # locked at last in-range position


def test_canvas_out_of_range_motion_while_following_disengages(canvas):
    """motion_notify_event with inaxes=None while following auto-untracks."""
    canvas.render_curves(**_make_render_kwargs())
    ax = canvas._get_ax()
    assert ax is not None

    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)

    canvas._on_press(_FakeAxesEvent(ax, 0.5))  # engage follow
    canvas._on_move(_FakeAxesEvent(ax, 0.7))  # follow → marker at 0.7

    # Out-of-range motion: inaxes is None, xdata is None.
    class _OutOfRangeEvent:
        inaxes = None
        xdata = None

    canvas._on_move(_OutOfRangeEvent())  # backstop path

    assert canvas._following is False
    assert len(locks) == 1
    assert locks[0] == pytest.approx(0.7)


def test_canvas_motion_after_leave_disengage_does_not_follow(canvas):
    """After auto-untrack via leave, in-range motion must NOT move marker or call on_follow."""
    canvas.render_curves(**_make_render_kwargs())
    ax = canvas._get_ax()
    assert ax is not None

    received: list[float] = []
    canvas.bind_callbacks(on_follow=received.append, on_lock=lambda _v: None)

    canvas._on_press(_FakeAxesEvent(ax, 0.5))  # engage
    canvas._on_move(_FakeAxesEvent(ax, 0.6))  # follow
    canvas._on_axes_leave(_FakeLeaveEvent())  # leave → disengage
    received.clear()

    canvas._on_move(_FakeAxesEvent(ax, 0.8))  # in-range motion after disengage
    assert received == []
    assert canvas._marker_value == pytest.approx(0.6)  # marker stays at 0.6


def test_canvas_leave_event_when_not_following_is_noop(canvas):
    """axes_leave_event while NOT following must not fire on_lock."""
    canvas.render_curves(**_make_render_kwargs())

    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)

    canvas._on_axes_leave(_FakeLeaveEvent())  # not following — no-op
    assert locks == []


# ---------------------------------------------------------------------------
# highlight=None renders all curves in normal style
# ---------------------------------------------------------------------------


def test_canvas_render_curves_highlight_none_no_error(canvas):
    """render_curves with highlight=None must not raise."""
    kwargs = _make_render_kwargs()
    kwargs["highlight"] = None
    canvas.render_curves(**kwargs)
    assert len(canvas.figure.get_axes()) > 0


def test_canvas_render_curves_highlight_none_all_normal_lw(canvas):
    """With highlight=None, all curve lines use the normal (non-highlighted) linewidth."""
    from zcu_tools.gui.session.ui.predictor_canvas import _NORMAL_LW

    kwargs = _make_render_kwargs()
    kwargs["highlight"] = None
    canvas.render_curves(**kwargs)
    for line in canvas._curve_lines.values():
        assert line.get_linewidth() == pytest.approx(_NORMAL_LW)


def test_canvas_set_highlight_none_reverts_all_to_normal(canvas):
    """set_highlight(None) reverts all previously highlighted curves to normal."""
    from zcu_tools.gui.session.ui.predictor_canvas import _NORMAL_LW

    canvas.render_curves(**_make_render_kwargs(n_transitions=4))
    canvas.set_highlight(None)
    for line in canvas._curve_lines.values():
        assert line.get_linewidth() == pytest.approx(_NORMAL_LW)


def test_canvas_render_curves_dynamic_count(canvas):
    """render_curves with varying transition counts must not raise."""
    for n_tr in (5, 6):
        values = np.linspace(0.0, 1.0, 20)
        labels = tuple(f"0→{i}" for i in range(1, n_tr + 1))
        series = np.ones((n_tr, 20), dtype=np.float64) * 1000.0
        canvas.render_curves(
            values=values,
            labels=labels,
            series=series,
            ylabel="Frequency (MHz)",
            highlight=None,
            marker_value=0.5,
            flux_window=(0.4, 1.1),
            value_to_flux=_value_to_flux,
            flux_to_value=_flux_to_value,
        )
        ax = canvas._get_ax()
        assert ax is not None
        data_lines = [ln for ln in ax.lines if not ln.get_label().startswith("_")]
        assert len(data_lines) == n_tr


def test_canvas_render_curves_custom_ylabel(canvas):
    """ylabel is applied to the y-axis label."""
    values = np.linspace(0.0, 1.0, 20)
    labels = ("0→1",)
    series = np.ones((1, 20), dtype=np.float64)
    canvas.render_curves(
        values=values,
        labels=labels,
        series=series,
        ylabel="|<i|n|j>|",
        highlight=None,
        marker_value=0.5,
        flux_window=(0.4, 1.1),
        value_to_flux=_value_to_flux,
        flux_to_value=_flux_to_value,
    )
    ax = canvas._get_ax()
    assert ax is not None
    assert ax.get_ylabel() == "|<i|n|j>|"
