"""Tests for PredictorCurveCanvas — window-shift logic and marker interaction."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backend_bases import Event, MouseButton, MouseEvent
from matplotlib.backends.backend_qt import NavigationToolbar2QT


def _value_to_flux(v: float) -> float:
    """Simple identity-affine for tests: flux = value."""
    return v


def _flux_to_value(f: float) -> float:
    """Inverse of _value_to_flux."""
    return f


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
        highlights=((0, 1),),
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
    ax = canvas.figure.axes[0]
    assert ax.get_xlabel() == "Device value"


def test_canvas_render_creates_four_curves(canvas):
    """render_curves must draw exactly four data lines (one per transition)."""
    canvas.render_curves(**_make_render_kwargs(n_transitions=4))
    ax = canvas.figure.axes[0]
    data_lines = [ln for ln in ax.lines if not ln.get_label().startswith("_")]
    assert len(data_lines) == 4


def test_canvas_set_marker_no_error(canvas):
    """set_marker must not raise even before render_curves."""
    canvas.set_marker(0.5)  # no-op — no axes yet


def test_canvas_set_marker_after_render(canvas):
    """set_marker after render_curves moves the marker line."""
    canvas.render_curves(**_make_render_kwargs())
    canvas.set_marker(0.3)
    assert canvas.figure.axes[0].lines[-1].get_xdata() == pytest.approx([0.3, 0.3])


def test_canvas_set_marker_preserves_user_zoom_width(canvas):
    """set_marker pans the current zoom window rather than resetting to flux_window."""
    canvas.render_curves(**_make_render_kwargs())
    canvas.apply_xlim((0.45, 0.55))

    canvas.set_marker(0.8)

    xlim = canvas.current_xlim()
    assert xlim is not None
    x_lo, x_hi = xlim
    assert x_hi == pytest.approx(0.8)
    assert (x_hi - x_lo) == pytest.approx(0.1)


def test_canvas_apply_xlim_notify_contract(canvas):
    """apply_xlim suppresses synced updates unless notify=True is requested."""
    canvas.render_curves(**_make_render_kwargs())
    received: list[tuple[object, tuple[float, float]]] = []
    canvas.bind_xlim_changed(lambda source, xlim: received.append((source, xlim)))

    canvas.apply_xlim((0.2, 0.3), notify=False)

    assert received == []

    canvas.apply_xlim((0.4, 0.5), notify=True)

    assert len(received) == 1
    assert received[0][0] is canvas
    assert received[0][1] == pytest.approx((0.4, 0.5))


def test_canvas_clear_no_error(canvas):
    """clear removes the plotted curves and marker."""
    canvas.render_curves(**_make_render_kwargs())
    canvas.clear()
    assert canvas.figure.get_axes() == []
    assert canvas.current_xlim() is None


def _mouse_event(canvas, name: str, x: float | None) -> None:
    """Dispatch a real Matplotlib event at a device value, or outside the axes."""
    canvas.canvas.draw()
    ax = canvas.figure.axes[0]
    pixels = (
        (-1, -1) if x is None else ax.transData.transform((x, sum(ax.get_ylim()) / 2))
    )
    button = MouseButton.LEFT if name == "button_press_event" else None
    event = MouseEvent(name, canvas.canvas, *pixels, button=button)
    canvas.canvas.callbacks.process(name, event)
    if name == "button_press_event":
        release = MouseEvent(
            "button_release_event", canvas.canvas, *pixels, button=button
        )
        canvas.canvas.callbacks.process("button_release_event", release)


def test_canvas_toolbar_mode_blocks_marker_follow(canvas):
    """Native toolbar pan/zoom should own mouse events while active."""
    canvas.render_curves(**_make_render_kwargs())
    received: list[float] = []
    canvas.bind_callbacks(on_follow=received.append, on_lock=lambda _v: None)

    toolbar = NavigationToolbar2QT(canvas.canvas, canvas)
    toolbar.pan()

    _mouse_event(canvas, "button_press_event", 0.5)
    _mouse_event(canvas, "motion_notify_event", 0.6)
    assert received == []


def test_canvas_toolbar_mode_blocks_marker_motion(canvas):
    """Active toolbar mode disengages marker follow on motion events too."""
    canvas.render_curves(**_make_render_kwargs())
    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)
    _mouse_event(canvas, "button_press_event", 0.5)

    toolbar = NavigationToolbar2QT(canvas.canvas, canvas)
    toolbar.pan()

    _mouse_event(canvas, "motion_notify_event", 0.6)

    assert locks == pytest.approx([0.5])


def test_canvas_follow_callback_fires_on_motion_without_button(canvas):
    """While following, motion (button NOT held) fires on_follow with the cursor x."""
    canvas.render_curves(**_make_render_kwargs())

    received: list[float] = []
    canvas.bind_callbacks(on_follow=received.append, on_lock=lambda _v: None)

    _mouse_event(canvas, "button_press_event", 0.5)  # engage follow
    _mouse_event(canvas, "motion_notify_event", 0.6)  # cursor moves while following
    _mouse_event(canvas, "motion_notify_event", 0.65)

    assert received == pytest.approx([0.6, 0.65])


def test_canvas_no_follow_motion_before_engage(canvas):
    """Motion without an engaging click must NOT fire on_follow."""
    canvas.render_curves(**_make_render_kwargs())

    received: list[float] = []
    canvas.bind_callbacks(on_follow=received.append, on_lock=lambda _v: None)

    _mouse_event(canvas, "motion_notify_event", 0.6)  # not following yet
    assert received == []


def test_canvas_second_click_locks_and_disengages(canvas):
    """A second click disengages follow and fires on_lock at the current marker."""
    canvas.render_curves(**_make_render_kwargs())

    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)

    _mouse_event(canvas, "button_press_event", 0.5)  # engage
    _mouse_event(canvas, "motion_notify_event", 0.7)  # follow → marker at 0.7
    _mouse_event(canvas, "button_press_event", 0.7)  # second click → lock + disengage

    assert len(locks) == 1
    assert locks[0] == pytest.approx(0.7)


def test_canvas_motion_after_lock_does_not_follow(canvas):
    """After locking, further motion must NOT fire on_follow."""
    canvas.render_curves(**_make_render_kwargs())

    received: list[float] = []
    canvas.bind_callbacks(on_follow=received.append, on_lock=lambda _v: None)

    _mouse_event(canvas, "button_press_event", 0.5)  # engage
    _mouse_event(canvas, "motion_notify_event", 0.7)  # follow
    _mouse_event(canvas, "button_press_event", 0.7)  # lock
    received.clear()
    _mouse_event(canvas, "motion_notify_event", 0.9)  # motion after lock → ignored

    assert received == []


# ---------------------------------------------------------------------------
# Auto-untrack: leaving the axes while following disengages without a click
# ---------------------------------------------------------------------------


def test_canvas_leave_event_while_following_disengages(canvas):
    """axes_leave_event while following disengages and fires on_lock at last position."""
    canvas.render_curves(**_make_render_kwargs())

    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)

    _mouse_event(canvas, "button_press_event", 0.5)  # engage follow
    _mouse_event(canvas, "motion_notify_event", 0.65)  # move → marker at 0.65
    canvas.canvas.callbacks.process(
        "axes_leave_event", Event("axes_leave_event", canvas.canvas)
    )  # cursor leaves axes

    assert len(locks) == 1
    assert locks[0] == pytest.approx(0.65)  # locked at last in-range position


def test_canvas_out_of_range_motion_while_following_disengages(canvas):
    """motion_notify_event with inaxes=None while following auto-untracks."""
    canvas.render_curves(**_make_render_kwargs())

    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)

    _mouse_event(canvas, "button_press_event", 0.5)  # engage follow
    _mouse_event(canvas, "motion_notify_event", 0.7)  # follow → marker at 0.7

    _mouse_event(canvas, "motion_notify_event", None)

    assert len(locks) == 1
    assert locks[0] == pytest.approx(0.7)


def test_canvas_motion_after_leave_disengage_does_not_follow(canvas):
    """After auto-untrack via leave, in-range motion must NOT move marker or call on_follow."""
    canvas.render_curves(**_make_render_kwargs())

    received: list[float] = []
    canvas.bind_callbacks(on_follow=received.append, on_lock=lambda _v: None)

    _mouse_event(canvas, "button_press_event", 0.5)  # engage
    _mouse_event(canvas, "motion_notify_event", 0.6)  # follow
    canvas.canvas.callbacks.process(
        "axes_leave_event", Event("axes_leave_event", canvas.canvas)
    )  # leave → disengage
    received.clear()

    _mouse_event(canvas, "motion_notify_event", 0.8)  # in-range motion after disengage
    assert received == []
    assert canvas.figure.axes[0].lines[-1].get_xdata() == pytest.approx([0.6, 0.6])


def test_canvas_leave_event_when_not_following_is_noop(canvas):
    """axes_leave_event while NOT following must not fire on_lock."""
    canvas.render_curves(**_make_render_kwargs())

    locks: list[float] = []
    canvas.bind_callbacks(on_follow=lambda _v: None, on_lock=locks.append)

    canvas.canvas.callbacks.process(
        "axes_leave_event", Event("axes_leave_event", canvas.canvas)
    )  # not following — no-op
    assert locks == []


# ---------------------------------------------------------------------------
# highlights=() renders all curves in normal style
# ---------------------------------------------------------------------------


def test_canvas_render_curves_highlights_empty_no_error(canvas):
    """render_curves with highlights=() must not raise."""
    kwargs = _make_render_kwargs()
    kwargs["highlights"] = ()
    canvas.render_curves(**kwargs)
    assert len(canvas.figure.get_axes()) > 0


def test_canvas_render_curves_highlights_empty_all_normal_lw(canvas):
    """With highlights=(), all curves use the previously unhighlighted style."""
    kwargs = _make_render_kwargs()
    canvas.render_curves(**kwargs)
    normal_width = canvas.figure.axes[0].lines[1].get_linewidth()
    kwargs["highlights"] = ()
    canvas.render_curves(**kwargs)
    assert [line.get_linewidth() for line in canvas.figure.axes[0].lines[:4]] == [
        normal_width
    ] * 4


def test_canvas_set_highlights_empty_reverts_all_to_normal(canvas):
    """set_highlights(()) reverts all previously highlighted curves to normal."""
    canvas.render_curves(**_make_render_kwargs(n_transitions=4))
    normal_width = canvas.figure.axes[0].lines[1].get_linewidth()
    canvas.set_highlights(())
    assert [line.get_linewidth() for line in canvas.figure.axes[0].lines[:4]] == [
        normal_width
    ] * 4


def test_canvas_set_highlights_marks_multiple_curves(canvas):
    """set_highlights can mark more than one rendered transition."""
    canvas.render_curves(**_make_render_kwargs(n_transitions=4))
    canvas.set_highlights(((0, 1), (0, 3)))
    lines = canvas.figure.axes[0].lines
    assert lines[0].get_linewidth() == lines[2].get_linewidth()
    assert lines[0].get_linewidth() > lines[1].get_linewidth()
    assert lines[1].get_linewidth() == lines[3].get_linewidth()


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
            highlights=(),
            marker_value=0.5,
            flux_window=(0.4, 1.1),
            value_to_flux=_value_to_flux,
            flux_to_value=_flux_to_value,
        )
        ax = canvas.figure.axes[0]
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
        highlights=(),
        marker_value=0.5,
        flux_window=(0.4, 1.1),
        value_to_flux=_value_to_flux,
        flux_to_value=_flux_to_value,
    )
    ax = canvas.figure.axes[0]
    assert ax.get_ylabel() == "|<i|n|j>|"


@pytest.mark.parametrize("leave_event", ["axes_leave_event", "figure_leave_event"])
def test_canvas_leave_events_lock_last_followed_marker(canvas, leave_event):
    """Both Matplotlib leave events lock once at the last followed device value."""
    canvas.render_curves(**_make_render_kwargs())
    canvas.canvas.draw()
    ax = canvas.figure.axes[0]
    y = sum(ax.get_ylim()) / 2
    followed: list[float] = []
    locked: list[float] = []
    canvas.bind_callbacks(on_follow=followed.append, on_lock=locked.append)

    x_pixel, y_pixel = ax.transData.transform((0.5, y))
    press = MouseEvent(
        "button_press_event", canvas.canvas, x_pixel, y_pixel, button=MouseButton.LEFT
    )
    canvas.canvas.callbacks.process("button_press_event", press)
    assert followed == []
    assert locked == []

    x_pixel, y_pixel = ax.transData.transform((0.65, y))
    move = MouseEvent("motion_notify_event", canvas.canvas, x_pixel, y_pixel)
    canvas.canvas.callbacks.process("motion_notify_event", move)
    assert followed == pytest.approx([0.65])
    assert locked == []

    canvas.canvas.callbacks.process(leave_event, Event(leave_event, canvas.canvas))
    assert locked == pytest.approx([0.65])

    x_pixel, y_pixel = ax.transData.transform((0.8, y))
    move_after_leave = MouseEvent(
        "motion_notify_event", canvas.canvas, x_pixel, y_pixel
    )
    canvas.canvas.callbacks.process("motion_notify_event", move_after_leave)
    canvas.canvas.callbacks.process(leave_event, Event(leave_event, canvas.canvas))
    assert followed == pytest.approx([0.65])
    assert locked == pytest.approx([0.65])
