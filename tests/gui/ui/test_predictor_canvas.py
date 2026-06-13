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


def test_canvas_drag_callback_fires(canvas):
    """on_drag callback is called when _on_move fires while dragging."""
    canvas.render_curves(**_make_render_kwargs())

    received: list[float] = []
    canvas.bind_callbacks(on_drag=received.append, on_drop=lambda _v: None)

    ax = canvas._get_ax()
    assert ax is not None

    class FakeEvent:
        def __init__(self, x: float) -> None:
            self.inaxes = ax
            self.xdata = x

    canvas._on_press(FakeEvent(0.5))  # grab the marker (within tol)
    canvas._on_move(FakeEvent(0.6))  # drag to 0.6
    canvas._on_release(FakeEvent(0.6))  # release

    assert len(received) >= 1
    assert received[-1] == pytest.approx(0.6)


def test_canvas_drop_callback_fires(canvas):
    """on_drop callback is called on release after an actual drag."""
    canvas.render_curves(**_make_render_kwargs())

    drops: list[float] = []
    canvas.bind_callbacks(on_drag=lambda _v: None, on_drop=drops.append)

    ax = canvas._get_ax()
    assert ax is not None

    class FakeEvent:
        def __init__(self, x: float) -> None:
            self.inaxes = ax
            self.xdata = x

    canvas._on_press(FakeEvent(0.5))
    canvas._on_move(FakeEvent(0.7))
    canvas._on_release(FakeEvent(0.7))

    assert len(drops) == 1
    assert drops[0] == pytest.approx(0.7)


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
