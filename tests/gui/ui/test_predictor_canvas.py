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
# Headless canvas smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def canvas(qapp):
    from zcu_tools.gui.session.ui.predictor_canvas import PredictorCurveCanvas

    return PredictorCurveCanvas(figsize=(4.0, 3.0))


def _make_curve_result(n: int = 20):
    from zcu_tools.gui.session.services.connection import PredictCurveResult

    values = np.linspace(0.0, 1.0, n)
    transitions = ((0, 1), (0, 2), (0, 3), (0, 4))
    freqs = np.random.default_rng(42).uniform(100, 5000, size=(len(transitions), n))
    return PredictCurveResult(
        labels=("0→1", "0→2", "0→3", "0→4"),
        values=values,
        fluxs=values.copy(),
        freqs_mhz=freqs,
    )


def test_canvas_render_curves_no_error(canvas):
    """render_curves must not raise on valid inputs."""
    result = _make_curve_result()
    canvas.render_curves(
        result,
        highlight=(0, 1),
        marker_value=0.5,
        flux_window=(0.4, 1.1),
        value_to_flux=_value_to_flux,
        flux_to_value=_flux_to_value,
    )
    # Canvas should have axes after rendering.
    assert len(canvas.figure.get_axes()) > 0


def test_canvas_render_creates_four_curves(canvas):
    """render_curves must draw exactly four data lines (one per transition)."""
    result = _make_curve_result()
    canvas.render_curves(
        result,
        highlight=(0, 1),
        marker_value=0.5,
        flux_window=(0.4, 1.1),
        value_to_flux=_value_to_flux,
        flux_to_value=_flux_to_value,
    )
    # The primary axes has exactly four regular lines (excluding the axvline marker,
    # which is also a Line2D but labelled "_marker").
    ax = canvas._get_ax()
    assert ax is not None
    data_lines = [l for l in ax.lines if not l.get_label().startswith("_")]
    assert len(data_lines) == 4


def test_canvas_set_marker_no_error(canvas):
    """set_marker must not raise even before render_curves."""
    canvas.set_marker(0.5)  # no-op — no axes yet


def test_canvas_set_marker_after_render(canvas):
    """set_marker after render_curves moves the marker line."""
    result = _make_curve_result()
    canvas.render_curves(
        result,
        highlight=(0, 1),
        marker_value=0.5,
        flux_window=(0.4, 1.1),
        value_to_flux=_value_to_flux,
        flux_to_value=_flux_to_value,
    )
    canvas.set_marker(0.3)
    assert canvas._marker_value == pytest.approx(0.3)


def test_canvas_clear_no_error(canvas):
    """clear must not raise and must remove the marker state."""
    result = _make_curve_result()
    canvas.render_curves(
        result,
        highlight=(0, 1),
        marker_value=0.5,
        flux_window=(0.4, 1.1),
        value_to_flux=_value_to_flux,
        flux_to_value=_flux_to_value,
    )
    canvas.clear()
    assert canvas._marker_line is None
    assert canvas._marker_value is None


def test_canvas_drag_callback_fires(canvas):
    """on_drag callback is called when _on_move fires while dragging."""
    result = _make_curve_result()
    canvas.render_curves(
        result,
        highlight=(0, 1),
        marker_value=0.5,
        flux_window=(0.4, 1.1),
        value_to_flux=_value_to_flux,
        flux_to_value=_flux_to_value,
    )

    received: list[float] = []
    canvas.bind_callbacks(on_drag=received.append, on_drop=lambda _v: None)

    # Simulate a press + move + release cycle directly via the internal methods.
    # Build a minimal mock event.
    ax = canvas._get_ax()
    assert ax is not None

    class FakeEvent:
        def __init__(self, x: float) -> None:
            self.inaxes = ax
            self.xdata = x

    canvas._on_press(FakeEvent(0.5))  # grab the marker (within tol)
    canvas._on_move(FakeEvent(0.6))  # drag to 0.6
    canvas._on_release(FakeEvent(0.6))  # release

    # The drag callback should have fired with ~0.6.
    assert len(received) >= 1
    assert received[-1] == pytest.approx(0.6)


def test_canvas_drop_callback_fires(canvas):
    """on_drop callback is called on release after an actual drag."""
    result = _make_curve_result()
    canvas.render_curves(
        result,
        highlight=(0, 1),
        marker_value=0.5,
        flux_window=(0.4, 1.1),
        value_to_flux=_value_to_flux,
        flux_to_value=_flux_to_value,
    )

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
