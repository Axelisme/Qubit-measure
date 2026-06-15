"""Tests for the passive, toolkit-agnostic TwoLinePicker core (headless, no Qt).

The core only needs a matplotlib Figure with a (headless Agg) canvas — it never
imports Qt and only repaints through matplotlib's own backend-agnostic
timer/draw_idle — so it is driven here by feeding x/y coordinates straight to its
mouse handlers and calling its toolbar-action methods, then asserting on the
mutated state (positions, loss-view axes).
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from zcu_tools.notebook.analysis.fluxdep.interactive.two_line_picker import (
    TwoLinePicker,
    find_best_mirror_position,
    fold_initial_lines,
)


def _spectrum(n_dev: int = 60, n_freq: int = 30):
    devs = np.linspace(-5.0, 5.0, n_dev).astype(np.float64)
    freqs = np.linspace(4.0, 5.0, n_freq).astype(np.float64)
    # a symmetric feature about dev=0 so mirror-loss has a clear minimum there
    sig = np.zeros((n_dev, n_freq), dtype=np.complex128)
    sig += np.exp(-(devs[:, None] ** 2) / (2 * 1.0**2))
    return sig, devs, freqs


def _make_picker(**kwargs) -> TwoLinePicker:
    sig, devs, freqs = _spectrum()
    fig = Figure()
    FigureCanvasAgg(fig)  # headless renderer so tight_layout works
    return TwoLinePicker(fig, sig, devs, freqs, **kwargs)


# --- pure helpers (re-exported from the core) ------------------------------


def test_fold_initial_lines_defaults():
    _sig, devs, _freqs = _spectrum()
    half, integer = fold_initial_lines(devs, None, None)
    assert devs[0] <= half <= devs[-1]
    assert devs[0] <= integer <= devs[-1]
    assert isinstance(half, float) and isinstance(integer, float)


def test_find_best_mirror_position_in_range():
    sig, devs, _freqs = _spectrum()
    pos = find_best_mirror_position(
        devs, np.abs(sig), current_pos=1.0, search_width=2.0
    )
    assert devs[0] <= pos <= devs[-1]


def test_find_best_mirror_position_perfect_symmetry_returns_zero_loss():
    # Regression for: when the search center is the true symmetry axis, every
    # in-bounds diff_mirror value is exactly 0 (perfect symmetry).  The old
    # code filtered ``diff_amps[diff_amps != 0.0]`` which then had length 0 and
    # triggered ``continue``, skipping the best candidate.  The fixed code uses
    # the in-bounds mask instead and correctly yields loss = 0.0 (not NaN) so
    # the symmetric center wins.
    n = 51  # odd -> dev[25] is the exact center
    devs = np.linspace(-5.0, 5.0, n, dtype=np.float64)
    center = float(devs[n // 2])  # exact grid symmetry axis: 0.0

    # Construct a perfectly left–right symmetric 1-D signal (shape (n,1) so
    # diff_mirror shape matches real_signals expectations).
    col = np.abs(devs).reshape(n, 1).astype(np.float64)  # symmetric about 0

    # Run with a search window that includes the true center.
    pos = find_best_mirror_position(devs, col, current_pos=center, search_width=1.0)

    # The symmetric center must be chosen (loss = 0.0, strictly less than any
    # asymmetric candidate).
    assert pos == pytest.approx(center, abs=0.5 * (devs[1] - devs[0]))


# --- core construction + queries -------------------------------------------


def test_construct_reports_positions():
    picker = _make_picker()
    half, integer = picker.positions()
    assert isinstance(half, float) and isinstance(integer, float)
    assert "flux period" in picker.info_text()
    assert picker.period() == 2 * abs(integer - half)


# --- drag interaction ------------------------------------------------------


def test_press_then_move_drags_the_nearest_line():
    picker = _make_picker()
    half0, _int0 = picker.positions()
    picker.on_press(half0)
    target = half0 + 1.0
    picker.on_move(target)
    half1, _int1 = picker.positions()
    assert abs(half1 - target) < 1e-6
    assert half1 != half0


def test_press_far_from_both_picks_nothing():
    picker = _make_picker()
    half0, int0 = picker.positions()
    picker.on_press(1e6)  # nowhere near either line
    picker.on_move(0.0)  # ignored (nothing picked)
    assert picker.positions() == (half0, int0)


def test_conjugate_drag_moves_both_lines_together():
    picker = _make_picker()
    half0, int0 = picker.positions()
    picker.set_conjugate(True)
    picker.on_press(half0)
    picker.on_move(half0 + 0.5)
    half1, int1 = picker.positions()
    # both shift by the same delta -> their gap (period) is preserved
    assert abs((half1 - half0) - (int1 - int0)) < 1e-6


def test_min_distance_clamp_keeps_lines_apart():
    picker = _make_picker()
    half0, int0 = picker.positions()
    # Drag the half line to just left of the int line (inside the min gap); the
    # clamp must push it back so the two stay at least min_flux_dist apart.
    picker.on_press(half0)
    picker.on_move(int0 - 0.001)
    half1, int1 = picker.positions()
    assert abs(half1 - int1) >= picker._min_flux_dist - 1e-9


def test_swap_exchanges_positions():
    picker = _make_picker()
    half0, int0 = picker.positions()
    picker.swap()
    half1, int1 = picker.positions()
    assert (half1, int1) == (int0, half0)


# --- live mirror-loss refresh (picker-owned throttle timer) ----------------


def test_loss_timer_refreshes_view_at_picked_line():
    # Dragging only moves the line; the throttle timer is what recomputes the
    # mirror-loss view at the line's LATEST position. Fire it directly: the loss
    # subplot's x-zoom must re-centre on the dragged line.
    picker = _make_picker()
    half0, _int0 = picker.positions()
    picker.on_press(half0)  # grab the half line
    target = half0 + 1.5
    picker.on_move(target)  # line follows; loss not refreshed yet
    picker._on_loss_timer()
    lo, hi = picker._ax_loss.get_xlim()
    assert abs((lo + hi) / 2 - target) < 1e-6


def test_loss_refresh_is_noop_without_a_picked_line():
    picker = _make_picker()
    # Nothing picked -> scheduling does not arm, and firing is a safe no-op.
    picker._schedule_loss_refresh()
    assert picker._loss_refresh_pending is False
    picker._on_loss_timer()  # must not raise


# --- heavy action: compute (pure) + apply (main-thread mutate) -------------


def test_compute_aligned_positions_is_pure_and_in_range():
    picker = _make_picker()
    before = picker.positions()
    half, integer = picker.compute_aligned_positions()
    # compute returns candidate positions WITHOUT mutating the picker
    assert picker.positions() == before
    assert -5.0 <= half <= 5.0
    assert -5.0 <= integer <= 5.0


def test_apply_positions_sets_both_lines():
    picker = _make_picker()
    picker.apply_positions(1.25, -2.5)
    assert picker.positions() == (1.25, -2.5)


def test_auto_align_keeps_positions_in_range():
    picker = _make_picker()
    picker.auto_align()
    half, integer = picker.positions()
    assert -5.0 <= half <= 5.0
    assert -5.0 <= integer <= 5.0


# --- magnitude toggle ------------------------------------------------------


def test_magnitude_only_toggle():
    picker = _make_picker()
    assert picker.magnitude_only is False
    picker.set_magnitude_only(True)
    assert picker.magnitude_only is True


def test_force_magnitude_starts_on():
    picker = _make_picker(force_magnitude=True)
    assert picker.magnitude_only is True
