"""Tests for the toolkit-agnostic TwoLinePicker core (headless, no Qt).

The core only needs a matplotlib Figure with a (headless Agg) canvas — it never
imports Qt — so it is driven here by feeding x/y coordinates straight to its
mouse handlers and calling its toolbar-action methods.
"""

from __future__ import annotations

import numpy as np
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


def _make_picker(**kwargs) -> tuple[TwoLinePicker, list[int]]:
    sig, devs, freqs = _spectrum()
    fig = Figure()
    FigureCanvasAgg(fig)  # headless renderer so tight_layout/draw work
    redraws: list[int] = []
    picker = TwoLinePicker(
        fig, sig, devs, freqs, redraw=lambda: redraws.append(1), **kwargs
    )
    return picker, redraws


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


# --- core construction + queries -------------------------------------------


def test_construct_redraws_and_reports_positions():
    picker, redraws = _make_picker()
    half, integer = picker.positions()
    assert isinstance(half, float) and isinstance(integer, float)
    assert redraws, "construction should trigger an initial redraw"
    assert "flux period" in picker.info_text()
    assert picker.period() == 2 * abs(integer - half)


# --- drag interaction ------------------------------------------------------


def test_press_then_move_drags_the_nearest_line():
    picker, _ = _make_picker()
    half0, _int0 = picker.positions()
    # Press right on the half line, then move it to a new x.
    picker.on_press(half0)
    target = half0 + 1.0
    picker.on_move(target)
    half1, _int1 = picker.positions()
    assert abs(half1 - target) < 1e-6
    assert half1 != half0


def test_press_far_from_both_picks_nothing():
    picker, _ = _make_picker()
    half0, int0 = picker.positions()
    picker.on_press(1e6)  # nowhere near either line
    picker.on_move(0.0)  # should be ignored (nothing picked)
    assert picker.positions() == (half0, int0)


def test_conjugate_drag_moves_both_lines_together():
    picker, _ = _make_picker()
    half0, int0 = picker.positions()
    picker.set_conjugate(True)
    picker.on_press(half0)
    picker.on_move(half0 + 0.5)
    half1, int1 = picker.positions()
    # both shift by the same delta -> their gap (period) is preserved
    assert abs((half1 - half0) - (int1 - int0)) < 1e-6


def test_min_distance_clamp_keeps_lines_apart():
    picker, _ = _make_picker()
    half0, int0 = picker.positions()
    # Drag the half line to just left of the int line (inside the min gap); the
    # clamp must push it back so the two stay at least min_flux_dist apart.
    picker.on_press(half0)
    picker.on_move(int0 - 0.001)
    half1, int1 = picker.positions()
    assert abs(half1 - int1) >= picker._min_flux_dist - 1e-9


def test_swap_exchanges_positions():
    picker, _ = _make_picker()
    half0, int0 = picker.positions()
    picker.swap()
    half1, int1 = picker.positions()
    assert (half1, int1) == (int0, half0)


def test_auto_align_keeps_positions_in_range():
    picker, _ = _make_picker()
    picker.auto_align()
    half, integer = picker.positions()
    assert -5.0 <= half <= 5.0
    assert -5.0 <= integer <= 5.0


def test_magnitude_only_toggle():
    picker, _ = _make_picker()
    assert picker.magnitude_only is False
    picker.set_magnitude_only(True)
    assert picker.magnitude_only is True


def test_force_magnitude_starts_on():
    picker, _ = _make_picker(force_magnitude=True)
    assert picker.magnitude_only is True
