from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep import (
    TwoLinePicker,
    find_best_mirror_position,
    fold_initial_lines,
)


def _spectrum(n_dev: int = 60, n_freq: int = 30):
    devs = np.linspace(-5.0, 5.0, n_dev).astype(np.float64)
    freqs = np.linspace(4.0, 5.0, n_freq).astype(np.float64)
    sig = np.zeros((n_dev, n_freq), dtype=np.complex128)
    sig += np.exp(-(devs[:, None] ** 2) / (2 * 1.0**2))
    return sig, devs, freqs


def _make_picker(**kwargs) -> TwoLinePicker:
    sig, devs, freqs = _spectrum()
    fig = Figure()
    FigureCanvasAgg(fig)
    return TwoLinePicker(fig, sig, devs, freqs, **kwargs)


def test_fold_initial_lines_defaults() -> None:
    _sig, devs, _freqs = _spectrum()
    half, integer = fold_initial_lines(devs, None, None)
    assert devs[0] <= half <= devs[-1]
    assert devs[0] <= integer <= devs[-1]


def test_find_best_mirror_position_prefers_symmetric_center() -> None:
    n = 51
    devs = np.linspace(-5.0, 5.0, n, dtype=np.float64)
    center = float(devs[n // 2])
    col = np.abs(devs).reshape(n, 1).astype(np.float64)
    pos = find_best_mirror_position(devs, col, center, search_width=1.0)
    assert pos == pytest.approx(center, abs=0.5 * (devs[1] - devs[0]))


def test_picker_drag_swap_and_compute_apply() -> None:
    picker = _make_picker()
    half0, int0 = picker.positions()
    picker.on_press(half0)
    picker.on_move(half0 + 0.5)
    half1, int1 = picker.positions()
    assert half1 != half0
    assert int1 == int0

    picker.swap()
    assert picker.positions() == (int1, half1)

    before = picker.positions()
    aligned = picker.compute_aligned_positions()
    assert picker.positions() == before
    picker.apply_positions(*aligned)
    assert picker.positions() == aligned


def test_picker_magnitude_toggle() -> None:
    picker = _make_picker()
    assert picker.magnitude_only is False
    picker.set_magnitude_only(True)
    assert picker.magnitude_only is True
