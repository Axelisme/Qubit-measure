from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from zcu_tools.notebook.analysis.fluxdep.interactive import find_line
from zcu_tools.notebook.analysis.fluxdep.interactive.find_line import InteractiveLines


def _spectrum() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    devs = np.linspace(-5.0, 5.0, 60, dtype=np.float64)
    freqs = np.linspace(4.0, 5.0, 30, dtype=np.float64)
    signals = np.exp(-(devs[:, None] ** 2) / 2).astype(np.complex128)
    return signals, devs, freqs


def test_interactive_lines_accepts_equal_initial_lines(monkeypatch) -> None:
    monkeypatch.setattr(find_line, "display", lambda _widget: None)
    signals, devs, freqs = _spectrum()

    picker = InteractiveLines(signals, devs, freqs, flux_half=0.0, flux_int=0.0)

    assert picker.get_positions(finish=False) == (0.0, 0.0)
    picker.finish_interactive()


def test_interactive_lines_forwards_selection_and_drag(monkeypatch) -> None:
    monkeypatch.setattr(find_line, "display", lambda _widget: None)
    signals, devs, freqs = _spectrum()
    picker = InteractiveLines(signals, devs, freqs)
    half0, _ = picker.get_positions(finish=False)

    picker.set_picked_half_flux(None)
    picker.onmove(
        SimpleNamespace(
            inaxes=picker.picker._ax_main,
            xdata=half0 + 0.5,
            ydata=float(freqs[len(freqs) // 2]),
        )
    )

    half1, _ = picker.get_positions(finish=False)
    assert half1 == half0 + 0.5
    picker.finish_interactive()


def test_interactive_lines_ignores_loss_axes_motion(monkeypatch) -> None:
    monkeypatch.setattr(find_line, "display", lambda _widget: None)
    signals, devs, freqs = _spectrum()
    picker = InteractiveLines(signals, devs, freqs)
    before = picker.get_positions(finish=False)

    picker.set_picked_half_flux(None)
    picker.onmove(
        SimpleNamespace(
            inaxes=picker.picker._ax_loss,
            xdata=before[0] + 1.0,
            ydata=float(freqs[len(freqs) // 2]),
        )
    )

    assert picker.get_positions(finish=False) == before
    picker.finish_interactive()
