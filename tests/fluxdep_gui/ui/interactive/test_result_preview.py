"""Headless test for ResultPreviewWidget (read-only finished-spectrum view)."""

from __future__ import annotations

import numpy as np
from zcu_tools.fluxdep_gui.state import SpectrumEntry
from zcu_tools.fluxdep_gui.ui.interactive.result_preview import ResultPreviewWidget
from zcu_tools.notebook.persistance import PointsData, SpectrumData


def _entry() -> SpectrumEntry:
    dev = np.linspace(-5.0, 5.0, 8).astype(np.float64)
    freqs = np.linspace(5.0, 6.0, 6).astype(np.float64)
    raw = SpectrumData(
        dev_values=dev,
        fluxs=dev,
        freqs=freqs,
        signals=np.ones((8, 6), dtype=np.complex128),
    )
    pts = PointsData(
        dev_values=np.array([-2.0, 1.0]),
        fluxs=np.array([0.3, 0.6]),
        freqs=np.array([5.2, 5.5]),
    )
    return SpectrumEntry(
        name="s.hdf5",
        spec_type="TwoTone",
        raw=raw,
        points=pts,
        flux_half=0.0,
        flux_int=1.0,
        aligned=True,
        points_selected=True,
    )


def test_result_preview_builds(qapp):
    w = ResultPreviewWidget(_entry())
    # one axes with the spectrum image + the scatter + two flux markers
    ax = w._figure.axes[0]
    assert len(ax.images) == 1  # the (no-mask) spectrum
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(dashed) == 2  # half + int flux markers
    w.deleteLater()
