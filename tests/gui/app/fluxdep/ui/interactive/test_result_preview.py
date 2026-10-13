"""Headless test for ResultPreviewWidget (read-only finished-spectrum view)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep.models import PointsData
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry
from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import ResultPreviewWidget

from tests.gui.app.fluxdep._window import preview_actions, preview_button


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
        points_completed=True,
    )


def _figure(widget: ResultPreviewWidget) -> Figure:
    canvas = widget.findChild(FigureCanvasQTAgg)
    assert canvas is not None, "the preview must show a figure"
    return canvas.figure


def test_result_preview_builds(qapp):
    w = ResultPreviewWidget(_entry())
    # one axes with the spectrum image + the scatter + two flux markers
    ax = _figure(w).axes[0]
    assert len(ax.images) == 1  # the (no-mask) spectrum
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(dashed) == 2  # half + int flux markers
    w.deleteLater()


@pytest.mark.parametrize("aligned", [False, True])
def test_next_steps_sit_at_the_bottom_of_a_right_column(qapp, aligned):
    entry = dataclasses.replace(_entry(), aligned=aligned)
    widget = ResultPreviewWidget(entry, on_align=lambda: None, on_sample=lambda: None)
    try:
        widget.resize(1000, 700)
        widget.show()
        qapp.processEvents()
        actions = ["realign", "sample"] if aligned else ["align"]
        assert preview_actions(widget) == set(actions)
        canvas = widget.findChild(FigureCanvasQTAgg)
        assert canvas is not None
        canvas_right = canvas.mapTo(widget, canvas.rect().topRight()).x()
        tops = []
        for action in actions:
            button = preview_button(widget, action)
            top_left = button.mapTo(widget, button.rect().topLeft())
            assert top_left.x() > canvas_right, f"{action} must sit right of the plot"
            tops.append(top_left.y())
        last = preview_button(widget, actions[-1])
        bottom = last.mapTo(widget, last.rect().bottomLeft()).y()
        assert bottom > 0.9 * widget.height(), "steps are pinned to the column bottom"
        assert tops == sorted(tops) and len(set(tops)) == len(tops)
    finally:
        widget.deleteLater()
