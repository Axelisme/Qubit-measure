"""Line-picker background projections match the corresponding spectrum editors."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from zcu_tools.analysis.fluxdep.processing import cast2real_and_norm
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import EventOrigin

from tests.gui.app.fluxdep._window import editor


def _background(widget) -> tuple[np.ndarray, tuple[float, float]]:
    """Image array and colour limits of the first axes of an editor's figure."""
    image = widget.figure.axes[0].get_images()[0]
    return np.asarray(image.get_array()), image.get_clim()


def test_onetone_line_picker_draws_the_sample_picker_background(
    qapp, onetone_controller
):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget

    ctrl = onetone_controller
    win = MainWindow(ctrl)
    try:
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            ctrl.interactive.begin_sample_pick("one")
        sample = editor(win)
        assert isinstance(sample, OneToneWidget)
        expected, expected_clim = _background(sample)
        ctrl.interactive.cancel("one")
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            ctrl.interactive.begin_line_pick("one")
        lines = editor(win)
        assert isinstance(lines, LinePickerWidget)
        background, clim = _background(lines)
        np.testing.assert_allclose(background, expected)
        assert clim == pytest.approx(expected_clim)
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def _background_at_coordinates(
    canvas: FigureCanvasQTAgg,
    dev_values: np.ndarray,
    freqs: np.ndarray,
    clim: tuple[float, float],
) -> np.ndarray:
    """Sample the public image raster without grid or marker overlays."""
    axes = canvas.figure.axes[0]
    image = axes.get_images()[0]
    assert image.get_cmap().name == "gray_r"
    original_clim = image.get_clim()
    try:
        # Compare projections, not the editors' independently chosen contrast.
        image.set_clim(*clim)
        canvas.draw()
        rgba, left, bottom, _ = image.make_image(canvas.get_renderer())
        assert rgba is not None
        colours = []
        for dev in dev_values:
            for freq in freqs:
                x, y = axes.transData.transform((dev, freq))
                ix, iy = int(x - left), int(y - bottom)
                assert 0 <= ix < rgba.shape[1] and 0 <= iy < rgba.shape[0]
                colours.append(rgba[iy, ix])
        return np.asarray(colours)
    finally:
        image.set_clim(*original_clim)


def test_twotone_line_picker_keeps_the_phase_projection_of_its_preview(
    qapp, twotone_controller
):
    from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    ctrl = twotone_controller
    entry = ctrl.state.spectrums["two"]
    devs, freqs = entry.raw["dev_values"], entry.raw["freqs"]
    # Probe actual sample centres, excluding the preview's clipped endpoints.
    probe_devs, probe_freqs = devs[1:-1], freqs[1:-1]
    preview = ResultPreviewWidget(entry)
    try:
        image = preview.figure.axes[0].get_images()[0]
        values = np.asarray(image.get_array())
        clim = float(values.min()), float(values.max())
        expected = _background_at_coordinates(
            preview.canvas, probe_devs, probe_freqs, clim
        )
        # A magnitude-only regression must remain distinguishable after raster
        # quantization; comparing two equally wrong backgrounds is not enough.
        image.set_data(cast2real_and_norm(entry.raw["signals"], use_phase=False).T)
        magnitude = _background_at_coordinates(
            preview.canvas, probe_devs, probe_freqs, clim
        )
        assert not np.allclose(expected, magnitude, rtol=0, atol=1)
    finally:
        preview.deleteLater()
    win = MainWindow(ctrl)
    try:
        ctrl.interactive.cancel("two")  # the window opened the sample picker
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            ctrl.interactive.begin_line_pick("two")
        lines = editor(win)
        assert isinstance(lines, LinePickerWidget)
        background = _background_at_coordinates(
            lines.canvas, probe_devs, probe_freqs, clim
        )
        np.testing.assert_allclose(background, expected, rtol=0, atol=1)
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()
