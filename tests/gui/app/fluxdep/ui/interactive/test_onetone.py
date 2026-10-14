"""One-tone Qt controls operate on the app-owned committed Session."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtWidgets import QSlider
from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import ResultPreviewWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow

from tests.gui.app.fluxdep._window import (
    column_tops,
    picker_button,
    unlabelled_inputs,
)


@pytest.fixture
def widget(qapp, onetone_controller):
    context = onetone_controller.interactive.begin_onetone_pick("one")
    view = OneToneWidget(context)
    yield view, context
    view.teardown()
    view.deleteLater()
    qapp.processEvents()


def test_controls_command_and_undo_share_committed_points(widget):
    view, context = widget
    slider = view.findChildren(QSlider)[0]
    slider.setValue(10)
    low = context.session.snapshot()
    assert low.threshold == 0.1
    low_points = view.get_result()
    assert low_points[0].size == 2
    context.plugin.execute_command(context.session, "set_threshold", {"threshold": 5.0})
    assert slider.value() == 500
    assert view.get_result()[0].size == 0
    picker_button(view, "undo").click()
    assert context.session.snapshot() == low
    assert slider.value() == 10
    np.testing.assert_array_equal(view.get_result()[0], low_points[0])


def test_finish_request_rejected_after_terminal_input(widget, onetone_controller):
    view, context = widget
    fired: list[bool] = []
    view.finished.connect(lambda: fired.append(True))
    finish = picker_button(view, "finish")
    finish.click()
    assert fired == [True]
    committed = context.session.snapshot()
    onetone_controller.interactive.finish_onetone_pick()
    assert context.session.snapshot() == committed
    finish.click()
    assert fired == [True]


def test_detach_remount_retains_domain_and_undo(widget, onetone_controller, qapp):
    view, context = widget
    seed = context.session.snapshot()
    context.plugin.execute_command(context.session, "set_threshold", {"threshold": 0.1})
    committed = context.session.snapshot()
    view.teardown()
    view.teardown()
    view.findChildren(QSlider)[0].setValue(500)
    assert context.session.snapshot() == committed
    assert onetone_controller.interactive.current_onetone_pick() is context
    replacement = OneToneWidget(context)
    try:
        assert replacement.findChildren(QSlider)[0].value() == 10
        picker_button(replacement, "undo").click()
        assert context.session.snapshot() == seed
    finally:
        replacement.teardown()
        replacement.deleteLater()
        qapp.processEvents()


def test_main_window_finish_publishes_and_advances_to_preview(qapp, onetone_controller):
    window = MainWindow(onetone_controller)
    try:
        view = window.findChildren(OneToneWidget)[0]
        context = onetone_controller.interactive.current_onetone_pick()
        assert context is not None
        context.plugin.execute_command(
            context.session, "set_threshold", {"threshold": 0.1}
        )
        expected = view.get_result()
        picker_button(view, "finish").click()
        entry = onetone_controller.state.spectrums["one"]
        assert entry.points_completed
        np.testing.assert_array_equal(entry.points["dev_values"], np.sort(expected[0]))
        assert window.findChildren(ResultPreviewWidget)
        assert onetone_controller.interactive.current_onetone_pick() is None
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_control_column_runs_settings_edits_status_then_finish(widget):
    view, _context = widget
    view.resize(1000, 700)
    view.show()
    tops = column_tops(view, ["threshold", "undo", "status", "finish"])
    assert tops == sorted(tops) and len(set(tops)) == len(tops)
    finish = picker_button(view, "finish")
    assert finish.mapTo(view, finish.rect().bottomLeft()).y() > 0.9 * view.height()
    assert view.canvas.x() < finish.mapTo(view, finish.rect().topLeft()).x()
    assert unlabelled_inputs(view) == []


def test_background_matches_the_preview_and_draws_the_dip_dark(
    widget, onetone_controller
):
    view, _context = widget
    entry = onetone_controller.state.spectrums["one"]
    preview = ResultPreviewWidget(entry)
    try:
        canvas = preview.findChild(FigureCanvasQTAgg)
        assert canvas is not None
        shown = canvas.figure.axes[0].get_images()[0]
        picked = view.figure.axes[0].get_images()[0]
        picked_values, shown_values = picked.get_array(), shown.get_array()
        assert picked_values is not None and shown_values is not None
        # The sample picker and the preview normalize the spectrum the same way.
        np.testing.assert_allclose(picked_values, shown_values)
        assert picked.get_clim() == pytest.approx(shown.get_clim())
        # Under gray_r the resonator dip is dark on a light background.
        signals = np.abs(entry.raw["signals"])
        dev, freq = np.unravel_index(np.argmin(signals), signals.shape)
        level = np.asarray(picked.norm(picked_values))
        assert level[freq, dev] > 0.9
        assert np.median(level) < 0.5
    finally:
        preview.deleteLater()
