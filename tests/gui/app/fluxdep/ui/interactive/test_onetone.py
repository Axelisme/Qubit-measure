"""One-tone Qt controls operate on the app-owned committed Session."""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtWidgets import QPushButton, QSlider
from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import ResultPreviewWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow


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
    undo = next(b for b in view.findChildren(QPushButton) if b.text() == "Undo")
    undo.click()
    assert context.session.snapshot() == low
    assert slider.value() == 10
    np.testing.assert_array_equal(view.get_result()[0], low_points[0])


def test_finish_request_rejected_after_terminal_input(widget, onetone_controller):
    view, context = widget
    fired: list[bool] = []
    view.finished.connect(lambda: fired.append(True))
    finish = next(b for b in view.findChildren(QPushButton) if b.text() == "Finish")
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
        undo = next(
            b for b in replacement.findChildren(QPushButton) if b.text() == "Undo"
        )
        undo.click()
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
        finish = next(b for b in view.findChildren(QPushButton) if b.text() == "Finish")
        finish.click()
        entry = onetone_controller.state.spectrums["one"]
        assert entry.points_selected
        np.testing.assert_array_equal(entry.points["dev_values"], np.sort(expected[0]))
        assert window.findChildren(ResultPreviewWidget)
        assert onetone_controller.interactive.current_onetone_pick() is None
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
