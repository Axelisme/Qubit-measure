"""Production composition and line-picker Finish without independent processes."""

from __future__ import annotations

import time

import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QPushButton
from zcu_tools.gui.app.fluxdep.app import FluxDepGuiBehavior
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow


@pytest.fixture
def composition(qapp):
    assembly = FluxDepGuiBehavior().assemble(None)
    assert isinstance(assembly.controller, Controller)
    assert isinstance(assembly.window, MainWindow)
    yield assembly.controller, assembly.window
    assembly.window.close()
    assembly.window.deleteLater()
    qapp.processEvents()


def _wait_for_alignment(plugin, qapp) -> None:
    deadline = time.monotonic() + 3.0
    while plugin.alignment_busy and time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
        qapp.processEvents()
    assert not plugin.alignment_busy


def test_production_twotone_preview_and_finish_advance_stage(
    composition, spectrum_hdf5, qapp
):
    import numpy as np
    from zcu_tools.analysis.fluxdep.twotone import analyze_twotone_pick
    from zcu_tools.gui.app.fluxdep.ui.interactive.find_points import FindPointsWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.result_preview import (
        ResultPreviewWidget,
    )

    ctrl, window = composition
    name = ctrl.load_spectrum(spectrum_hdf5[0], spec_type="TwoTone")
    ctrl.set_active_spectrum(name)
    ctrl.set_alignment(name, 0.0, 1.0)
    context = ctrl.interactive.current_twotone_pick()
    widget = window.findChild(FindPointsWidget)
    assert context is not None
    assert widget is not None
    deadline = time.monotonic() + 3.0
    while widget.preview_view() is None and time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
        qapp.processEvents()
    view = widget.preview_view()
    assert view is not None
    expected = analyze_twotone_pick(context.plugin.inputs, context.session.snapshot())
    np.testing.assert_array_equal(view.result.dev_values, expected.dev_values)
    finish = next(
        item for item in widget.findChildren(QPushButton) if item.text() == "Finish"
    )
    finish.click()
    assert ctrl.state.spectrums[name].points_completed
    assert ctrl.interactive.current_twotone_pick() is None
    assert window.findChild(ResultPreviewWidget) is not None


def test_production_background_delivery_and_gui_finish_publish_same_session(
    composition, spectrum_hdf5, qapp
):
    ctrl, window = composition
    name = ctrl.load_spectrum(spectrum_hdf5[0], spec_type="OneTone")
    ctrl.set_active_spectrum(name)
    context = ctrl.interactive.current_line_pick()
    assert context is not None
    context.plugin.start_alignment(context.session)
    _wait_for_alignment(context.plugin, qapp)
    assert context.plugin.info()["alignment_error"] is None
    assert context.session.can_undo()
    context.session.undo()
    context.plugin.execute_command(
        context.session, "move_line", {"role": "half", "position": 0.5}
    )
    context.plugin.execute_command(
        context.session, "move_line", {"role": "integer", "position": 2.0}
    )
    widget = window.findChild(LinePickerWidget)
    assert widget is not None
    assert widget.get_result() == (0.5, 2.0)
    finish = next(
        button
        for button in widget.findChildren(QPushButton)
        if button.text() == "Finish"
    )
    finish.click()
    assert ctrl.state.spectrums[name].aligned
    assert ctrl.state.spectrums[name].flux_period == 3.0
    assert ctrl.interactive.current_line_pick() is None
