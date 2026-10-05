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
