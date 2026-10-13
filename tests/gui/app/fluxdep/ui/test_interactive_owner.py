"""Production composition and line-picker Finish without independent processes."""

from __future__ import annotations

import time
from copy import deepcopy

import pytest
from qtpy.QtCore import QEventLoop, QTimer
from zcu_tools.gui.app.fluxdep.app import FluxDepGuiBehavior
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import EventOrigin

from tests.gui.app.fluxdep._window import editor, picker_button, spectra_list


@pytest.fixture
def composition(qapp):
    assembly = FluxDepGuiBehavior().assemble(None)
    assert isinstance(assembly.controller, Controller)
    assert isinstance(assembly.window, MainWindow)
    yield assembly.controller, assembly.window
    assembly.window.close()
    assembly.window.deleteLater()
    qapp.processEvents()


def test_agent_follows_named_draft_and_remote_user_can_switch_display(
    qapp, onetone_controller
):
    ctrl = onetone_controller
    other = deepcopy(ctrl.state.get_spectrum("one"))
    other.name = "other"
    ctrl.state.put_spectrum(other)
    view = MainWindow(ctrl)
    try:
        first = ctrl.interactive.begin_line_pick("one")
        ctrl.control_identity.connect("agent-link", "agent", "owner")
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            second = ctrl.interactive.begin_line_pick("other")
        assert ctrl.state.active_spectrum == "other", (
            "GUI must follow the named agent draft"
        )
        spectra = spectra_list(view)
        assert spectra.isEnabled()
        spectra.setCurrentRow(0)
        assert ctrl.state.active_spectrum == "one"
        first.session.ensure_input_open()
        second.session.ensure_input_open()
        displayed = editor(view)
        assert isinstance(displayed, LinePickerWidget)
        assert not picker_button(displayed, "finish").isEnabled()
    finally:
        ctrl.control_identity.reclaim_local()
        view.close()
        view.deleteLater()
        qapp.processEvents()


def _wait_for_alignment(plugin, qapp) -> None:
    deadline = time.monotonic() + 3.0
    while plugin.alignment_busy and time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
        qapp.processEvents()
    assert not plugin.alignment_busy


@pytest.mark.parametrize("spec_type", ["OneTone", "TwoTone"])
def test_agent_reopens_completed_point_picker_and_gui_uses_same_session(
    composition, spectrum_hdf5, qapp, spec_type
):
    import numpy as np
    from zcu_tools.gui.app.fluxdep.ui.interactive.find_points import FindPointsWidget
    from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget
    from zcu_tools.gui.event_bus import EventOrigin

    ctrl, window = composition
    name = ctrl.load_spectrum(spectrum_hdf5[0], spec_type=spec_type)
    ctrl.set_active_spectrum(name)
    ctrl.set_alignment(name, 0.0, 1.0)
    ctrl.set_points(name, np.array([0.0]), np.array([5.0]))
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        if spec_type == "OneTone":
            context = ctrl.interactive.begin_onetone_pick(name)
            context.plugin.execute_command(
                context.session, "set_threshold", {"threshold": 0.5}
            )
        else:
            context = ctrl.interactive.begin_twotone_pick(name)
            context.plugin.execute_command(context.session, "clear", {})
    widget = editor(window)
    if spec_type == "OneTone":
        assert isinstance(widget, OneToneWidget)
    else:
        assert isinstance(widget, FindPointsWidget)
    active = ctrl.interactive.inspect()
    assert active is not None and active.context is context
    expected = context.plugin.build_result(context.session.snapshot())
    devs, freqs = widget.get_result()
    np.testing.assert_array_equal(devs, expected.dev_values)
    np.testing.assert_array_equal(freqs, expected.freqs)
    assert context.session.can_undo()
    assert ctrl.state.spectrums[name].points_completed


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
    picker_button(widget, "finish").click()
    assert ctrl.state.spectrums[name].points_completed
    assert ctrl.interactive.current_twotone_pick() is None
    assert window.findChild(ResultPreviewWidget) is not None


def test_production_background_delivery_and_gui_finish_publish_same_session(
    composition, spectrum_hdf5, qapp
):
    ctrl, window = composition
    name = ctrl.load_spectrum(spectrum_hdf5[0], spec_type="OneTone")
    spectra = spectra_list(window)
    # A human list pick opens the line picker; an active change alone only displays.
    spectra.setCurrentRow(0)
    assert ctrl.state.active_spectrum == name
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
    picker_button(widget, "finish").click()
    assert ctrl.state.spectrums[name].aligned
    assert ctrl.state.spectrums[name].flux_period == 3.0
    assert ctrl.interactive.current_line_pick() is None
