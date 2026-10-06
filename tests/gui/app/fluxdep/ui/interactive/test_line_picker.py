"""Session-backed line-picker controls and disposable Qt preview."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from qtpy.QtWidgets import QCheckBox, QPushButton
from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
    FluxPickState,
    analyze_flux_pick,
)
from zcu_tools.gui.app.fluxdep.interactive import LinePickContext
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.gui.interactive.flux_pick import SharedFluxPickPlugin
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


@pytest.fixture
def context():
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.asarray(
        np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30)), dtype=np.complex128
    )
    inputs = FluxPickInputs(signals, devs, freqs)
    plugin = SharedFluxPickPlugin(
        inputs,
        FluxPickState(flux_half=0.0, flux_int=2.0, magnitude_only=True),
        build_result=lambda state: analyze_flux_pick(inputs, state),
    )
    session = plugin.open(ManualOwnerScheduler())
    yield LinePickContext("sample", plugin, session)
    session.dispose()


@pytest.fixture
def widget(qapp, context):
    view = LinePickerWidget(context)
    yield view
    view.teardown()
    view.deleteLater()
    qapp.processEvents()


def _button(widget: LinePickerWidget, label: str):
    return next(
        button for button in widget.findChildren(QPushButton) if button.text() == label
    )


def _event(widget: LinePickerWidget, kind: str, x: float) -> MouseEvent:
    widget.canvas.draw()
    axes = widget.figure.axes[0]
    px, py = axes.transData.transform((x, sum(axes.get_ylim()) / 2))
    return MouseEvent(kind, widget.canvas, px, py, button=MouseButton.LEFT)


def test_gui_controls_and_command_share_state_with_undo(widget, context):
    box = next(
        box for box in widget.findChildren(QCheckBox) if box.text() == "Conjugate Line"
    )
    box.click()
    assert context.session.snapshot().conjugate
    context.plugin.execute_command(
        context.session, "move_line", {"role": "half", "position": 0.5}
    )
    assert widget.get_result() == (0.5, 2.5)
    _button(widget, "Swap Lines").click()
    assert widget.get_result() == (2.5, 0.5)
    _button(widget, "Undo").click()
    assert widget.get_result() == (0.5, 2.5)


def test_pointer_preview_does_not_commit_until_release(widget, context):
    before = context.session.snapshot()
    widget.on_press(_event(widget, "button_press_event", 0.0))
    widget.on_move(_event(widget, "motion_notify_event", 0.8))
    assert context.session.snapshot() == before
    widget.on_release(_event(widget, "button_release_event", 0.8))
    assert context.session.snapshot().flux_half == pytest.approx(0.8, abs=0.03)
    assert context.session.undo() == before


def test_command_cancels_old_pointer_preview(widget, context):
    widget.on_press(_event(widget, "button_press_event", 0.0))
    widget.on_move(_event(widget, "motion_notify_event", 0.8))
    context.plugin.execute_command(
        context.session, "move_line", {"role": "half", "position": 1.0}
    )
    widget.on_release(_event(widget, "button_release_event", 0.8))
    assert widget.get_result() == (1.0, 2.0)


def test_teardown_detaches_view_without_closing_owner_session(widget, context):
    widget.teardown()
    widget.teardown()
    context.plugin.actions.move.execute(context.session, ("half", 1.0))
    assert context.session.snapshot().flux_half == 1.0
    _button(widget, "Swap Lines").click()
    assert context.session.snapshot().flux_half == 1.0


def test_finish_requests_owner_without_using_preview(widget, context):
    requests: list[tuple[float, float]] = []
    widget.finished.connect(lambda: requests.append(widget.get_result()))
    widget.on_press(_event(widget, "button_press_event", 0.0))
    widget.on_move(_event(widget, "motion_notify_event", 0.8))
    _button(widget, "Finish").click()
    assert requests == [(0.0, 2.0)]
    assert context.session.snapshot().flux_half == 0.0
