"""Qt TwoTone controls, drag commits and externally ordered preview delivery."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QSlider
from zcu_tools.analysis.fluxdep.twotone import (
    TwoTonePickView,
    TwoToneTool,
    analyze_twotone_pick,
)
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import SpectrumChangedPayload
from zcu_tools.gui.app.fluxdep.state import spectrum_version_key
from zcu_tools.gui.app.fluxdep.twotone import TwoTonePickPlugin
from zcu_tools.gui.app.fluxdep.ui.interactive.find_points import FindPointsWidget
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


@dataclass
class PreviewRequest:
    """Captured projection and its owner-loop delivery callbacks."""

    compute: Callable[[], TwoTonePickView]
    on_done: Callable[[TwoTonePickView], None]
    on_error: Callable[[Exception], None]


class PreviewQueue:
    """Hold submitted preview work to control public completion order."""

    def __init__(self) -> None:
        self.requests: list[PreviewRequest] = []

    def submit(
        self,
        compute: Callable[[], TwoTonePickView],
        on_done: Callable[[TwoTonePickView], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self.requests.append(PreviewRequest(compute, on_done, on_error))


def wait_for_requests(queue: PreviewQueue, count: int) -> None:
    deadline = time.monotonic() + 2.0
    while len(queue.requests) < count and time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
    assert len(queue.requests) >= count


def button(widget: FindPointsWidget, text: str) -> QPushButton:
    return next(
        item for item in widget.findChildren(QPushButton) if item.text() == text
    )


def slider(widget: FindPointsWidget, label: str) -> QSlider:
    return next(
        item for item in widget.findChildren(QSlider) if item.accessibleName() == label
    )


def combo(widget: FindPointsWidget, label: str) -> QComboBox:
    return next(
        item
        for item in widget.findChildren(QComboBox)
        if item.accessibleName() == label
    )


@pytest.fixture
def presented(
    twotone_controller: Controller, qapp
) -> Iterator[tuple[FindPointsWidget, PreviewQueue, Controller]]:
    context = twotone_controller.interactive.begin_twotone_pick("two")
    queue = PreviewQueue()
    widget = FindPointsWidget(context, submit_preview=queue.submit)
    yield widget, queue, twotone_controller
    widget.teardown()
    widget.deleteLater()
    qapp.processEvents()


def test_command_controls_undo_and_tool_history_share_session(presented) -> None:
    widget, _queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    before = context.session.snapshot()
    slider(widget, "Threshold").setValue(2000)
    assert context.session.snapshot().threshold == 2.0
    slider(widget, "Brush width").setValue(4)
    combo(widget, "Operation").setCurrentText("Erase")
    assert context.session.snapshot().width == 0.004
    assert context.session.snapshot().mode == "erase"
    button(widget, "Undo").click()
    state = context.session.snapshot()
    assert state.threshold == before.threshold
    assert state.width == before.width
    assert state.mode == before.mode
    context.plugin.execute_command(
        context.session, "set_tool", {"width": 0.03, "mode": "erase"}
    )
    assert slider(widget, "Brush width").value() == 30
    assert combo(widget, "Operation").currentText() == "Erase"
    assert not context.session.can_undo()
    combo(widget, "Smooth method").setCurrentText("Gaussian")
    slider(widget, "Smooth").setValue(2000)
    assert context.session.snapshot().sigma == 2.0
    assert context.session.snapshot().smooth_method == "gaussian"


def test_rejected_gaussian_switch_reprojects_wavelet_zero_sigma_and_shows_error(
    presented,
) -> None:
    widget, _queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    slider(widget, "Smooth").setValue(0)
    before = context.session.snapshot()
    assert before.sigma == 0.0
    assert before.smooth_method == "wavelet"
    status = next(
        item
        for item in widget.findChildren(QLabel)
        if item.accessibleName() == "Status"
    )
    previous_feedback = status.text()
    combo(widget, "Smooth method").setCurrentText("Gaussian")
    after = context.session.snapshot()
    assert after.sigma == 0.0
    assert after.smooth_method == "wavelet"
    assert combo(widget, "Smooth method").currentText() == "Wavelet"
    assert slider(widget, "Smooth").value() == 0
    np.testing.assert_array_equal(after.mask, before.mask)
    assert status.text() != previous_feedback
    assert "sigma" in status.text().lower()
    assert context.session.can_undo()
    restored = context.session.undo()
    assert restored.sigma == 1.0
    assert restored.smooth_method == "wavelet"


def test_show_options_do_not_change_state_or_create_history(presented) -> None:
    widget, _queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    before = context.session.snapshot()
    for control in widget.findChildren(QCheckBox):
        control.setChecked(not control.isChecked())
    assert not context.session.can_undo()
    actual = context.session.snapshot()
    np.testing.assert_array_equal(actual.mask, before.mask)
    assert actual.threshold == before.threshold
    assert actual.width == before.width


def test_perform_all_mode_and_clear_use_same_actions(presented) -> None:
    widget, _queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    combo(widget, "Operation").setCurrentText("Erase")
    button(widget, "Perform on all").click()
    assert not context.session.snapshot().mask.any()
    assert widget.get_result()[0].size == 0
    combo(widget, "Operation").setCurrentText("Select")
    button(widget, "Perform on all").click()
    assert context.session.snapshot().mask.all()
    assert widget.get_result()[0].size > 0
    button(widget, "Clear").click()
    assert not context.session.snapshot().mask.any()
    button(widget, "Undo").click()
    assert context.session.snapshot().mask.all()


def mouse(widget: FindPointsWidget, name: str, x: float, y: float) -> MouseEvent:
    axis = widget.figure.axes[0]
    px, py = axis.transData.transform((x, y))
    return MouseEvent(
        name, widget.canvas, round(float(px)), round(float(py)), button=MouseButton.LEFT
    )


def test_drag_collects_preview_then_one_release_matches_vertices_command(
    presented,
) -> None:
    widget, _queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    context.plugin.set_tool.execute(context.session, TwoToneTool(0.03, "erase"))
    widget.canvas.draw()
    before = context.session.snapshot()
    observed: list[bool] = []
    context.session.subscribe(lambda: observed.append(True))
    events = [
        mouse(widget, "button_press_event", -0.8, 4.56),
        mouse(widget, "motion_notify_event", 0.0, 4.8),
        mouse(widget, "button_release_event", 0.8, 5.04),
    ]
    for event in events[:2]:
        widget.canvas.callbacks.process(event.name, event)
    np.testing.assert_array_equal(context.session.snapshot().mask, before.mask)
    assert observed == []
    widget.canvas.callbacks.process(events[2].name, events[2])
    assert observed == [True]
    peer = TwoTonePickPlugin(context.plugin.inputs)
    peer_session = peer.open(ManualOwnerScheduler())
    peer.execute_command(
        peer_session,
        "stroke",
        {
            "vertices": [[event.xdata, event.ydata] for event in events],
            "width": 0.03,
            "mode": "erase",
        },
    )
    np.testing.assert_array_equal(
        context.session.snapshot().mask, peer_session.snapshot().mask
    )
    assert not context.session.snapshot().mask.all()
    expected = analyze_twotone_pick(peer.inputs, peer_session.snapshot())
    actual_dev, actual_freq = widget.get_result()
    np.testing.assert_array_equal(actual_dev, expected.dev_values)
    np.testing.assert_array_equal(actual_freq, expected.freqs)
    assert observed == [True]


def test_outside_release_finishes_once_and_tool_is_captured_on_press(presented) -> None:
    widget, _queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    context.plugin.set_tool.execute(context.session, TwoToneTool(0.02, "erase"))
    widget.canvas.draw()
    # Center on a device row so the captured radius reaches actual mask cells.
    press_x = float(context.plugin.inputs.spectrum.dev_values[12])
    press = mouse(widget, "button_press_event", press_x, 4.8)
    widget.canvas.callbacks.process(press.name, press)
    context.plugin.execute_command(
        context.session, "set_tool", {"width": 0.004, "mode": "select"}
    )
    release = MouseEvent(
        "button_release_event", widget.canvas, -10, -10, button=MouseButton.LEFT
    )
    widget.canvas.callbacks.process(release.name, release)
    state = context.session.snapshot()
    assert state.width == 0.02
    assert state.mode == "erase"
    assert not state.mask.all()
    assert state.last_change is not None
    assert len(state.last_change.vertices) == 1
    context.session.undo()
    assert context.session.snapshot().width == 0.004
    assert context.session.snapshot().mode == "select"


def test_latest_preview_settlement_ignores_stale_success_and_error(presented) -> None:
    widget, queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    wait_for_requests(queue, 1)
    old = queue.requests[0]
    context.plugin.clear.execute(context.session, None)
    assert widget.preview_view() is None
    wait_for_requests(queue, 2)
    latest = queue.requests[1]
    latest.on_done(latest.compute())
    settled = widget.preview_view()
    assert settled is not None
    assert settled.result.dev_values.size == 0
    old.on_done(old.compute())
    old.on_error(ValueError("obsolete preview failed"))
    after = widget.preview_view()
    assert after is not None
    assert after.result.dev_values.size == 0
    assert not after.state.mask.any()
    detached = widget.preview_view()
    assert detached is not None
    detached.state.mask[:] = True
    assert not context.session.snapshot().mask.any()
    final = widget.preview_view()
    assert final is not None
    assert not final.state.mask.any()


def test_latest_failure_is_not_an_empty_success_and_finish_recomputes(
    presented,
) -> None:
    widget, queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    wait_for_requests(queue, 1)
    request = queue.requests[0]
    request.on_error(ValueError("latest preview failed"))
    assert widget.preview_view() is None
    assert widget.get_result()[0].size > 0
    expected = analyze_twotone_pick(context.plugin.inputs, context.session.snapshot())
    result = ctrl.interactive.finish_twotone_pick()
    np.testing.assert_array_equal(result.dev_values, expected.dev_values)
    request.on_done(request.compute())
    assert widget.preview_view() is None


def test_undo_invalidates_pending_preview_and_closed_widget_cannot_finish(
    presented,
) -> None:
    widget, queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    wait_for_requests(queue, 1)
    context.plugin.clear.execute(context.session, None)
    wait_for_requests(queue, 2)
    stale = queue.requests[1]
    context.session.undo()
    wait_for_requests(queue, 3)
    restored = queue.requests[2]
    restored.on_done(restored.compute())
    stale.on_done(stale.compute())
    view = widget.preview_view()
    assert view is not None
    assert view.state.mask.all()
    assert view.result.dev_values.size > 0
    assert view.mask_added == view.state.mask.size
    assert view.mask_removed == 0
    assert view.added_points.shape[0] == view.result.dev_values.size
    emitted: list[bool] = []
    widget.finished.connect(lambda: emitted.append(True))
    ctrl.interactive.cancel()
    button(widget, "Finish").click()
    assert emitted == []
    assert widget.preview_view() is None


def test_teardown_and_remount_preserve_domain_and_discard_late_delivery(
    presented, qapp
) -> None:
    widget, queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    wait_for_requests(queue, 1)
    pending = queue.requests[0]
    context.plugin.clear.execute(context.session, None)
    widget.quiesce()
    widget.teardown()
    pending.on_done(pending.compute())
    pending.on_error(ValueError("late detached error"))
    assert widget.preview_view() is None
    replacement = FindPointsWidget(context, submit_preview=queue.submit)
    try:
        assert replacement.get_result()[0].size == 0
        assert ctrl.interactive.begin_twotone_pick("two") is context
        assert context.session.can_undo()
        button(replacement, "Undo").click()
        assert replacement.get_result()[0].size > 0
    finally:
        replacement.teardown()
        replacement.deleteLater()
        qapp.processEvents()


def test_finish_pending_preview_publishes_committed_selection(presented) -> None:
    widget, queue, ctrl = presented
    context = ctrl.interactive.current_twotone_pick()
    assert context is not None
    wait_for_requests(queue, 1)
    pending = queue.requests[0]
    stale_preview = pending.compute()
    assert stale_preview.result.dev_values.size > 0
    context.plugin.clear.execute(context.session, None)
    assert widget.preview_view() is None
    version_key = spectrum_version_key("two")
    version_before = ctrl.state.version.get(version_key)
    changes: list[str] = []
    subscription = ctrl.bus.subscribe(
        SpectrumChangedPayload, lambda event: changes.append(event.name)
    )
    try:
        widget.finished.connect(ctrl.interactive.finish_twotone_pick)
        button(widget, "Finish").click()
        entry = ctrl.state.spectrums["two"]
        assert not entry.points_selected
        assert entry.points["dev_values"].size == 0
        assert entry.points["freqs"].size == 0
        assert entry.points["fluxs"].size == 0
        assert ctrl.state.version.get(version_key) == version_before + 1
        assert changes == ["two"]
        assert ctrl.interactive.current_twotone_pick() is None
        pending.on_done(stale_preview)
        assert widget.preview_view() is None
        assert ctrl.state.version.get(version_key) == version_before + 1
        assert changes == ["two"]
        assert ctrl.state.spectrums["two"].points["dev_values"].size == 0
        with pytest.raises(FailedPreconditionError):
            context.plugin.clear.execute(context.session, None)
    finally:
        subscription.unsubscribe()
