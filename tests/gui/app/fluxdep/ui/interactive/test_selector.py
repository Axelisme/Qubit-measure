"""Selector controls/canvas and injected preview delivery through public seams."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from qtpy import QtCore, QtWidgets
from zcu_tools.analysis.fluxdep.cross_selection import CrossSelectionView
from zcu_tools.analysis.fluxdep.stroke import BrushPoint, BrushStroke, BrushTool
from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget
from zcu_tools.gui.expected_error import FailedPreconditionError


@dataclass
class PreviewJob:
    compute: Callable[[], CrossSelectionView]
    on_done: Callable[[CrossSelectionView], None]
    on_error: Callable[[Exception], None]


def wait_for_debounce():
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(100, loop.quit)
    loop.exec()


def control[Control: QtWidgets.QWidget](
    widget: QtWidgets.QWidget, kind: type[Control], name: str
) -> Control:
    return next(
        item for item in widget.findChildren(kind) if item.accessibleName() == name
    )


@pytest.fixture
def selector(qapp, cross_controller):
    context = cross_controller.interactive.begin_cross_selection()
    jobs: list[PreviewJob] = []

    def submit(compute, on_done, on_error):
        jobs.append(PreviewJob(compute, on_done, on_error))

    widget = SelectorWidget(
        context,
        on_apply=cross_controller.interactive.apply_cross_selection,
        submit_preview=submit,
    )
    yield widget, context, jobs
    widget.teardown()
    widget.deleteLater()
    qapp.processEvents()


def test_controls_share_state_tool_undo_and_nonterminal_apply(
    selector, cross_controller
):
    widget, context, _ = selector
    width = control(widget, QtWidgets.QSlider, "Brush width")
    distance = control(widget, QtWidgets.QSlider, "Min distance")
    mode = control(widget, QtWidgets.QComboBox, "Operation")
    width.setValue(20)
    distance.setValue(60)
    mode.setCurrentText("Erase")
    assert (
        context.session.snapshot().width,
        context.session.snapshot().min_distance,
    ) == (0.02, 0.06)
    control(widget, QtWidgets.QPushButton, "Perform on all").click()
    assert not widget.get_result().selected.any()
    control(widget, QtWidgets.QPushButton, "Apply").click()
    assert not cross_controller.state.selection.selected.any()
    assert cross_controller.interactive.current_cross_selection() is context
    control(widget, QtWidgets.QPushButton, "Undo").click()
    assert widget.get_result().selected.all()
    mode.setCurrentText("Select")
    control(widget, QtWidgets.QPushButton, "Clear").click()
    assert not context.session.snapshot().selected.any()


def test_pointer_only_preview_release_once_and_outside_axes(selector):
    widget, context, _ = selector
    context.plugin.set_tool.execute(context.session, BrushTool(0.02, "erase"))
    widget.canvas.draw()
    axes = widget.figure.axes[0]
    start = axes.transData.transform((0.5, 4.5))
    notifications = []
    context.session.subscribe(lambda: notifications.append(context.session.snapshot()))
    press = MouseEvent(
        "button_press_event", widget.canvas, *start, button=MouseButton.LEFT
    )
    widget.canvas.callbacks.process("button_press_event", press)
    assert context.session.snapshot().selected.all()
    outside = MouseEvent(
        "button_release_event", widget.canvas, -10, -10, button=MouseButton.LEFT
    )
    widget.canvas.callbacks.process("button_release_event", outside)
    assert len(notifications) == 1
    np.testing.assert_array_equal(
        widget.get_result().selected, [True, False, True, False]
    )


def test_failed_zero_width_moving_gesture_is_atomic_and_restores_controls(selector):
    widget, context, _ = selector
    context.plugin.set_tool.execute(context.session, BrushTool(0.0, "erase"))
    widget.canvas.draw()
    axes = widget.figure.axes[0]
    for event_name, point in (
        ("button_press_event", (0.1, 4.1)),
        ("motion_notify_event", (0.9, 4.9)),
        ("button_release_event", (0.9, 4.9)),
    ):
        position = axes.transData.transform(point)
        event = MouseEvent(
            event_name, widget.canvas, *position, button=MouseButton.LEFT
        )
        widget.canvas.callbacks.process(event_name, event)
    assert context.session.snapshot().selected.all()
    assert not context.session.can_undo()
    assert control(widget, QtWidgets.QSlider, "Brush width").value() == 0
    assert control(widget, QtWidgets.QLabel, "Status").text()


def test_tool_only_commit_updates_controls_without_resubmitting_downsample(selector):
    widget, context, jobs = selector
    wait_for_debounce()
    count = len(jobs)
    context.plugin.set_tool.execute(context.session, BrushTool(0.03, "erase"))
    wait_for_debounce()
    assert len(jobs) == count
    assert control(widget, QtWidgets.QSlider, "Brush width").value() == 30
    assert control(widget, QtWidgets.QComboBox, "Operation").currentText() == "Erase"


def test_preview_presents_current_stroke_and_undo_inverse_geometry(selector):
    widget, context, jobs = selector
    vertices = (BrushPoint(0.5, 4.5),)
    context.plugin.stroke.execute(context.session, BrushStroke(vertices, 0.02, "erase"))
    wait_for_debounce()
    jobs[-1].on_done(jobs[-1].compute())
    view = widget.preview_view()
    assert view is not None
    assert view.stroke_vertices == vertices and view.stroke_width == 0.02
    np.testing.assert_array_equal(view.removed_points, [[0.5, 4.5], [0.5, 4.5]])
    control(widget, QtWidgets.QPushButton, "Undo").click()
    wait_for_debounce()
    jobs[-1].on_done(jobs[-1].compute())
    inverse = widget.preview_view()
    assert inverse is not None
    assert inverse.stroke_vertices == vertices and inverse.stroke_width == 0.02
    np.testing.assert_array_equal(inverse.added_points, [[0.5, 4.5], [0.5, 4.5]])


def test_latest_success_discards_old_error_and_old_success(selector):
    widget, context, jobs = selector
    wait_for_debounce()
    old = jobs[-1]
    context.plugin.clear.execute(context.session, None)
    wait_for_debounce()
    latest = jobs[-1]
    latest.on_done(latest.compute())
    assert widget.preview_view() is not None
    old.on_error(RuntimeError("obsolete error"))
    old.on_done(old.compute())
    view = widget.preview_view()
    assert view is not None and not view.result.selected.any()
    status = control(widget, QtWidgets.QLabel, "Status")
    assert "obsolete error" not in status.text()


def test_latest_error_is_visible_and_cannot_block_synchronous_apply(
    selector, cross_controller
):
    widget, context, jobs = selector
    wait_for_debounce()
    context.plugin.clear.execute(context.session, None)
    wait_for_debounce()
    jobs[-1].on_error(RuntimeError("latest failure"))
    assert widget.preview_view() is None
    assert "latest failure" in control(widget, QtWidgets.QLabel, "Status").text()
    control(widget, QtWidgets.QPushButton, "Apply").click()
    assert not cross_controller.state.selection.selected.any()
    assert context.session.can_undo()


def test_teardown_rejects_stale_apply_and_late_preview_without_cancelling_owner(
    selector, cross_controller
):
    widget, context, jobs = selector
    wait_for_debounce()
    job = jobs[-1]
    apply_button = control(widget, QtWidgets.QPushButton, "Apply")
    widget.teardown()
    job.on_done(job.compute())
    job.on_error(RuntimeError("late failure"))
    assert widget.preview_view() is None
    apply_button.click()
    assert cross_controller.state.selection.selected is None
    assert cross_controller.interactive.current_cross_selection() is context
    with pytest.raises(FailedPreconditionError):
        widget.get_result()
