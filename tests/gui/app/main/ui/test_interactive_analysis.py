"""Production flux frontend through real Qt widgets and Matplotlib mouse events."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtCore import QEvent, Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QFocusEvent, QKeyEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QPushButton,
    QStackedWidget,
    QWidget,
)
from zcu_tools.experiment.v2_gui.adapters._support.flux_pick_frontend import (
    FluxPickFrontend,
)
from zcu_tools.experiment.v2_gui.adapters._support.flux_pick_plugin import (
    make_flux_pick_plugin,
)
from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


class _DeferredEnv:
    def __init__(self) -> None:
        self.pending: list[
            tuple[
                Callable[[], object],
                Callable[[object], None],
                Callable[[Exception], None],
            ]
        ] = []

    def run_background(self, compute, on_done, on_error) -> None:
        self.pending.append((compute, on_done, on_error))


def _frontend(qapp, *, finish=None):
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30))
    req = AnalyzeRequest(
        run_result=SimpleNamespace(signals=signals, values=devs, freqs=freqs),
        analyze_params=object(),
        md=MetaDict(),
        ml=ModuleLibrary(),
        predictor=None,
    )
    plugin = make_flux_pick_plugin(req, force_magnitude=True)
    session = plugin.open(ManualOwnerScheduler())
    env = _DeferredEnv()
    plugin.bind_background(env.run_background)
    completed = []
    cancelled = []

    def on_finish(figure):
        if finish is not None:
            return finish(session, figure)
        completed.append(plugin.finish(session, figure))
        session.dispose()
        return True

    def on_cancel():
        cancelled.append(True)
        session.dispose()
        return True

    widget = FluxPickFrontend(plugin, session, env, on_finish, on_cancel)
    widget.show()
    qapp.processEvents()
    canvas = widget.findChild(FigureCanvasQTAgg)
    assert canvas is not None
    canvas.draw()
    return widget, plugin, session, env, completed, cancelled, canvas


def _button(widget, label):
    return next(b for b in widget.findChildren(QPushButton) if b.text() == label)


def _pointer(canvas, name: str, x: float, y: float = 4.5):
    px, py = canvas.figure.axes[0].transData.transform((x, y))
    event = MouseEvent(name, canvas, int(px), int(py), button=MouseButton.LEFT)
    canvas.callbacks.process(name, event)


def test_drag_is_preview_until_valid_release_and_uses_latest_committed_state(qapp):
    widget, plugin, session, _env, _done, _cancel, canvas = _frontend(qapp)
    start = session.snapshot()
    _pointer(canvas, "button_press_event", start.flux_half)
    _pointer(canvas, "motion_notify_event", start.flux_half + 0.5)
    assert widget.preview_active is True
    assert session.snapshot() == start
    # An external command wins during the drag, cancels the preview, and repaints.
    plugin.execute_command(
        session, "move_line", {"role": "half", "position": start.flux_half + 0.8}
    )
    remote = session.snapshot()
    assert widget.preview_active is False
    assert np.asarray(canvas.figure.axes[0].lines[0].get_xdata())[0] == pytest.approx(
        remote.flux_half
    )
    # A new drag commits the release coordinate, not the old preview snapshot.
    _pointer(canvas, "button_press_event", remote.flux_half)
    _pointer(canvas, "motion_notify_event", remote.flux_half + 0.2)
    assert session.snapshot() == remote
    _pointer(canvas, "button_release_event", remote.flux_half + 0.3)
    assert session.snapshot().flux_half == pytest.approx(
        remote.flux_half + 0.3, abs=0.03
    )
    assert widget.preview_active is False
    widget.teardown()
    widget.deleteLater()


def test_preview_cancels_on_escape_hide_and_finish_uses_committed_values(qapp):
    widget, _plugin, session, _env, completed, _cancel, canvas = _frontend(qapp)
    start = session.snapshot()
    _pointer(canvas, "button_press_event", start.flux_half)
    _pointer(canvas, "motion_notify_event", start.flux_half + 0.5)
    widget.keyPressEvent(
        QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
        )
    )
    assert widget.preview_active is False
    assert session.snapshot() == start
    _pointer(canvas, "button_press_event", start.flux_half)
    _pointer(canvas, "motion_notify_event", start.flux_half + 0.4)
    widget.hide()
    assert widget.preview_active is False
    assert session.snapshot() == start
    _button(widget, "Done").click()
    assert len(completed) == 1
    assert completed[0].flx_half == start.flux_half
    assert completed[0].figure is widget.figure
    _button(widget, "Done").click()
    assert len(completed) == 1
    widget.deleteLater()


def test_invalid_release_and_tab_hide_discard_preview_without_committing(qapp):
    widget, _plugin, session, _env, _done, _cancel, canvas = _frontend(qapp)
    start = session.snapshot()
    _pointer(canvas, "button_press_event", start.flux_half)
    _pointer(canvas, "motion_notify_event", start.flux_half + 0.5)
    _pointer(canvas, "button_release_event", 100.0)
    assert widget.preview_active is False
    assert session.snapshot() == start
    _pointer(canvas, "button_press_event", start.flux_half)
    _pointer(canvas, "motion_notify_event", start.flux_half + 0.5)
    widget.hide()
    assert session.snapshot() == start
    widget.teardown()
    widget.deleteLater()


def test_tab_switch_and_focus_loss_cancel_preview_without_canceling_analysis(qapp):
    widget, _plugin, session, _env, _done, cancelled, canvas = _frontend(qapp)
    stack = QStackedWidget()
    other = QWidget()
    stack.addWidget(widget)
    stack.addWidget(other)
    stack.setCurrentWidget(widget)
    stack.show()
    qapp.processEvents()
    canvas.draw()
    start = session.snapshot()

    _pointer(canvas, "button_press_event", start.flux_half)
    _pointer(canvas, "motion_notify_event", start.flux_half + 0.4)
    assert widget.preview_active
    qapp.sendEvent(widget, QFocusEvent(QEvent.Type.FocusOut))
    assert not widget.preview_active
    assert session.snapshot() == start

    _pointer(canvas, "button_press_event", start.flux_half)
    _pointer(canvas, "motion_notify_event", start.flux_half + 0.4)
    stack.setCurrentWidget(other)
    qapp.processEvents()
    assert not widget.preview_active
    assert session.snapshot() == start
    assert cancelled == []
    widget.teardown()
    stack.deleteLater()


def test_conjugate_swap_and_auto_align_single_flight_with_late_delivery(qapp):
    widget, plugin, session, env, completed, _cancel, canvas = _frontend(qapp)
    start = session.snapshot()
    checkbox = next(
        c for c in widget.findChildren(QCheckBox) if c.text() == "Conjugate Line"
    )
    assert checkbox.isChecked() is False
    checkbox.setChecked(True)
    assert session.snapshot().conjugate is True
    _pointer(canvas, "button_press_event", start.flux_half)
    _pointer(canvas, "motion_notify_event", start.flux_half + 0.4)
    assert session.snapshot().flux_int == start.flux_int
    _pointer(canvas, "button_release_event", start.flux_half + 0.4)
    moved = session.snapshot()
    assert moved.flux_int - start.flux_int == pytest.approx(0.4, abs=0.03)
    _button(widget, "Swap Lines").click()
    assert session.snapshot().flux_half == moved.flux_int

    _button(widget, "Auto Align").click()
    _button(widget, "Auto Align").click()
    assert len(env.pending) == 1
    # While calculation is pending, Done still completes with the current state.
    committed = session.snapshot()
    _button(widget, "Done").click()
    assert completed[0].flx_half == committed.flux_half
    compute, on_done, _on_error = env.pending[0]
    on_done(compute())
    assert len(completed) == 1
    assert plugin.plugin_id == "flux_pick"
    widget.deleteLater()


def test_auto_align_failure_keeps_session_editable_and_cancel_ignores_late_result(qapp):
    widget, plugin, session, env, _done, cancelled, _canvas = _frontend(qapp)
    start = session.snapshot()
    _button(widget, "Auto Align").click()
    compute, on_done, on_error = env.pending[0]
    on_error(RuntimeError("alignment failed"))
    assert session.snapshot() == start
    _button(widget, "Swap Lines").click()
    assert session.snapshot().flux_half == start.flux_int
    _button(widget, "Auto Align").click()
    assert len(env.pending) == 2
    _button(widget, "Cancel").click()
    assert cancelled == [True]
    on_done(compute())
    assert plugin.plugin_id == "flux_pick"
    widget.deleteLater()
