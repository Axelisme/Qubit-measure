"""Tests for InteractiveAnalysisWidget (the measure-gui InteractiveHost).

Headless: a fake InteractiveSession is bound and the widget is driven by
clicking its rendered buttons and feeding it fake matplotlib mouse events.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from qtpy.QtWidgets import QPushButton
from zcu_tools.gui.app.main.ui.interactive_analysis import InteractiveAnalysisWidget


def _fake_session() -> MagicMock:
    session = MagicMock()
    session.actions.return_value = [("auto_align", "Auto Align"), ("swap", "Swap")]
    session.info_text.return_value = "half: 1\nint: 2"
    return session


def _buttons(widget) -> dict[str, QPushButton]:
    return {b.text(): b for b in widget.findChildren(QPushButton)}


def test_figure_property_is_a_real_figure(qapp):  # noqa: ARG001
    from matplotlib.figure import Figure

    w = InteractiveAnalysisWidget()
    assert isinstance(w.figure, Figure)
    w.deleteLater()


def test_bind_renders_one_button_per_action_plus_done(qapp):  # noqa: ARG001
    w = InteractiveAnalysisWidget()
    w.bind(_fake_session(), on_done=lambda: None)
    labels = set(_buttons(w))
    assert {"Auto Align", "Swap", "Done"} <= labels
    w.deleteLater()


def test_action_button_dispatches_its_id(qapp):  # noqa: ARG001
    session = _fake_session()
    w = InteractiveAnalysisWidget()
    w.bind(session, on_done=lambda: None)
    _buttons(w)["Auto Align"].click()
    session.invoke_action.assert_called_once_with("auto_align")
    w.deleteLater()


def test_done_calls_on_done_and_disables_buttons(qapp):  # noqa: ARG001
    fired: list[bool] = []
    w = InteractiveAnalysisWidget()
    w.bind(_fake_session(), on_done=lambda: fired.append(True))
    done = _buttons(w)["Done"]
    done.click()
    assert fired == [True]
    assert done.isEnabled() is False
    assert _buttons(w)["Auto Align"].isEnabled() is False
    w.deleteLater()


def test_canvas_events_forward_to_session(qapp):  # noqa: ARG001
    session = _fake_session()
    w = InteractiveAnalysisWidget()
    w.bind(session, on_done=lambda: None)
    in_axes = SimpleNamespace(inaxes=object(), xdata=1.5, ydata=4.2)
    out_axes = SimpleNamespace(inaxes=None, xdata=None, ydata=None)

    w._on_press(in_axes)
    w._on_move(in_axes)
    w._on_release(in_axes)
    w._on_press(out_axes)  # ignored — outside axes

    session.on_press.assert_called_once_with(1.5)
    session.on_move.assert_called_once_with(1.5)
    session.on_release.assert_called_once_with(1.5, 4.2)
    w.deleteLater()


def test_redraw_refreshes_info_from_session(qapp):  # noqa: ARG001
    session = _fake_session()
    w = InteractiveAnalysisWidget()
    w.bind(session, on_done=lambda: None)
    session.info_text.return_value = "updated info"
    w.redraw()
    assert w._info.text() == "updated info"
    w.deleteLater()


def test_run_background_marshals_result_to_main_thread(qapp):
    import time

    w = InteractiveAnalysisWidget()
    got: list[object] = []
    w.run_background(lambda: 42, on_done=got.append)
    # Pump the event loop until the worker's queued done-signal is delivered.
    for _ in range(100):
        if got:
            break
        qapp.processEvents()
        time.sleep(0.01)
    assert got == [42]
    w.deleteLater()
