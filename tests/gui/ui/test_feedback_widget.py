"""Tests for FeedbackPanel (docked collapsible panel, ADR-0025 §Stop-gating).

Uses a mock Controller so no Qt event loop or real services are needed for
the widget logic. The qapp fixture (provided by pytest-qt) ensures a
QApplication exists for widget construction.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import QLineEdit, QPushButton
from zcu_tools.gui.app.main.ui.feedback_widget import FeedbackPanel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctrl(*, can_cancel: bool = True) -> MagicMock:
    """Build a mock Controller with the minimal surface FeedbackPanel uses."""
    ctrl = MagicMock()
    ctrl.send_feedback = MagicMock(return_value=None)
    ctrl.can_cancel_active_operation = MagicMock(return_value=can_cancel)
    return ctrl


def _find_btn(widget: FeedbackPanel, text: str) -> QPushButton:
    for btn in widget.findChildren(QPushButton):
        if btn.text() == text:
            return btn
    raise AssertionError(f"Button {text!r} not found")


def _get_input(widget: FeedbackPanel) -> QLineEdit:
    inputs = widget.findChildren(QLineEdit)
    assert len(inputs) == 1, f"Expected 1 QLineEdit, found {len(inputs)}"
    return inputs[0]


# ---------------------------------------------------------------------------
# Send button
# ---------------------------------------------------------------------------


def test_send_calls_send_feedback_stop_false(qapp):
    ctrl = _make_ctrl()
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("hello agent")

    _find_btn(w, "Send").click()

    ctrl.send_feedback.assert_called_once_with("hello agent", stop=False)


def test_send_clears_input_after_click(qapp):
    ctrl = _make_ctrl()
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("some text")

    _find_btn(w, "Send").click()

    assert _get_input(w).text() == ""


def test_send_stop_calls_send_feedback_stop_true(qapp):
    ctrl = _make_ctrl(can_cancel=True)
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("stop now")

    _find_btn(w, "Send & Stop").click()

    ctrl.send_feedback.assert_called_once_with("stop now", stop=True)


def test_send_stop_clears_input_after_click(qapp):
    ctrl = _make_ctrl(can_cancel=True)
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("stop now")

    _find_btn(w, "Send & Stop").click()

    assert _get_input(w).text() == ""


# ---------------------------------------------------------------------------
# Empty input disables Send
# ---------------------------------------------------------------------------


def test_send_btn_disabled_when_input_blank(qapp):
    ctrl = _make_ctrl()
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("")

    send_btn = _find_btn(w, "Send")
    assert not send_btn.isEnabled()


def test_send_btn_enabled_when_input_non_blank(qapp):
    ctrl = _make_ctrl()
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("  msg  ")

    assert _find_btn(w, "Send").isEnabled()


def test_send_btn_disabled_for_whitespace_only(qapp):
    """strip() is applied — only-whitespace text counts as blank."""
    ctrl = _make_ctrl()
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("   ")

    assert not _find_btn(w, "Send").isEnabled()


# ---------------------------------------------------------------------------
# Stop-gating: refresh_gating() enable/disable "Send & Stop"
# ---------------------------------------------------------------------------


def test_stop_btn_disabled_when_no_cancel_hook(qapp):
    """refresh_gating() with can_cancel=False disables 'Send & Stop'."""
    ctrl = _make_ctrl(can_cancel=False)
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("msg")
    w.refresh_gating()

    assert not _find_btn(w, "Send & Stop").isEnabled()


def test_stop_btn_enabled_when_cancel_hook_present_and_text_nonempty(qapp):
    ctrl = _make_ctrl(can_cancel=True)
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("msg")
    w.refresh_gating()

    assert _find_btn(w, "Send & Stop").isEnabled()


def test_stop_btn_disabled_when_text_empty_even_if_hook_present(qapp):
    """Send & Stop also requires non-blank text (same rule as Send)."""
    ctrl = _make_ctrl(can_cancel=True)
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("")
    w.refresh_gating()

    assert not _find_btn(w, "Send & Stop").isEnabled()


def test_stop_btn_disabled_when_no_hook_and_text_nonempty(qapp):
    ctrl = _make_ctrl(can_cancel=False)
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("msg")

    # Gating is refreshed on text change; re-check after setText.
    assert not _find_btn(w, "Send & Stop").isEnabled()


# ---------------------------------------------------------------------------
# clear_input
# ---------------------------------------------------------------------------


def test_clear_input_wipes_text(qapp):
    ctrl = _make_ctrl()
    w = FeedbackPanel(ctrl)
    _get_input(w).setText("some pending message")

    w.clear_input()

    assert _get_input(w).text() == ""
