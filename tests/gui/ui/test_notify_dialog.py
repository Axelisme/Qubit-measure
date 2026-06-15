"""Tests for NotifyUserDialog (Stage 4b, ADR-0025 §dialog-timeout).

Uses a mock Controller; qapp fixture (pytest-qt) ensures a QApplication
exists for widget construction. The dialog is non-modal so open() is never
called — we poke its slots directly to exercise each close path.

Covers:
  - Reply button → ctrl.reply_notify called with reply text
  - Dismiss button → ctrl.dismiss_notify called
  - closeEvent (window-X) → ctrl.dismiss_notify called
  - QTimer expiry → ctrl.timeout_notify called
  - _closed flag: second action after first is silently ignored
"""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import QLineEdit, QPushButton
from zcu_tools.gui.app.main.ui.notify_dialog import NotifyUserDialog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.reply_notify = MagicMock()
    ctrl.dismiss_notify = MagicMock()
    ctrl.timeout_notify = MagicMock()
    return ctrl


def _make_dlg(
    ctrl: MagicMock,
    token: int = 7,
    message: str = "test prompt",
    timeout: float = 0,  # 0 → timer never fires in synchronous tests
    qapp=None,
) -> NotifyUserDialog:
    # timeout=0 prevents the QTimer from firing while the test runs.
    dlg = NotifyUserDialog(token, message, timeout, ctrl)
    return dlg


def _find_btn(dlg: NotifyUserDialog, text: str) -> QPushButton:
    for btn in dlg.findChildren(QPushButton):
        if btn.text() == text:
            return btn
    raise AssertionError(f"Button {text!r} not found in dialog")


def _get_reply_edit(dlg: NotifyUserDialog) -> QLineEdit:
    edits = dlg.findChildren(QLineEdit)
    assert len(edits) == 1, f"Expected 1 QLineEdit, found {len(edits)}"
    return edits[0]


# ---------------------------------------------------------------------------
# Reply path
# ---------------------------------------------------------------------------


def test_reply_btn_calls_reply_notify(qapp) -> None:
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    _get_reply_edit(dlg).setText("my answer")
    _find_btn(dlg, "Reply").click()
    ctrl.reply_notify.assert_called_once_with(7, "my answer")


def test_reply_btn_empty_string_is_valid(qapp) -> None:
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    _get_reply_edit(dlg).setText("")
    _find_btn(dlg, "Reply").click()
    ctrl.reply_notify.assert_called_once_with(7, "")


def test_reply_btn_does_not_call_dismiss(qapp) -> None:
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    _find_btn(dlg, "Reply").click()
    ctrl.dismiss_notify.assert_not_called()
    ctrl.timeout_notify.assert_not_called()


# ---------------------------------------------------------------------------
# Dismiss path
# ---------------------------------------------------------------------------


def test_dismiss_btn_calls_dismiss_notify(qapp) -> None:
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    _find_btn(dlg, "Dismiss").click()
    ctrl.dismiss_notify.assert_called_once_with(7)


def test_dismiss_btn_does_not_call_reply(qapp) -> None:
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    _find_btn(dlg, "Dismiss").click()
    ctrl.reply_notify.assert_not_called()
    ctrl.timeout_notify.assert_not_called()


# ---------------------------------------------------------------------------
# Window-X close path (closeEvent → dismiss)
# ---------------------------------------------------------------------------


def test_close_event_calls_dismiss_notify(qapp) -> None:
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    dlg.close()  # triggers closeEvent
    ctrl.dismiss_notify.assert_called_once_with(7)


def test_close_event_does_not_call_reply_or_timeout(qapp) -> None:
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    dlg.close()
    ctrl.reply_notify.assert_not_called()
    ctrl.timeout_notify.assert_not_called()


# ---------------------------------------------------------------------------
# QTimer path
# ---------------------------------------------------------------------------


def test_timer_calls_timeout_notify(qapp) -> None:
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    # Invoke the timer slot directly (no need to start the timer).
    dlg._on_timer()
    ctrl.timeout_notify.assert_called_once_with(7)


def test_timer_does_not_call_dismiss_or_reply(qapp) -> None:
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    dlg._on_timer()
    ctrl.dismiss_notify.assert_not_called()
    ctrl.reply_notify.assert_not_called()


# ---------------------------------------------------------------------------
# _closed double-fire guard
# ---------------------------------------------------------------------------


def test_double_fire_guard_reply_then_timer(qapp) -> None:
    """Reply fires first; QTimer fires later — only reply reaches ctrl."""
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    _get_reply_edit(dlg).setText("first")
    _find_btn(dlg, "Reply").click()
    dlg._on_timer()  # second fire — must be silently ignored
    ctrl.reply_notify.assert_called_once_with(7, "first")
    ctrl.timeout_notify.assert_not_called()


def test_double_fire_guard_dismiss_then_close(qapp) -> None:
    """Dismiss button fires; then closeEvent fires — only dismiss reaches ctrl."""
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    _find_btn(dlg, "Dismiss").click()
    # closeEvent will try to fire again because close() is called inside _on_dismiss;
    # the _closed flag must block the second dismiss.
    ctrl.dismiss_notify.assert_called_once_with(7)


def test_double_fire_guard_timer_then_close(qapp) -> None:
    """Timer fires; then closeEvent fires (close() called from _on_timer)."""
    ctrl = _make_ctrl()
    dlg = _make_dlg(ctrl)
    dlg._on_timer()
    # simulate a second close (e.g. the OS sends another close event)
    dlg._on_dismiss()  # _closed is already True — should be silently ignored
    ctrl.dismiss_notify.assert_not_called()
    ctrl.timeout_notify.assert_called_once_with(7)
