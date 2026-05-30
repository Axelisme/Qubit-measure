"""Tests for install_global_exception_hook.

Validates that sys.excepthook and threading.excepthook are replaced correctly,
KeyboardInterrupt is passed through without showing a dialog, and ordinary
exceptions trigger _show_error_dialog.

IMPORTANT: Every test that calls install_global_exception_hook() MUST restore
sys.excepthook and threading.excepthook to the pristine built-in values in its
finally block, not to the "previous" hook. This prevents hook chains from
accumulating across the session and avoids crashes during Qt teardown.
"""

from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock, patch


def _restore_hooks() -> None:
    """Unconditionally reset both hooks to the Python built-ins."""
    import _thread

    sys.excepthook = sys.__excepthook__
    # Python 3.9 does not expose threading.__excepthook__; the original is _thread._excepthook.
    threading.excepthook = _thread._excepthook  # type: ignore[attr-defined]


def test_install_replaces_sys_excepthook(qapp):  # noqa: ARG001
    from zcu_tools.gui.utils.error_handler import install_global_exception_hook

    try:
        install_global_exception_hook()
        assert sys.excepthook is not sys.__excepthook__
    finally:
        _restore_hooks()


def test_install_replaces_threading_excepthook(qapp):  # noqa: ARG001
    from zcu_tools.gui.utils.error_handler import install_global_exception_hook

    try:
        install_global_exception_hook()
        import _thread

        assert threading.excepthook is not _thread._excepthook  # type: ignore[attr-defined]
    finally:
        _restore_hooks()


def test_keyboard_interrupt_bypasses_dialog(qapp):  # noqa: ARG001
    import zcu_tools.gui.utils.error_handler as _mod
    from zcu_tools.gui.utils.error_handler import install_global_exception_hook

    # Reset first so install() wraps exactly one layer.
    _restore_hooks()
    try:
        install_global_exception_hook()
        with patch.object(_mod, "_show_error_dialog") as mock_show:
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
            mock_show.assert_not_called()
    finally:
        _restore_hooks()


def test_ordinary_exception_calls_show_dialog(qapp):  # noqa: ARG001
    import zcu_tools.gui.utils.error_handler as _mod
    from zcu_tools.gui.utils.error_handler import install_global_exception_hook

    _restore_hooks()
    try:
        install_global_exception_hook()
        with patch.object(_mod, "_show_error_dialog") as mock_show:
            exc_type = ValueError
            exc_val = ValueError("boom")
            sys.excepthook(exc_type, exc_val, None)
            mock_show.assert_called_once_with(exc_type, exc_val, None)
    finally:
        _restore_hooks()


def test_thread_keyboard_interrupt_bypasses_log(qapp):  # noqa: ARG001
    from zcu_tools.gui.utils.error_handler import install_global_exception_hook

    _restore_hooks()
    try:
        install_global_exception_hook()
        with patch("zcu_tools.gui.utils.error_handler.logger") as mock_log:
            args = threading.ExceptHookArgs(
                [KeyboardInterrupt, KeyboardInterrupt(), None, None]
            )
            threading.excepthook(args)
            mock_log.error.assert_not_called()
    finally:
        _restore_hooks()


def test_thread_ordinary_exception_logs_error(qapp):  # noqa: ARG001
    from zcu_tools.gui.utils.error_handler import install_global_exception_hook

    _restore_hooks()
    try:
        install_global_exception_hook()
        with patch("zcu_tools.gui.utils.error_handler.logger") as mock_log:
            exc_type = RuntimeError
            exc_val = RuntimeError("thread boom")
            exc_tb = MagicMock()  # fake traceback — avoids actual exception raising

            args = threading.ExceptHookArgs([exc_type, exc_val, exc_tb, None])
            threading.excepthook(args)
            mock_log.error.assert_called_once()
    finally:
        _restore_hooks()


def test_show_error_dialog_formats_exception(qapp):  # noqa: ARG001
    from zcu_tools.gui.utils.error_handler import _show_error_dialog

    with patch("zcu_tools.gui.utils.error_handler.QMessageBox") as mock_mb_cls:
        mock_mb = MagicMock()
        mock_mb_cls.return_value = mock_mb
        mock_mb_cls.Icon = MagicMock()
        mock_mb_cls.Icon.Critical = MagicMock()

        try:
            raise ValueError("test error")
        except ValueError:
            exc_type, exc_val, exc_tb = sys.exc_info()

        assert exc_type is not None and exc_val is not None
        _show_error_dialog(exc_type, exc_val, exc_tb)

        mock_mb.setText.assert_called_once()
        text_arg = mock_mb.setText.call_args[0][0]
        assert "ValueError" in text_arg
        assert "test error" in text_arg
        mock_mb.exec.assert_called_once()
