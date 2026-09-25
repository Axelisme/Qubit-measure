"""Tests for the Qt exception-dialog presenter."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


def test_show_error_dialog_formats_exception(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.ui.error_handler import show_error_dialog

    with patch("zcu_tools.gui.app.main.ui.error_handler.QMessageBox") as mock_mb_cls:
        mock_mb = MagicMock()
        mock_mb_cls.return_value = mock_mb
        mock_mb_cls.Icon = MagicMock()
        mock_mb_cls.Icon.Critical = MagicMock()

        try:
            raise ValueError("test error")
        except ValueError:
            exc_type, exc_val, exc_tb = sys.exc_info()

        assert exc_type is not None and exc_val is not None
        show_error_dialog(exc_type, exc_val, exc_tb)

        mock_mb.setText.assert_called_once()
        text_arg = mock_mb.setText.call_args[0][0]
        assert "ValueError" in text_arg
        assert "test error" in text_arg
        mock_mb.exec.assert_called_once()
