"""Qt presenter for unexpected process-level exceptions."""

from __future__ import annotations

import logging
import traceback
from types import TracebackType

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMessageBox

from zcu_tools.gui.widgets import DialogRefStore

_dialog_refs = DialogRefStore()

logger = logging.getLogger(__name__)


def show_error_dialog(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    """Retain a non-modal owner-thread traceback report without delaying Qt exit."""
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    msg_box = QMessageBox()
    # Error reports are auxiliary windows, not a reason to keep the app running.
    msg_box.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Unhandled Exception")
    msg_box.setText(f"{exc_type.__name__}: {exc_value}")
    msg_box.setDetailedText(tb_text)
    _dialog_refs.open_transient(msg_box, modal=False)


__all__ = ["show_error_dialog"]
