"""Qt presenter for unexpected process-level exceptions."""

from __future__ import annotations

import logging
import traceback
from types import TracebackType

from qtpy.QtWidgets import QMessageBox  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def show_error_dialog(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Unhandled Exception")
    msg_box.setText(f"{exc_type.__name__}: {exc_value}")
    msg_box.setDetailedText(tb_text)
    msg_box.exec()


__all__ = ["show_error_dialog"]
