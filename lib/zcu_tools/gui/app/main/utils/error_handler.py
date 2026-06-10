import logging
import sys
import threading
import traceback
from collections.abc import Callable
from typing import Any

from qtpy.QtWidgets import QMessageBox  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# An unhandled-exception presenter: (exc_type, exc_value, exc_traceback) -> None.
ShowDialogFn = Callable[[type, BaseException, Any], None]


def _show_error_dialog(
    exc_type: type, exc_value: BaseException, exc_traceback: Any
) -> None:
    """Show a critical QMessageBox for unhandled exceptions."""
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

    # Format the traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = "".join(tb_lines)

    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Unhandled Exception")
    msg_box.setText(f"{exc_type.__name__}: {exc_value}")
    msg_box.setDetailedText(tb_text)
    msg_box.exec()


def install_global_exception_hook(
    show_dialog: ShowDialogFn = _show_error_dialog,
) -> None:
    """Installs global exception hooks for PyQt.

    Catches both main thread sys.excepthook and threading.excepthook,
    displaying a QMessageBox for unexpected errors to ensure Fast Fail
    and minimal surprise.

    ``show_dialog`` is the unhandled-exception presenter (defaults to the
    QMessageBox one); injecting it lets tests pass a recording fake instead of
    patching the module-private symbol.
    """
    original_excepthook = sys.excepthook

    def _excepthook(
        exc_type: type, exc_value: BaseException, exc_traceback: Any
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            original_excepthook(exc_type, exc_value, exc_traceback)
            return
        show_dialog(exc_type, exc_value, exc_traceback)
        original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = _excepthook

    original_thread_excepthook = threading.excepthook

    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type and issubclass(args.exc_type, KeyboardInterrupt):
            original_thread_excepthook(args)
            return

        # Thread exceptions shouldn't directly show a Qt dialog unless using invokeMethod
        # but for simplicity we log them and let PyQt's event loop catch them if they cross boundary.
        # Alternatively, we could post an event to the main thread, but QThread already
        # routes run_failed signals. Here we just ensure it's logged cleanly.
        if args.exc_type and args.exc_value and args.exc_traceback:
            logger.error(
                "Unhandled thread exception",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        original_thread_excepthook(args)

    threading.excepthook = _thread_excepthook
    logger.info("Global exception hook installed.")
