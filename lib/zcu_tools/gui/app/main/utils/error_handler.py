import logging
import sys
import threading
from collections.abc import Callable
from types import TracebackType

logger = logging.getLogger(__name__)

# An unhandled-exception presenter: (exc_type, exc_value, exc_traceback) -> None.
ShowDialogFn = Callable[
    [type[BaseException], BaseException, TracebackType | None],
    None,
]


def install_global_exception_hook(
    show_dialog: ShowDialogFn,
) -> None:
    """Install process-level exception hooks with an injected presenter.

    Unexpected exceptions reaching ``sys.excepthook`` are presented and then
    delegated to the prior hook. Worker-thread exceptions are logged and delegated;
    they never call the presenter from a foreign thread. KeyboardInterrupt always
    bypasses presentation and logging.

    ``show_dialog`` is the required presentation port; Qt composition injects its
    QMessageBox adapter while tests pass a recording fake.
    """
    original_excepthook = sys.excepthook

    def _excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
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

        # A worker must not call an owner-thread presenter directly. Operation
        # executors marshal their expected terminal failures through their own ports;
        # this process-level fallback only records otherwise-unhandled failures.
        if args.exc_type and args.exc_value and args.exc_traceback:
            logger.error(
                "Unhandled thread exception",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        original_thread_excepthook(args)

    threading.excepthook = _thread_excepthook
    logger.info("Global exception hook installed.")
