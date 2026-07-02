from __future__ import annotations

import logging
import sys


def log_current_exception(
    logger: logging.Logger,
    message: str = "Unhandled exception",
) -> None:
    """Log the active exception, including Pyro's remote traceback when present."""

    err_msg = sys.exc_info()[1]
    if err_msg is None:
        return

    pyro_traceback = getattr(err_msg, "_pyroTraceback", None)
    if isinstance(pyro_traceback, list):
        logger.error(
            "%s\n%s",
            message,
            "".join(str(line) for line in pyro_traceback),
            exc_info=True,
        )
        return

    logger.error(message, exc_info=True)
