from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from types import ModuleType
from typing import IO

import zcu_tools

__all__ = ["enable_debug", "disable_debug", "debug_scope", "log_current_exception"]


class _ZcuToolsHandler(logging.StreamHandler[IO[str]]):
    """Handler added by enable_debug and removed by disable_debug."""


@contextmanager
def debug_scope(
    module: ModuleType | str | None = None,
    level: int = logging.DEBUG,
    stream: IO[str] | None = None,
) -> Generator[None]:
    enable_debug(module, level, stream)
    try:
        yield
    finally:
        disable_debug(module)


def enable_debug(
    module: ModuleType | str | None = None,
    level: int = logging.DEBUG,
    stream: IO[str] | None = None,
) -> None:
    """Enable debug logging for all loggers under the target module namespace."""
    if module is None:
        module = zcu_tools

    module_name = module.__name__ if not isinstance(module, str) else module

    target_stream: IO[str] = sys.stderr if stream is None else stream

    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        if isinstance(handler, _ZcuToolsHandler):
            logger.removeHandler(handler)
            handler.close()

    handler = _ZcuToolsHandler(stream=target_stream)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def disable_debug(module: ModuleType | str | None = None) -> None:
    """Disable debug logging and remove handlers added by :func:`enable_debug`."""
    if module is None:
        module = zcu_tools

    module_name = module.__name__ if not isinstance(module, str) else module

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers[:]:
        if isinstance(handler, _ZcuToolsHandler):
            logger.removeHandler(handler)
            handler.close()


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
