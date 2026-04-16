import logging
import sys
from types import ModuleType
from typing import IO, Optional

import zcu_tools


def enable_debug(
    module: Optional[ModuleType] = None,
    level: int = logging.DEBUG,
    stream: Optional[IO[str]] = None,
) -> None:
    """Enable debug logging for all loggers under the target module namespace."""
    if module is None:
        module = zcu_tools
    if stream is None:
        stream = sys.stderr

    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(module.__name__)
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        if getattr(handler, "_added_by_zcu_tools", False):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(formatter)
    handler._added_by_zcu_tools = True  # type: ignore
    logger.addHandler(handler)


def disable_debug(module: Optional[ModuleType] = None) -> None:
    """Disable debug logging and remove handlers added by :func:`enable_debug`."""
    if module is None:
        module = zcu_tools

    logger = logging.getLogger(module.__name__)
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers[:]:
        if getattr(handler, "_added_by_zcu_tools", False):
            logger.removeHandler(handler)
            handler.close()
