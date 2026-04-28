from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from types import ModuleType

from typing_extensions import IO, Iterator, Optional, Union

import zcu_tools


@contextmanager
def debug_scope(
    module: Union[ModuleType, str, None] = None,
    level: int = logging.DEBUG,
    stream: Optional[IO[str]] = None,
) -> Iterator[None]:
    enable_debug(module, level, stream)
    try:
        yield
    finally:
        disable_debug(module)


def enable_debug(
    module: Union[ModuleType, str, None] = None,
    level: int = logging.DEBUG,
    stream: Optional[IO[str]] = None,
) -> None:
    """Enable debug logging for all loggers under the target module namespace."""
    if module is None:
        module = zcu_tools

    module_name = module.__name__ if not isinstance(module, str) else module

    if stream is None:
        stream = sys.stderr

    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        if getattr(handler, "_added_by_zcu_tools", False):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(formatter)
    handler._added_by_zcu_tools = True  # type: ignore
    logger.addHandler(handler)


def disable_debug(module: Union[ModuleType, str, None] = None) -> None:
    """Disable debug logging and remove handlers added by :func:`enable_debug`."""
    if module is None:
        module = zcu_tools

    module_name = module.__name__ if not isinstance(module, str) else module

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers[:]:
        if getattr(handler, "_added_by_zcu_tools", False):
            logger.removeHandler(handler)
            handler.close()
