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

    module_list = []
    module_stack = [module]
    while len(module_stack) > 0:
        mod = module_stack.pop()
        module_list.append(mod.__name__)
        for child in mod.__dict__.values():
            if isinstance(child, ModuleType) and child.__name__.startswith(mod.__name__):
                module_stack.append(child)

    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    for module_name in module_list:
        logger = logging.getLogger(module_name)
        logger.setLevel(level)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        handler = logging.StreamHandler(stream=stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def disable_debug() -> None:
    """Disable debug logging and remove handlers added by :func:`enable_debug`."""
    for logger in logging.root.manager.loggerDict.values():
        if not isinstance(logger, logging.Logger):
            continue
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
