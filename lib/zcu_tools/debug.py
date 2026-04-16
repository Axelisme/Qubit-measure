import logging
import sys


def enable_debug(level: int = logging.DEBUG) -> None:
    """Enable debug logging for the ``zcu_tools`` logger hierarchy.

    Adds a ``StreamHandler`` to the ``zcu_tools`` root logger if none is
    present yet, so that debug messages are printed to stdout.
    """
    root_logger = logging.getLogger("zcu_tools")
    root_logger.setLevel(level)
    if not root_logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(levelname)s] %(name)s: %(message)s",
            )
        )
        root_logger.addHandler(handler)


def disable_debug() -> None:
    """Disable debug logging and remove handlers added by :func:`enable_debug`."""
    root_logger = logging.getLogger("zcu_tools")
    root_logger.setLevel(logging.WARNING)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
