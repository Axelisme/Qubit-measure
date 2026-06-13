from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .base import BaseProgressBar

_pbar_factory: ContextVar[Callable[..., BaseProgressBar] | None] = ContextVar(
    "pbar_factory", default=None
)


def make_pbar(*args: Any, **kwargs: Any) -> BaseProgressBar:
    """Create a progress bar using the currently active factory.

    Defaults to TQDMProgressBar when no factory is set via use_pbar_factory().
    """
    factory = _pbar_factory.get()
    if factory is not None:
        return factory(*args, **kwargs)
    from .backend.tqdm import TQDMProgressBar

    return TQDMProgressBar(*args, **kwargs)


@contextmanager
def use_pbar_factory(
    factory: Callable[..., BaseProgressBar],
) -> Generator[None, None, None]:
    """Context manager that installs a custom pbar factory for its duration.

    Each thread/task has its own context snapshot, so nested use and concurrent
    workers are both safe without global mutation.
    """
    token = _pbar_factory.set(factory)
    try:
        yield
    finally:
        _pbar_factory.reset(token)
