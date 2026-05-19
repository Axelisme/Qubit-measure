from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from .base import BaseProgressBar

_pbar_factory: Optional[Callable[..., BaseProgressBar]] = None


def make_pbar(*args: Any, **kwargs: Any) -> BaseProgressBar:
    """Create a progress bar using the currently active factory.

    Defaults to TQDMProgressBar when no factory is set via use_pbar_factory().
    """
    if _pbar_factory is not None:
        return _pbar_factory(*args, **kwargs)
    from .backend.tqdm import TQDMProgressBar

    return TQDMProgressBar(*args, **kwargs)


@contextmanager
def use_pbar_factory(
    factory: Callable[..., BaseProgressBar],
) -> Iterator[None]:
    """Context manager that installs a custom pbar factory for its duration.

    On exit the previous factory is restored, so nested use and test isolation
    are both safe.
    """
    global _pbar_factory
    previous = _pbar_factory
    _pbar_factory = factory
    try:
        yield
    finally:
        _pbar_factory = previous
