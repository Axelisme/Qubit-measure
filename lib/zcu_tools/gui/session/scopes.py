"""Session-layer ambient scope helpers (ADR-0066).

``progress_ambient`` is the only helper here: it carries the pbar ContextVar
into a worker thread. It is session-layer (no Qt, no figure routing) so both
the session device service and the app-layer run/analyze services can use it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any


@contextmanager
def progress_ambient(
    pbar_factory: Callable[..., Any] | None,
) -> Iterator[None]:
    """Install ``pbar_factory`` as the per-worker pbar ContextVar for the
    current thread.  If ``pbar_factory`` is ``None`` this is a no-op.

    Session-layer: imports only ``progress_bar.interface`` (no Qt, no routing).
    Used by device-setup work thunks (session) and run work thunks (app).
    """
    if pbar_factory is None:
        with nullcontext():
            yield
        return

    from zcu_tools.progress_bar.interface import use_pbar_factory

    with use_pbar_factory(pbar_factory):
        yield
