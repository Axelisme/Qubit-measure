"""App-layer ambient scope helper: figure routing + liveplot backend (ADR-0026 §2).

``figure_ambient`` installs the matplotlib routing ContextVar *and* the
``QtLivePlotBackend`` for the duration of a worker thunk.  Both are co-dependent
(the liveplot backend requires a routing container), so they are entered together
as a single facet driven by ``figure_container``.

Kept in the app layer because ``QtLivePlotBackend`` is Qt-specific.  Session
services must NOT import this module — use ``gui.session.scopes.progress_ambient``
instead (ADR-0026 §2 layer split).

``QtLivePlotBackend`` is imported lazily (inside the context manager) to keep
module-load time free of pyplot, preserving the gui import-clean invariant
(same reason as the old ``_entered`` in ``background.py``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any


@contextmanager
def figure_ambient(figure_container: Any | None) -> Iterator[None]:
    """Install ``routing_scope`` + ``QtLivePlotBackend`` on the current thread.

    These two are a co-dependent facet: routing tells pyplot/liveplot where to
    send figures; the backend actually renders them.  If ``figure_container`` is
    ``None`` this is a no-op.

    App-layer: lazily imports ``QtLivePlotBackend`` (Qt) only when a container
    is provided.  Must not be used by session-layer code.
    """
    if figure_container is None:
        with nullcontext():
            yield
        return

    from zcu_tools.gui.app.main.adapters.qt_liveplot_backend import QtLivePlotBackend
    from zcu_tools.gui.plotting import routing_scope
    from zcu_tools.liveplot.backend import set_liveplot_backend

    with ExitStack() as stack:
        stack.enter_context(routing_scope(figure_container))
        stack.enter_context(set_liveplot_backend(QtLivePlotBackend()))
        yield
