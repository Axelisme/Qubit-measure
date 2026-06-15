"""Process-wide serialization of matplotlib mathtext parsing (BUG-1).

matplotlib's mathtext parser is a class-level singleton
(``MathTextParser.parse`` reuses ``self.__class__._parser``, a pyparsing
grammar that is NOT thread-safe). The GUIs run domain ``analyze()`` on off-main
worker threads, and that work parses ``$...$`` titles (e.g. ``set_title`` +
``tight_layout``). When two workers — or a worker and a main-thread draw — parse
concurrently they corrupt the shared parser's mutable state and raise a
non-deterministic ``pyparsing.ParseException``.

The contained fix (ADR-0017) is a single process-wide lock around the public
parse entry point plus a main-thread prewarm so the parser's lazy
initialization never first happens under contention. This serializes mathtext
parsing across every thread/path without changing any plotting code or the
domain ``$...$`` title strings.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# Process-wide: the corrupted resource (``MathTextParser._parser``) is a single
# class attribute shared by every parser instance, so one lock for the whole
# process is both necessary and sufficient.
_MATHTEXT_LOCK = threading.Lock()

# Sentinel marking that ``MathTextParser.parse`` is already wrapped, so repeated
# install calls (three GUI entry points all install) do not stack wrappers.
_INSTALLED_FLAG = "_zcu_mathtext_lock_installed"


def install_mathtext_lock() -> None:
    """Wrap ``MathTextParser.parse`` to serialize parsing under a global lock.

    Idempotent: a sentinel attribute on the wrapped function prevents stacking
    wrappers when more than one GUI entry point installs the lock in the same
    process. Fast-fails if matplotlib's parse entry point is missing — that
    means the assumption this fix relies on no longer holds and silently
    skipping would let the race resurface.
    """
    import matplotlib.mathtext as mathtext

    original: Callable[..., Any] | None = getattr(
        mathtext.MathTextParser, "parse", None
    )
    if original is None:
        raise RuntimeError(
            "matplotlib.mathtext.MathTextParser.parse is missing; the mathtext "
            "thread-safety lock (BUG-1) targets that entry point and cannot be "
            "installed. matplotlib's mathtext API likely changed."
        )

    if getattr(original, _INSTALLED_FLAG, False):
        return

    def locked_parse(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Holding the lock across the whole parse keeps the singleton parser's
        # mutable pyparsing state consistent for the duration of one parse.
        with _MATHTEXT_LOCK:
            return original(self, *args, **kwargs)

    setattr(locked_parse, _INSTALLED_FLAG, True)
    mathtext.MathTextParser.parse = locked_parse  # type: ignore[method-assign]


def prewarm_mathtext() -> None:
    """Parse one ``$...$`` string on the calling thread to force lazy init.

    Call this on the Qt main thread at startup (after ``install_mathtext_lock``)
    so the parser's first-use construction happens single-threaded, never first
    under contention from a worker thread.
    """
    from matplotlib.mathtext import MathTextParser

    MathTextParser("path").parse("$x_0$")
