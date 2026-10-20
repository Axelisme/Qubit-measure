"""Bounded retries for atomic publication under transient Windows file locks."""

import time
from collections.abc import Callable
from pathlib import Path

# Both Path.replace and os.replace are valid providers; their return is ignored.
type ReplaceFile = Callable[[Path, Path], Path | None]


def atomic_replace(
    source: Path, destination: Path, *, replace_fn: ReplaceFile | None = None
) -> None:
    """Atomically replace destination with an already closed source file.

    Paths and filesystem constraints follow Path.replace. By default call
    Path.replace; replace_fn may supply the same operation with two Path
    arguments and a Path or None return value (ignored).

    Retry only OSError.winerror 32 (sharing violation) or 33 (lock violation),
    with a 0.5-second monotonic budget including replace calls. Do not start
    retries after that deadline; an OS call or scheduler delay can overrun it.
    Other errors and the last lock error propagate unchanged. The caller owns
    directory creation, source cleanup and any error translation.
    """
    replace = Path.replace if replace_fn is None else replace_fn
    deadline = time.monotonic() + 0.5
    while True:
        try:
            replace(source, destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (32, 33):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(0.01, remaining))
            if time.monotonic() >= deadline:
                raise
