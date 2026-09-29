"""mtime-based two-way sync base class for persisted resources.

Instance state:

- ``_path``: backing file path; ``None`` means memory-only.
- ``_modify_time``: file mtime in nanoseconds at the last load or dump.
- ``_dirty``: in-memory data has changes not yet written back.
- ``_readonly``: write-back is forbidden.

``has_persistence`` is the public way to ask whether an object is bound to a
file; callers outside this module do not read ``_path``.

``sync()`` runs before every decorated read or write:

- memory-only objects return immediately;
- if the file exists, a dirty writable object dumps to disk, otherwise a file
  whose mtime is at least ``_modify_time`` is loaded again;
- if the file does not exist, a writable object dumps to create it.

Memory wins: when ``_dirty`` is set, the in-memory data overwrites the file even
if the file changed on disk. There is no file lock; two processes writing the
same file trigger a conflict warning and the local dirty data wins.

``auto_sync("read")`` syncs before the method; ``auto_sync("write")`` syncs
before and after it. The decorator accepts only ``SyncFile`` instance methods
and raises ``TypeError`` when the first argument is not a ``SyncFile``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T")


def auto_sync(
    time: Literal["read", "write"],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            sync_file = args[0]
            if not isinstance(sync_file, SyncFile):
                raise TypeError(
                    f"Expected first argument to be SyncFile, got {args} and {kwargs}"
                )
            sync_file = cast(SyncFile, sync_file)

            if time in ["read", "write"]:
                sync_file.sync()

            result = func(*args, **kwargs)

            if time in ["write"]:
                sync_file.sync()

            return result

        return wrapper

    return decorator


class SyncFile(ABC):
    def __init__(self, path: str | Path | None = None, readonly=False) -> None:
        self._path = Path(path) if path is not None else None
        self._modify_time = 0
        self._dirty = False
        self._readonly = readonly

        if path is not None and Path(path).exists():
            self.load()

    @property
    def has_persistence(self) -> bool:
        return self._path is not None

    @abstractmethod
    def _load(self, path: str) -> None: ...

    @abstractmethod
    def _dump(self, path: str) -> None: ...

    def require_writable(self) -> None:
        """Reject a write before callers modify or persist any content."""
        if self._readonly:
            raise RuntimeError(f"{self.__class__.__name__} is read-only")

    def update_modify_time(self) -> None:
        assert self._path is not None
        if self._path.exists():
            self._modify_time = self._path.stat().st_mtime_ns
        else:
            self._modify_time = 0

    def load(self) -> None:
        assert self._path is not None
        self._load(str(self._path))
        self.update_modify_time()
        self._dirty = False

    def dump(self) -> None:
        assert self._path is not None
        self.require_writable()
        self._dump(str(self._path))
        self.update_modify_time()
        self._dirty = False

    def sync(self) -> None:
        if self._path is None:
            return

        if self._path.exists():
            mtime = self._path.stat().st_mtime_ns
            if self._dirty and not self._readonly:
                if mtime > self._modify_time:
                    warnings.warn(
                        f"SyncFile conflict: {self._path} was modified by "
                        "another process; local changes kept, overwriting."
                    )
                self.dump()
            elif mtime >= self._modify_time:
                self.load()
        elif not self._readonly:
            self.dump()


__all__ = []
