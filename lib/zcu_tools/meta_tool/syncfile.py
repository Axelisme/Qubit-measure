from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path

from typing_extensions import (
    Callable,
    Literal,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    cast,
)

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
                warnings.warn(
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
    def __init__(self, path: Optional[Union[str, Path]] = None, readonly=False) -> None:
        self._path = Path(path) if path is not None else None
        self._modify_time = 0
        self._dirty = False
        self._readonly = readonly

        if path is not None and Path(path).exists():
            self.load()

    @abstractmethod
    def _load(self, path: str) -> None: ...

    @abstractmethod
    def _dump(self, path: str) -> None: ...

    def _check_can_write(self) -> None:
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
        self._check_can_write()
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
