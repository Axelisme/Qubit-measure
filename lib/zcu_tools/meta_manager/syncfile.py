from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path

from typing_extensions import Callable, Literal, Optional, ParamSpec, TypeVar, Union

P = ParamSpec("P")
T = TypeVar("T")


def auto_sync(
    time: Literal["read", "write"],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            assert isinstance(args[0], SyncFile)

            if time in ["read", "write"]:
                args[0].sync()

            result = func(*args, **kwargs)

            if time in ["write"]:
                args[0].sync()

            return result

        return wrapper

    return decorator


class SyncFile(ABC):
    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self._path = Path(path) if path is not None else None
        self._modify_time = 0
        self._dirty = False

        if path is not None and Path(path).exists():
            self.load()

    @abstractmethod
    def _load(self, path: str) -> None: ...

    @abstractmethod
    def _dump(self, path: str) -> None: ...

    def update_modify_time(self) -> None:
        assert self._path is not None
        self._modify_time = self._path.stat().st_mtime_ns

    def load(self) -> None:
        assert self._path is not None
        self._load(str(self._path))
        self.update_modify_time()
        self._dirty = False

    def dump(self) -> None:
        assert self._path is not None
        self._dump(str(self._path))
        self.update_modify_time()
        self._dirty = False

    def sync(self) -> None:
        if self._path is None:
            return

        if self._path.exists():
            mtime = self._path.stat().st_mtime_ns
            if self._dirty:
                self.dump()
            elif mtime >= self._modify_time:
                self.load()
        else:
            self.dump()
