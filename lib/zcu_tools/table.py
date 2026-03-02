from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Literal, Optional, TypeVar

import pandas as pd
from typing_extensions import ParamSpec

from zcu_tools.utils import numpy2number

P = ParamSpec("P")
T = TypeVar("T")


def auto_sync(
    time: Literal["after", "before", "both"],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            assert isinstance(args[0], SyncFile)

            if time in ["before", "both"]:
                args[0].sync()

            result = func(*args, **kwargs)

            if time in ["after", "both"]:
                args[0].sync()

            return result

        return wrapper

    return decorator


class SyncFile(ABC):
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path is not None else None
        self._modify_time = 0

        if path is not None:
            self.sync()

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

    def dump(self) -> None:
        assert self._path is not None
        self._dump(str(self._path))
        self.update_modify_time()

    def sync(self) -> None:
        if self._path is None:
            return

        if self._path.exists():
            mtime = self._path.stat().st_mtime_ns
            if mtime > self._modify_time:
                self.load()

        self.dump()


class SampleTable(SyncFile):
    def __init__(self, csv_path: Optional[str] = None) -> None:
        self.samples = pd.DataFrame()
        super().__init__(csv_path)

    def _load(self, path: str) -> None:
        try:
            self.samples = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            self.samples = pd.DataFrame()

    def _dump(self, path: str) -> None:
        # ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.samples.to_csv(path, index=False)  # write csv

    @auto_sync("both")
    def add_sample(self, **kwargs) -> None:
        if not kwargs:
            return
        new_df = pd.DataFrame([kwargs])
        self.samples = pd.concat([self.samples, new_df], ignore_index=True, sort=False)

    @auto_sync("both")
    def extend_samples(self, **kwargs) -> None:
        if not kwargs:
            return
        # pandas will raise if lengths mismatch
        new_df = pd.DataFrame(kwargs)
        self.samples = pd.concat([self.samples, new_df], ignore_index=True, sort=False)

    @auto_sync("both")
    def update_sample(self, idx: int, **kwargs) -> None:
        if not (0 <= idx < len(self.samples)):
            raise IndexError("sample index out of range")
        for key, value in kwargs.items():
            self.samples.loc[idx, key] = value

    @auto_sync("before")
    def get_samples(self) -> pd.DataFrame:
        return self.samples


def _restore_complex(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _restore_complex(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_complex(v) for v in obj]
    elif isinstance(obj, str):
        try:
            return complex(obj)
        except ValueError:
            return obj
    else:
        return obj


class MetaDict(SyncFile):
    _PROTECTED_KEYS = ["dump", "load", "sync", "update_modify_time", "clone"]

    def __init__(
        self, json_path: Optional[str] = None, read_only: bool = False
    ) -> None:
        self._data = {}
        self._read_only = read_only

        super().__init__(json_path)

    def _load(self, path: str) -> None:
        file_data = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            warnings.warn(f"Failed to load {self._path}, ignoring...")

        if file_data is not None:
            self._data.clear()
            self._data.update(_restore_complex(file_data))

    def _dump(self, path: str, force=True) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data_to_dump = numpy2number(self._data)

        mode = "w" if force else "x"

        with open(path, mode, encoding="utf-8") as f:
            json.dump(data_to_dump, f, indent=4, default=str)

    @classmethod
    def _is_protected(cls, name: str) -> bool:
        return name.startswith("_") or name in cls._PROTECTED_KEYS

    def __getattr__(self, name: str) -> Any:
        if MetaDict._is_protected(name):
            return object.__getattr__(self, name)  # type: ignore

        return self._data[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if MetaDict._is_protected(name):
            object.__setattr__(self, name, value)
            return

        if self._read_only:
            raise AttributeError("MetaDict is read-only")

        self._data[name] = value

    def __delattr__(self, name: str) -> None:
        if MetaDict._is_protected(name):
            object.__delattr__(self, name)
            return

        if self._read_only:
            raise AttributeError("MetaDict is read-only")

        del self._data[name]

    def __str__(self) -> str:
        return f"MetaTable({pformat(self._data)})"

    @auto_sync("before")
    def clone(self, path: Optional[str] = None, read_only=False) -> MetaDict:
        if path is not None:
            self._dump(path, force=False)
        return MetaDict(path, read_only=read_only)


if __name__ == "__main__":
    import os
    import tempfile

    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        test_json = os.path.join(tmpdir, "test_meta.json")
        # test_json = "test_meta.json"

        mt = MetaDict(test_json)
        mt.name = "Q1"
        print(mt)
