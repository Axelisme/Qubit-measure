from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
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
            assert isinstance(args[0], SyncTable)

            if time in ["before", "both"]:
                args[0].sync()

            result = func(*args, **kwargs)

            if time in ["after", "both"]:
                args[0].sync()

            return result

        return wrapper

    return decorator


class SyncTable(ABC):
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path is not None else None
        self._modify_time = 0

        if path is not None:
            self.sync()

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _dump(self) -> None: ...

    def update_modify_time(self) -> None:
        self._modify_time = self._path.stat().st_mtime_ns

    def load(self) -> None:
        assert self._path is not None
        self._load()
        self.update_modify_time()

    def dump(self) -> None:
        assert self._path is not None
        self._dump()
        self.update_modify_time()

    def sync(self) -> None:
        if self._path is None:
            return

        if self._path.exists():
            mtime = self._path.stat().st_mtime_ns
            if mtime > self._modify_time:
                self.load()

        self.dump()


class SampleTable(SyncTable):
    def __init__(self, csv_path: Optional[str] = None) -> None:
        self.samples = pd.DataFrame()
        super().__init__(csv_path)

    def _load(self) -> None:
        try:
            self.samples = pd.read_csv(self._path)
        except pd.errors.EmptyDataError:
            self.samples = pd.DataFrame()

    def _dump(self) -> None:
        # ensure directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.samples.to_csv(self._path, index=False)  # write csv

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


class MetaTableView:
    PROTECTED_KEYS = ["declare_subdict", "dump"]

    def __init__(self, data: dict, table: MetaTable) -> None:
        self._data = data
        self._table = table

    @classmethod
    def _is_protected(cls, name: str) -> bool:
        return (
            name.startswith("_")
            or name == "PROTECTED_KEYS"
            or name in cls.PROTECTED_KEYS
        )

    def __getattr__(self, name: str, /) -> Any:
        if MetaTableView._is_protected(name):
            return object.__getattr__(self, name)

        self._table.sync()
        value = self._data[name]
        if isinstance(value, dict):
            return MetaTableView(value, self._table)
        return value

    def __setattr__(self, name: str, value: Any, /) -> None:
        if MetaTableView._is_protected(name):
            object.__setattr__(self, name, value)
            return

        self._table.sync()
        self._data[name] = value
        self._table.sync()

    def __delattr__(self, name: str, /) -> None:
        if MetaTableView._is_protected(name):
            object.__delattr__(self, name)
            return

        self._table.sync()
        del self._data[name]
        self._table.sync()

    def __repr__(self) -> str:
        return f"MetaTableView({pformat(self._data)})"

    def declare_subdict(self, name: str) -> MetaTableView:
        if MetaTableView._is_protected(name):
            raise ValueError(f"{name} is protected name")

        sub_data = self._data.setdefault(name, {})
        if not isinstance(sub_data, dict):
            raise ValueError(f"{name} is already in the table, but not a dict")
        return MetaTableView(sub_data, self._table)

    def dump_json(self, path: str) -> None:
        dump_data = numpy2number(self._data)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=4)


class MetaTable(SyncTable):
    PROTECTED_KEYS = ["dump", "load", "sync", "clone"]

    def __init__(self, json_path: Optional[str] = None) -> None:
        self._data = {}
        self._view = MetaTableView(self._data, self)
        super().__init__(json_path)

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._data.clear()
                self._data.update(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            warnings.warn(f"Failed to load {self._path}, ignoring...")

    def _dump(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data_to_dump = numpy2number(self._data)

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data_to_dump, f, indent=4)

    @auto_sync("before")
    def clone(self, path: Optional[str] = None) -> MetaTable:
        new_table = MetaTable(path)
        new_table._data = deepcopy(self._data)
        new_table._view = MetaTableView(self._data, new_table)
        if path is not None:
            new_table.dump()
        return new_table

    @classmethod
    def _is_protected(cls, name: str) -> bool:
        return (
            name.startswith("_")
            or name == "PROTECTED_KEYS"
            or name in cls.PROTECTED_KEYS
        )

    def __getattr__(self, name: str) -> Any:
        if MetaTable._is_protected(name):
            return object.__getattr__(self, name)

        return getattr(self._view, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if MetaTable._is_protected(name):
            object.__setattr__(self, name, value)
            return

        setattr(self._view, name, value)

    def __delattr__(self, name: str) -> None:
        if MetaTable._is_protected(name):
            object.__delattr__(self, name)
            return

        delattr(self._view, name)

    def __repr__(self) -> str:
        return f"MetaTable({pformat(self._data)})"


if __name__ == "__main__":
    import os
    import tempfile

    import numpy as np

    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        test_json = os.path.join(tmpdir, "test_meta.json")
        # test_json = "test_meta.json"

        mt = MetaTable(test_json)
        mt.name = "Q1"
        config_mt = mt.declare_subdict("config")
        mt.config.power = 15
        config_mt.attenuation = 20
        subsub_mt = config_mt.declare_subdict("subconfig")
        subsub_mt.g = 10
        print(mt)
        print(config_mt)
        print(subsub_mt)

        mt.nest_dict = {
            "a": 1,
            "b": 2,
            "c": 3,
            "dd": {
                "d": 4,
                "e": 5,
                "f": 6,
                "k": np.arange(5),
            },
        }
        print(mt)
        mt.nest_dict.dump_json("nest_dict.json")
