from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar

import pandas as pd
from typing_extensions import ParamSpec


def _sync(time: Literal["after", "before", "both"]) -> Callable:
    P = ParamSpec("P")
    T = TypeVar("T")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(self: SampleTable, *args, **kwargs) -> T:
            if time in ["before", "both"]:
                self.sync()

            result = func(self, *args, **kwargs)

            if time in ["after", "both"]:
                self.sync()
            return result

        return wrapper

    return decorator


class SampleTable:
    def __init__(self, csv_path: Optional[str] = None, auto_load: bool = True) -> None:
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.modify_time = 0
        self.samples = pd.DataFrame()

        # if set auto_load, load the file in initialization
        if auto_load:
            self.sync()

    def load(self) -> None:
        assert self.csv_path is not None

        try:
            self.samples = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError:
            self.samples = pd.DataFrame()
        self.modify_time = self.csv_path.stat().st_mtime_ns

    def dump(self) -> None:
        assert self.csv_path is not None
        # ensure directory exists
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # write csv
        self.samples.to_csv(self.csv_path, index=False)
        self.modify_time = self.csv_path.stat().st_mtime_ns

    def sync(self) -> None:
        if self.csv_path is None:
            return  # do nothing

        if self.csv_path.exists():
            mtime = self.csv_path.stat().st_mtime_ns
            if mtime > self.modify_time:
                self.load()

        self.dump()

    @_sync("both")
    def add_sample(self, **kwargs) -> None:
        if not kwargs:
            return
        new_df = pd.DataFrame([kwargs])
        self.samples = pd.concat([self.samples, new_df], ignore_index=True, sort=False)

    @_sync("both")
    def extend_samples(self, **kwargs) -> None:
        if not kwargs:
            return
        # pandas will raise if lengths mismatch
        new_df = pd.DataFrame(kwargs)
        self.samples = pd.concat([self.samples, new_df], ignore_index=True, sort=False)

    @_sync("both")
    def update_sample(self, idx: int, **kwargs) -> None:
        if not (0 <= idx < len(self.samples)):
            raise IndexError("sample index out of range")
        for key, value in kwargs.items():
            self.samples.loc[idx, key] = value

    @_sync("before")
    def get_samples(self) -> pd.DataFrame:
        return self.samples
