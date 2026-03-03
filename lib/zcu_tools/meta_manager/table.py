from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .syncfile import SyncFile, auto_sync


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

    @auto_sync("write")
    def add_sample(self, **kwargs) -> None:
        if not kwargs:
            return
        new_df = pd.DataFrame([kwargs])
        self.samples = pd.concat([self.samples, new_df], ignore_index=True, sort=False)
        self._dirty = True

    @auto_sync("write")
    def extend_samples(self, **kwargs) -> None:
        if not kwargs:
            return
        # pandas will raise if lengths mismatch
        new_df = pd.DataFrame(kwargs)
        self.samples = pd.concat([self.samples, new_df], ignore_index=True, sort=False)
        self._dirty = True

    @auto_sync("write")
    def update_sample(self, idx: int, **kwargs) -> None:
        if not (0 <= idx < len(self.samples)):
            raise IndexError("sample index out of range")
        for key, value in kwargs.items():
            self.samples.loc[idx, key] = value
        self._dirty = True

    @auto_sync("read")
    def get_samples(self) -> pd.DataFrame:
        return self.samples
