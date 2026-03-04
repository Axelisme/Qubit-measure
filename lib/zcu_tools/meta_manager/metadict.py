from __future__ import annotations

import json
import warnings
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

from zcu_tools.utils import numpy2number

from .syncfile import SyncFile, auto_sync


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

    @auto_sync("read")
    def clone(self, dst_path: Optional[str] = None, read_only=False) -> MetaDict:
        if dst_path is not None and Path(dst_path).exists():
            raise FileExistsError(f"Destination path {dst_path} already exists")

        md = MetaDict(dst_path, read_only=read_only)
        md._data = deepcopy(self._data)
        md.update_modify_time()
        md.dump()

        return md

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

        self.sync()
        if name not in self._data:
            raise AttributeError(f"MetaDict has no attribute {name}")
        return self._data[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if MetaDict._is_protected(name):
            object.__setattr__(self, name, value)
            return

        if self._read_only:
            raise AttributeError("MetaDict is read-only")

        self.sync()
        self._data[name] = value
        self._dirty = True
        self.sync()

    def __delattr__(self, name: str) -> None:
        if MetaDict._is_protected(name):
            object.__delattr__(self, name)
            return

        if self._read_only:
            raise AttributeError("MetaDict is read-only")

        self.sync()
        del self._data[name]
        self._dirty = True
        self.sync()

    def __str__(self) -> str:
        self.sync()
        return f"MetaDict({pformat(self._data)})"

    def __repr__(self) -> str:
        return self.__str__()
