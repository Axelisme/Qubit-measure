from __future__ import annotations

import json
import warnings
from collections.abc import ItemsView, Mapping
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Any, Self

from zcu_tools.utils import format_obj

from .syncfile import SyncFile, auto_sync

_COMPLEX_TAG = "__complex__"
_STRING_TAG = "__metadict_string__"


def _looks_like_legacy_complex_string(value: str) -> bool:
    return "j" in value.strip().lower()


def _restore_complex_tag(raw_value: Any) -> complex:
    if not isinstance(raw_value, list) or len(raw_value) != 2:
        raise ValueError(
            f"Invalid MetaDict complex tag: expected [real, imag], got {raw_value!r}"
        )
    real, imag = raw_value
    if (
        isinstance(real, bool)
        or isinstance(imag, bool)
        or not isinstance(real, int | float)
        or not isinstance(imag, int | float)
    ):
        raise ValueError(
            f"Invalid MetaDict complex tag: real/imag must be JSON numbers, got {raw_value!r}"
        )
    return complex(float(real), float(imag))


def _restore_complex(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {_COMPLEX_TAG}:
            return _restore_complex_tag(obj[_COMPLEX_TAG])
        if set(obj) == {_STRING_TAG}:
            raw_string = obj[_STRING_TAG]
            if not isinstance(raw_string, str):
                raise ValueError(
                    f"Invalid MetaDict string escape tag: expected str, got {raw_string!r}"
                )
            return raw_string
        return {k: _restore_complex(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_complex(v) for v in obj]
    elif isinstance(obj, str) and _looks_like_legacy_complex_string(obj):
        try:
            restored = complex(obj)
        except ValueError:
            return obj
        warnings.warn(
            "Loading legacy MetaDict complex strings is deprecated; rewrite the "
            f"file so complex values use the {_COMPLEX_TAG!r} marker.",
            DeprecationWarning,
            stacklevel=2,
        )
        return restored
    else:
        return obj


def _dump_tagged_values(obj: Any) -> Any:
    if isinstance(obj, complex):
        return {_COMPLEX_TAG: [obj.real, obj.imag]}
    if isinstance(obj, dict):
        return {k: _dump_tagged_values(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_dump_tagged_values(v) for v in obj]
    if isinstance(obj, str) and _looks_like_legacy_complex_string(obj):
        return {_STRING_TAG: obj}
    return obj


class MetaDict(SyncFile):
    def __init__(
        self, json_path: str | Path | None = None, readonly: bool = False
    ) -> None:
        self._data = {}

        super().__init__(json_path, readonly=readonly)

    @auto_sync("read")
    def clone(self, dst_path: str | Path | None = None, readonly: bool = False) -> Self:
        if dst_path is not None and Path(dst_path).exists():
            raise FileExistsError(f"Destination path {dst_path} already exists")

        md = self.__class__(dst_path, readonly=readonly)
        md._data = deepcopy(self._data)
        if dst_path is not None:
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
            if not isinstance(file_data, dict):
                raise ValueError(
                    f"MetaDict file must contain a JSON object, got {type(file_data).__name__}"
                )
            restored = _restore_complex(file_data)
            try:
                self._validate_data_keys(restored)
            except (AttributeError, TypeError) as exc:
                raise ValueError(
                    "MetaDict file contains a protected or invalid data key"
                ) from exc
            self._data.clear()
            self._data.update(restored)

    def _dump(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data_to_dump = _dump_tagged_values(format_obj(self._data))

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_to_dump, f, indent=4, default=str)

    @classmethod
    def _is_protected(cls, name: str) -> bool:
        return name.startswith("_") or name in cls._protected_names()

    @classmethod
    def _protected_names(cls) -> set[str]:
        protected: set[str] = set()
        for base in cls.__mro__:
            protected.update(
                name
                for name in vars(base)
                if not (name.startswith("__") and name.endswith("__"))
            )
        return protected

    @classmethod
    def _ensure_data_key(cls, name: object) -> str:
        if not isinstance(name, str):
            raise TypeError(f"MetaDict keys must be str, got {type(name).__name__}")
        if cls._is_protected(name):
            raise AttributeError(f"{name!r} is a protected MetaDict attribute")
        return name

    @classmethod
    def _validate_data_keys(cls, data: Mapping[Any, Any]) -> None:
        for name in data:
            cls._ensure_data_key(name)

    def __getattr__(self, name: str) -> Any:
        if type(self)._is_protected(name):
            raise AttributeError(f"{name!r} is a protected MetaDict attribute")

        self.sync()
        if name not in self._data:
            raise AttributeError(f"MetaDict has no attribute {name}")
        return self._data[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if type(self)._is_protected(name):
            if name.startswith("_"):
                object.__setattr__(self, name, value)
                return
            raise AttributeError(f"{name!r} is a protected MetaDict attribute")

        if self._readonly:
            raise AttributeError("MetaDict is read-only")

        self.sync()
        self._data[name] = value
        self._dirty = True
        self.sync()

    def __delattr__(self, name: str) -> None:
        if type(self)._is_protected(name):
            if name.startswith("_"):
                object.__delattr__(self, name)
                return
            raise AttributeError(f"{name!r} is a protected MetaDict attribute")

        if self._readonly:
            raise AttributeError("MetaDict is read-only")

        self.sync()
        if name not in self._data:
            raise AttributeError(f"MetaDict has no attribute {name}")
        del self._data[name]
        self._dirty = True
        self.sync()

    def __str__(self) -> str:
        self.sync()
        return f"MetaDict({pformat(self._data)})"

    def __repr__(self) -> str:
        return self.__str__()

    @auto_sync("read")
    def keys(self):
        return self._data.keys()

    @auto_sync("read")
    def items(self) -> ItemsView[str, Any]:
        return self._data.items()

    @auto_sync("read")
    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    @auto_sync("write")
    def update(self, values: Mapping[str, Any] | None = None, /, **kwargs: Any) -> None:
        if self._readonly:
            raise AttributeError("MetaDict is read-only")

        updates: dict[str, Any] = {}
        if values is not None:
            updates.update(values)
        updates.update(kwargs)
        self._validate_data_keys(updates)
        if not updates:
            return

        self._data.update(updates)
        self._dirty = True

    def __dir__(self) -> list[str]:
        self.sync()
        public_protected = {
            name for name in type(self)._protected_names() if not name.startswith("_")
        }
        return sorted(set(self._data.keys()) | public_protected)
