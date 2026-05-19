from __future__ import annotations

from typing import Type

from .adapter import AbsExpAdapter


class Registry:
    """Maps experiment names to AbsExpAdapter subclasses."""

    def __init__(self) -> None:
        self._mapping: dict[str, Type[AbsExpAdapter]] = {}

    def register(self, name: str, adapter_cls: Type[AbsExpAdapter]) -> None:
        if name in self._mapping:
            raise ValueError(f"Adapter {name!r} is already registered")
        self._mapping[name] = adapter_cls

    def create(self, name: str) -> AbsExpAdapter:
        if name not in self._mapping:
            raise KeyError(
                f"Adapter {name!r} not found; available: {list(self._mapping)}"
            )
        return self._mapping[name]()

    def list_names(self) -> list[str]:
        return list(self._mapping)

    def has(self, name: str) -> bool:
        return name in self._mapping
