from __future__ import annotations

import logging
from typing import Any

from .adapter import AbsExpAdapter

logger = logging.getLogger(__name__)


class Registry:
    """Maps experiment names to AbsExpAdapter subclasses."""

    def __init__(self) -> None:
        self._mapping: dict[str, type[AbsExpAdapter[Any, Any, Any, Any]]] = {}

    def register(
        self, name: str, adapter_cls: type[AbsExpAdapter[Any, Any, Any, Any]]
    ) -> None:
        logger.debug("register: name=%r adapter=%s", name, adapter_cls.__name__)
        if name in self._mapping:
            raise ValueError(f"Adapter {name!r} is already registered")
        self._mapping[name] = adapter_cls

    def create(self, name: str) -> AbsExpAdapter[Any, Any, Any, Any]:
        if name not in self._mapping:
            raise KeyError(
                f"Adapter {name!r} not found; available: {list(self._mapping)}"
            )
        return self._mapping[name]()

    def list_names(self) -> list[str]:
        return list(self._mapping)

    def has(self, name: str) -> bool:
        return name in self._mapping
