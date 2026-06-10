from __future__ import annotations

import logging

from .adapter import ExpAdapterProtocol

logger = logging.getLogger(__name__)


class Registry:
    """Maps experiment names to adapter classes implementing ExpAdapterProtocol."""

    def __init__(self) -> None:
        self._mapping: dict[str, type[ExpAdapterProtocol]] = {}

    def register(self, name: str, adapter_cls: type[ExpAdapterProtocol]) -> None:
        logger.debug("register: name=%r adapter=%s", name, adapter_cls.__name__)
        if name in self._mapping:
            raise ValueError(f"Adapter {name!r} is already registered")
        self._mapping[name] = adapter_cls

    def create(self, name: str) -> ExpAdapterProtocol:
        if name not in self._mapping:
            raise KeyError(
                f"Adapter {name!r} not found; available: {list(self._mapping)}"
            )
        return self._mapping[name]()

    def list_names(self) -> list[str]:
        return list(self._mapping)

    def has(self, name: str) -> bool:
        return name in self._mapping
