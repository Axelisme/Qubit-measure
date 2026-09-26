from __future__ import annotations

import logging

from .adapter import AdapterCapabilities, ExpAdapterProtocol

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

    def validate(self) -> None:
        """Check constructibility and the framework contract before publication."""
        for name in self.list_names():
            if not name or name.strip() != name:
                raise ValueError(f"Invalid adapter name: {name!r}")
            adapter = self.create(name)
            if not isinstance(adapter, ExpAdapterProtocol):
                raise TypeError(f"Adapter {name!r} does not satisfy ExpAdapterProtocol")
            if not isinstance(adapter.capabilities, AdapterCapabilities):
                raise TypeError(f"Adapter {name!r} has invalid capabilities")

    def replace_from(self, candidate: Registry) -> None:
        """Publish a detached candidate while preserving this registry's identity.

        The caller validates the candidate first, on the State owner thread.
        Publication does not construct adapters or execute user code.
        """
        if candidate is self:
            raise ValueError("Cannot publish a registry into itself")
        self._mapping = candidate._mapping.copy()

    def clear(self) -> None:
        """Disable experiment creation after a failed destructive reload."""
        self._mapping = {}
