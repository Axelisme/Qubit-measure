from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from zcu_tools.cfg_model import TemplateDef

from .adapter import Adapter

# Host payload types are erased only at entry to this registry.
RegisteredAdapter = Adapter[Any, Any, Any, Any, Any, Any]

logger = logging.getLogger(__name__)


class Registry:
    """Map adapter names and hold caller-injected persistent template definitions.

    templates is copied once at construction. The read-only templates property
    exposes that catalog for composition and form binding; no library or user
    modules are imported. Detached publication replaces both mappings atomically.
    """

    def __init__(self, *, templates: Mapping[str, TemplateDef] | None = None) -> None:
        self._mapping: dict[str, RegisteredAdapter] = {}
        self._templates = MappingProxyType(dict(templates or {}))

    @property
    def templates(self) -> Mapping[str, TemplateDef]:
        """Return the immutable injected catalog; clearing adapters retains it."""
        return self._templates

    def register(
        self,
        name: str,
        adapter: object,
    ) -> None:
        """Store adapter under name; return None without invoking factories.

        name must be nonempty, have no surrounding whitespace and be unique in
        this registry. adapter must be an Adapter value, not an adapter class.
        Invalid or duplicate names raise ValueError; a non-Adapter raises
        TypeError. Rejection leaves the mapping unchanged. Registration performs
        no I/O, form construction or capability callback checks. create(name)
        returns this same immutable value.
        """
        if not name or name.strip() != name:
            raise ValueError(f"Invalid adapter name: {name!r}")
        if name in self._mapping:
            raise ValueError(f"Adapter {name!r} is already registered")
        if not isinstance(adapter, Adapter):
            raise TypeError(f"Adapter {name!r} must be an Adapter value")
        # tolerated(cast): heterogeneous adapter registry; per-adapter generics are checked at each DEFINITION
        self._mapping[name] = cast(RegisteredAdapter, adapter)

    def create(self, name: str) -> RegisteredAdapter:
        if name not in self._mapping:
            raise KeyError(
                f"Adapter {name!r} not found; available: {list(self._mapping)}"
            )
        return self._mapping[name]

    def list_names(self) -> list[str]:
        return list(self._mapping)

    def has(self, name: str) -> bool:
        return name in self._mapping

    def validate(self) -> None:
        """Recheck names before publishing the registration-validated catalog."""
        for name in self.list_names():
            if not name or name.strip() != name:
                raise ValueError(f"Invalid adapter name: {name!r}")

    def replace_from(self, candidate: Registry) -> None:
        """Publish a detached candidate while preserving this registry's identity.

        The caller validates the candidate first, on the State owner thread.
        Publication does not construct adapters or execute user code.
        """
        if candidate is self:
            raise ValueError("Cannot publish a registry into itself")
        self._mapping = candidate._mapping.copy()
        self._templates = MappingProxyType(dict(candidate.templates))

    def clear(self) -> None:
        """Disable experiment creation after a failed destructive reload."""
        self._mapping = {}
