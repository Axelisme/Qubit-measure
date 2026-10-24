from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from zcu_tools.cfg_model import TemplateDef

from .adapter import AdapterCapabilities, ExpAdapterProtocol

logger = logging.getLogger(__name__)


def _validate_capabilities(capabilities: object, name: str) -> None:
    if not isinstance(capabilities, AdapterCapabilities):
        raise TypeError(f"Adapter {name!r} has invalid capabilities")


def _validate_contract(adapter: object, name: str) -> None:
    # Registered plugins can violate annotations at runtime.
    if not isinstance(adapter, ExpAdapterProtocol):
        raise TypeError(f"Adapter {name!r} does not satisfy ExpAdapterProtocol")
    _validate_capabilities(adapter.capabilities, name)


@dataclass(frozen=True)
class ExperimentResolution:
    """Catalog identity for a historical experiment string.

    adapter_name is the exact registered name, or None when unresolved.
    reason is None on success, otherwise a diagnostic explaining missing identity,
    no load-capable tag owner, or ambiguous tag owners.
    """

    adapter_name: str | None
    reason: str | None


class Registry:
    """Map adapter names and hold caller-injected persistent template definitions.

    templates is copied once at construction. The read-only templates property
    exposes that catalog for composition and form binding; no library or user
    modules are imported. Detached publication replaces both mappings atomically.
    """

    def __init__(self, *, templates: Mapping[str, TemplateDef] | None = None) -> None:
        self._mapping: dict[str, type[ExpAdapterProtocol]] = {}
        self._templates = MappingProxyType(dict(templates or {}))

    @property
    def templates(self) -> Mapping[str, TemplateDef]:
        """Return the immutable injected catalog; clearing adapters retains it."""
        return self._templates

    def register(
        self,
        name: str,
        adapter_cls: type[ExpAdapterProtocol],
    ) -> None:
        """Register an adapter class without executing its form factory.

        name is the public experiment ID; adapter_cls is constructed by create.
        Duplicate names raise ValueError without publishing the adapter. Form
        creation and template validation belong to Open/Reset, not registration.
        """
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

    def resolve_experiment(self, experiment: str | None) -> ExperimentResolution:
        """Resolve a historical name/tag without creating a tab or reading files.

        Exact registered names win, even without Load capability. Otherwise only
        load-capable adapters with the matching experiment_tag are candidates.
        Missing, unknown and ambiguous identities return a reason, never a guess.
        Adapter construction failures propagate.
        """
        if not experiment:
            return ExperimentResolution(None, "Missing experiment identity")
        if self.has(experiment):
            return ExperimentResolution(experiment, None)
        candidates: list[str] = []
        for name in self.list_names():
            adapter = self.create(name)
            if adapter.capabilities.load_data and adapter.experiment_tag == experiment:
                candidates.append(name)
        if len(candidates) == 1:
            return ExperimentResolution(candidates[0], None)
        if not candidates:
            return ExperimentResolution(
                None, f"No load-capable adapter for experiment {experiment!r}"
            )
        return ExperimentResolution(
            None,
            f"Multiple load-capable adapters for experiment {experiment!r}: "
            + ", ".join(sorted(candidates)),
        )

    def validate(self) -> None:
        """Check constructibility and the framework contract before publication."""
        for name in self.list_names():
            if not name or name.strip() != name:
                raise ValueError(f"Invalid adapter name: {name!r}")
            _validate_contract(self.create(name), name)

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
