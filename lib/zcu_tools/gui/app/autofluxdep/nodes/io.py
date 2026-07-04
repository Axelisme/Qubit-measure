"""Node I/O containers — the snapshot in / patch out contract.

A Node never touches the orchestrator's master container. Instead:

1. Before running a Node, the orchestrator projects a **Snapshot** — a read-only
   view holding ONLY what the Node declared, already resolved. Reading something
   undeclared raises ``KeyError`` — the snapshot is exactly "what this Node asked
   for", nothing more.
2. The Node returns a **Patch** — what it produced. The orchestrator validates it
   against the Node's declared outputs (else ``PatchContractError``) and merges
   it back, so the declarations are real contracts, not documentation.

There are two parallel spaces, with distinct access methods because they hold
different kinds of thing:

- **info values** — plain quantities that flow between Nodes (``qubit_freq``,
  ``fit_kappa``, ``qfw_factor`` …). Read ``snapshot[key]``; produce
  ``patch.set(key, value)``;
  declared via ``requires`` / ``optional`` / ``provides``.
- **modules** — cfg components (a ``PulseReadoutCfg`` …). A Node can read a named
  module (from the ml library OR produced by another Node, e.g. ro_optimize's
  tuned readout) and can produce one. Read ``snapshot.module(name)``; produce
  ``patch.set_module(name, mod)``; declared via ``requires_modules`` /
  ``optional_modules`` / ``provides_modules``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class PatchContractError(ValueError):
    """A Patch produced an info key / module the Node did not declare."""


class Snapshot(Mapping[str, Any]):
    """Read-only projection of exactly what a Node declared.

    The mapping interface (``snapshot[key]``) reads info values; ``module(name)``
    reads modules. Both raise ``KeyError`` for anything undeclared. Immutable.
    """

    __slots__ = ("_data", "_modules")

    def __init__(
        self,
        data: Mapping[str, Any],
        modules: Mapping[str, Any] | None = None,
    ) -> None:
        self._data = dict(data)
        self._modules = dict(modules) if modules is not None else {}

    # --- info values ---
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # --- modules ---
    def module(self, name: str) -> Any:
        """The resolved module ``name`` (raises ``KeyError`` if undeclared)."""
        return self._modules[name]

    __hash__ = None  # mutable contents → unhashable

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Snapshot):
            return self._data == other._data and self._modules == other._modules
        return NotImplemented

    def __repr__(self) -> str:
        return f"Snapshot(data={self._data!r}, modules={self._modules!r})"


class Patch:
    """What a Node produced this run — info values and/or modules.

    Filled by the Node (``set`` / ``set_module``), read by the orchestrator,
    which validates each side against ``provides`` / ``provides_modules`` and
    merges into the master container.
    """

    __slots__ = ("_data", "_modules")

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        modules: Mapping[str, Any] | None = None,
    ) -> None:
        self._data: dict[str, Any] = dict(data) if data is not None else {}
        self._modules: dict[str, Any] = dict(modules) if modules is not None else {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def set_module(self, name: str, module: Any) -> None:
        self._modules[name] = module

    def values(self) -> dict[str, Any]:
        return dict(self._data)

    def modules(self) -> dict[str, Any]:
        return dict(self._modules)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Patch):
            return self._data == other._data and self._modules == other._modules
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Patch(data={self._data!r}, modules={self._modules!r})"


def validate_patch(
    patch: Patch,
    provides: tuple[str, ...],
    provides_modules: tuple[str, ...] = (),
) -> None:
    """Raise ``PatchContractError`` if the patch produces anything undeclared."""
    extra = [k for k in patch.values() if k not in set(provides)]
    if extra:
        raise PatchContractError(
            f"Node produced undeclared info key(s) {extra}; provides={provides}"
        )
    extra_mod = [m for m in patch.modules() if m not in set(provides_modules)]
    if extra_mod:
        raise PatchContractError(
            f"Node produced undeclared module(s) {extra_mod}; "
            f"provides_modules={provides_modules}"
        )
