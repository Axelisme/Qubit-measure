"""Indexed model-only whole-payload LRU and incremental retention ownership.

Caller operations require the Engine call lock and a drained worker pool.
During an admission session workers may only reclaim unprotected residents,
serialized by the admission account. Lookup and LRU touches remain caller-only.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from .kernel import Prepared

RESIDENT_METADATA_BYTES = 4608
"""Per-allocation retention estimate: measured 3998 bytes plus 15.3% headroom.

Includes Python/ndarray headers, keys, allocation wrappers and cache indexes in
wide-frequency RWA scans. This conservative estimate is not an RSS guarantee.
"""

type CacheKey = bytes
"""An opaque kernel prepare key, with no cache interpretation of its content."""

type CacheNamespace = tuple[str, str, float]
"""Kernel name, whole-source model identity and numeric accuracy budget."""


@dataclass(frozen=True)
class PreparedPayload:
    """Whole immutable batch supplied by its numerical owner.

    prepared owns a nonnegative integer array nbytes charge, not process RSS.
    keys lists every entry in kernel order, including unused entries. lifetime
    is model-only for persistent products or evolve for input-bound wrappers.
    The allocation and metadata must remain unchanged while retained.
    """

    prepared: Prepared
    keys: tuple[bytes, ...]
    lifetime: Literal["model-only", "evolve"] = "model-only"
    array_bytes: int | None = None
    """Previously measured immutable array charge, or None to read prepared once."""

    @cached_property
    def nbytes(self) -> int:
        """Validated array charge, computed once on this immutable wrapper."""
        return _nonnegative_integer(
            self.prepared.nbytes if self.array_bytes is None else self.array_bytes
        )

    @property
    def charged_bytes(self) -> int:
        """Array bytes plus the conservative per-allocation metadata estimate."""
        return self.nbytes + RESIDENT_METADATA_BYTES


@dataclass(slots=True)
class _Allocation:
    namespace: CacheNamespace
    payload: PreparedPayload
    order: int
    epoch: int
    rejected: bool


class PrepareCache:
    """Indexed whole-allocation LRU with a nonnegative integer byte limit.

    bytes counts retained arrays once, including protected residents.
    charged_bytes includes estimated metadata and is bounded by limit. Neither
    includes active scratch or represents process RSS. Zero disables admission.
    Invalid limits raise ValueError. Reusing a prepared object requires identical namespace and keys.
    Eviction releases cache ownership, not caller-held references. Complete
    inventory protection and its release visit only current hits, not history.
    """

    def __init__(self, limit: int):
        self.limit = _nonnegative_integer(limit)
        self._bytes = 0
        self._charged_bytes = 0
        self._current_bytes = 0
        self._order = 0
        self._epoch = 0
        self._active = False
        self._resident: OrderedDict[int, _Allocation] = OrderedDict()
        self._current: OrderedDict[int, _Allocation] = OrderedDict()
        self._rejected: OrderedDict[int, None] = OrderedDict()
        self._keys: dict[CacheNamespace, dict[bytes, set[int]]] = {}
        self._models: dict[tuple[str, float], dict[str, int]] = {}

    @property
    def bytes(self) -> int:
        """Retained whole-allocation array bytes, excluding metadata."""
        return self._bytes

    @property
    def charged_bytes(self) -> int:
        """Retained arrays plus estimated metadata, bounded by limit."""
        return self._charged_bytes

    @property
    def reclaimable_bytes(self) -> int:
        """Array/metadata charge not protected by the complete candidate inventory."""
        return self._charged_bytes - self._current_bytes

    @property
    def namespaces(self) -> frozenset[CacheNamespace]:
        """Resident namespaces without LRU touches; intended for diagnostics."""
        return frozenset(self._keys)

    def families(self, model: str, budget: float) -> frozenset[str]:
        """Resident families for one exact source/budget, without scanning history."""
        return frozenset(self._models.get((model, budget), ()))

    @property
    def payloads(self) -> tuple[PreparedPayload, ...]:
        """Whole retained references in LRU order, including rejected products."""
        identities = (*self._resident, *self._current)
        return tuple(
            self._allocation(identity).payload
            for identity in sorted(identities, key=self._lru_order)
        )

    def _matching(self, namespace: CacheNamespace, keys: set[CacheKey]) -> list[int]:
        index = self._keys.get(namespace, {})
        found: set[int] = set()
        for key in keys:
            found.update(index.get(key, ()))
        return sorted(found, key=self._lru_order)

    def _lru_order(self, identity: int) -> tuple[bool, int]:
        allocation = self._allocation(identity)
        # Restored failed products preceded ordinary stale products. An epoch
        # preserves that ordering lazily, without moving all historical entries.
        failed_stale = allocation.rejected and allocation.epoch < self._epoch
        return not failed_stale, allocation.order

    def _allocation(self, identity: int) -> _Allocation:
        allocation = self._current.get(identity)
        return allocation if allocation is not None else self._resident[identity]

    def protect(self, inventory: Mapping[CacheNamespace, set[CacheKey]]) -> set[int]:
        """Start one retention session and protect whole inventory intersections.

        Caller-only before workers start. Nonintersecting residents stay in
        place, reclaimable in rejected-first LRU order. Nested sessions fail.
        """
        if self._active:
            raise RuntimeError("resident protection session already active")
        self._active = True
        matches = {
            identity
            for namespace, keys in inventory.items()
            for identity in self._matching(namespace, keys)
        }
        for identity in sorted(matches, key=self._lru_order):
            allocation = self._resident.pop(identity)
            self._rejected.pop(identity, None)
            self._current[identity] = allocation
            self._current_bytes += allocation.payload.charged_bytes
        return matches

    def release_protection(self) -> None:
        """Finish after pool drain; current work becomes newer than stale residents."""
        self._epoch += 1
        for identity, allocation in self._current.items():
            allocation.epoch = self._epoch
            self._order += 1
            allocation.order = self._order
            self._resident[identity] = allocation
            if allocation.rejected:
                self._rejected[identity] = None
        self._current.clear()
        self._current_bytes = 0
        self._active = False

    def reclaim(self, size: int) -> int:
        """Release at least size stale bytes, rejected first, returning freed bytes.

        Only the admission lock may serialize worker calls. No caller lookup or
        LRU touch runs concurrently. Insufficient capacity changes nothing.
        """
        if size > self.reclaimable_bytes:
            return 0
        released = 0
        while released < size:
            identity = next(iter(self._rejected or self._resident))
            released += self._remove(identity).payload.charged_bytes
        return released

    def find(
        self, namespace: CacheNamespace, keys: set[CacheKey]
    ) -> list[PreparedPayload]:
        """Return intersecting whole batches and touch them in existing LRU order.

        Namespace equality never substitutes for row guard acceptance. Strong
        references survive eviction; partial key coverage charges the whole batch.
        """
        found = []
        for identity in self._matching(namespace, keys):
            allocation = self._allocation(identity)
            found.append(allocation.payload)
            self._touch(identity, allocation)
        return found

    def _touch(self, identity: int, allocation: _Allocation) -> None:
        self._order += 1
        allocation.order = self._order
        allocation.epoch = self._epoch
        owner = self._current if identity in self._current else self._resident
        owner.move_to_end(identity)
        if identity in self._rejected:
            self._rejected.move_to_end(identity)

    def is_rejected(self, payload: PreparedPayload) -> bool:
        """Whether this retained allocation has no accepted consumer yet."""
        identity = id(payload.prepared)
        allocation = self._current.get(identity) or self._resident.get(identity)
        return allocation is not None and allocation.rejected

    def add(
        self,
        namespace: CacheNamespace,
        payload: PreparedPayload,
        *,
        rejected: bool = False,
    ) -> None:
        """Admit whole immutable products, rejected before ordinary LRU eviction.

        Rejected products only fill idle space. Accepted reuse promotes them;
        rejected reuse never demotes normal products. Oversize/disabled admission
        does not evict. Invalid charge, evolve lifetime or conflicting metadata
        raises ValueError before eviction. Each wrapper reads its charge once.
        """
        if payload.lifetime != "model-only":
            raise ValueError("persistent cache requires model-only payloads")
        size = payload.charged_bytes
        identity = id(payload.prepared)
        existing = self._current.get(identity) or self._resident.get(identity)
        if existing is not None:
            if (
                existing.namespace != namespace
                or existing.payload.keys != payload.keys
                or existing.payload.nbytes != payload.nbytes
            ):
                raise ValueError("conflicting allocation metadata")
            if not rejected:
                existing.rejected = False
                self._rejected.pop(identity, None)
            self._touch(identity, existing)
            return
        if size > self.limit or self.limit == 0:
            return
        if rejected and self._charged_bytes + size > self.limit:
            return
        self.reclaim(max(0, self._charged_bytes + size - self.limit))
        if self._charged_bytes + size > self.limit:
            return
        self._order += 1
        allocation = _Allocation(namespace, payload, self._order, self._epoch, rejected)
        owner = self._current if self._active else self._resident
        owner[identity] = allocation
        if self._active:
            self._current_bytes += size
        elif rejected:
            self._rejected[identity] = None
        index = self._keys.setdefault(namespace, {})
        for key in payload.keys:
            index.setdefault(key, set()).add(identity)
        name, model, budget = namespace
        families = self._models.setdefault((model, budget), {})
        families[name] = families.get(name, 0) + 1
        self._bytes += payload.nbytes
        self._charged_bytes += size

    def _remove(self, identity: int) -> _Allocation:
        allocation = self._current.pop(identity, None)
        if allocation is not None:
            self._current_bytes -= allocation.payload.charged_bytes
        else:
            allocation = self._resident.pop(identity)
        self._rejected.pop(identity, None)
        namespace = allocation.namespace
        index = self._keys[namespace]
        for key in allocation.payload.keys:
            identities = index.get(key)
            if identities is not None:
                identities.discard(identity)
                if not identities:
                    del index[key]
        name, model, budget = namespace
        families = self._models[(model, budget)]
        families[name] -= 1
        if not families[name]:
            del families[name]
            del self._keys[namespace]
        if not families:
            del self._models[(model, budget)]
        self._bytes -= allocation.payload.nbytes
        self._charged_bytes -= allocation.payload.charged_bytes
        return allocation

    def replace(
        self,
        namespace: CacheNamespace,
        previous: PreparedPayload,
        payload: PreparedPayload,
    ) -> None:
        """Admit an owner's exact immutable extension with identical whole keys.

        Absent/oversize predecessors stay unchanged. Rejected extensions grow
        only into idle capacity. Validation fails before removal. Old caller
        references remain valid. Requires the Engine caller lock.
        """
        size = payload.charged_bytes
        if (
            previous.lifetime != "model-only"
            or payload.lifetime != "model-only"
            or previous.keys != payload.keys
        ):
            raise ValueError("replacement requires identical model-only whole keys")
        identity = id(previous.prepared)
        existing = self._current.get(identity) or self._resident.get(identity)
        if existing is None:
            return
        if existing.namespace != namespace or existing.payload.keys != previous.keys:
            raise ValueError("conflicting replacement namespace or keys")
        if payload.prepared is previous.prepared:
            self.add(namespace, payload, rejected=existing.rejected)
            return
        if (
            id(payload.prepared) in self._resident
            or id(payload.prepared) in self._current
        ):
            raise ValueError("replacement allocation is already resident")
        if size > self.limit or (
            existing.rejected
            and self._charged_bytes - existing.payload.charged_bytes + size > self.limit
        ):
            return
        self._remove(identity)
        self.add(namespace, payload, rejected=existing.rejected)

    def clear(self) -> None:
        """Release retained ownership without invalidating caller references."""
        self._resident.clear()
        self._current.clear()
        self._rejected.clear()
        self._keys.clear()
        self._models.clear()
        self._bytes = self._charged_bytes = self._current_bytes = 0
        self._active = False


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cache limit and prepared nbytes must be nonnegative integers")
    return value
