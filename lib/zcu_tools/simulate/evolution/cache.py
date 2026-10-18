"""Model-only whole-payload LRU.

Calls require the owning Engine call lock or equivalent external serialization.
Find/add operations touch/admit in caller order. Only model-only prepare/guard
products may persist. Input bindings, densities and endpoint results belong to
one evolve call. A matching key never replaces the current row's guard.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from .kernel import Prepared

type CacheKey = bytes
"""An opaque kernel prepare key, with no cache interpretation of its content."""

type CacheNamespace = tuple[str, str, float]
"""Kernel name, whole-source model identity and numeric accuracy budget."""


@dataclass(frozen=True)
class PreparedPayload:
    """Whole read-only batch supplied by its numerical owner.

    prepared is the kernel-owned allocation with a nonnegative integer nbytes
    array charge, not process RSS. keys lists every entry in kernel order, even
    when a consumer needs only a subset. lifetime is model-only for persistent
    preparation or evolve for input-bound wrappers. The allocation and its
    metadata must remain unchanged while retained.
    """

    prepared: Prepared
    keys: tuple[bytes, ...]
    lifetime: Literal["model-only", "evolve"] = "model-only"


@dataclass(frozen=True)
class _Allocation:
    namespace: CacheNamespace
    payload: PreparedPayload
    nbytes: int


class PrepareCache:
    """Model-only whole-allocation LRU.

    limit is a nonnegative integer resident byte budget (not bool); zero
    disables admission. Invalid limits raise ValueError. bytes counts resident
    allocations once, not active scheduler references or RSS. Reusing one
    prepared object requires identical namespace and whole-batch metadata.
    Eviction and clear release only cache ownership, not caller references.
    """

    def __init__(self, limit: int):
        self.limit = _nonnegative_integer(limit)
        self._bytes = 0
        self._resident: OrderedDict[int, _Allocation] = OrderedDict()
        self._rejected: OrderedDict[int, None] = OrderedDict()

    @property
    def bytes(self) -> int:
        """Resident whole-allocation array bytes, bounded by limit."""
        return self._bytes

    def find(
        self, namespace: CacheNamespace, keys: set[CacheKey]
    ) -> list[PreparedPayload]:
        """Return whole batches intersecting keys and mark them recently used.

        Matches require namespace equality, not row guard acceptance. Returned
        strong references survive eviction. Matching batches are touched in
        their existing LRU order. Partial key coverage still owns and charges
        the whole batch.
        """
        found: list[PreparedPayload] = []
        for identity, allocation in list(self._resident.items()):
            if allocation.namespace == namespace and keys.intersection(
                allocation.payload.keys
            ):
                found.append(allocation.payload)
                self._resident.move_to_end(identity)
                if identity in self._rejected:
                    self._rejected.move_to_end(identity)
        return found

    def is_rejected(self, payload: PreparedPayload) -> bool:
        """Whether this resident allocation has no accepted consumer yet."""
        return id(payload.prepared) in self._rejected

    def add(
        self,
        namespace: CacheNamespace,
        payload: PreparedPayload,
        *,
        rejected: bool = False,
    ) -> None:
        """Admit a whole batch, evicting rejected products before ordinary LRU.

        rejected=True only fills idle capacity; it never evicts any resident.
        Such a hit stays low priority until an accepted consumer adds it with
        rejected=False. Adding a rejected wrapper never demotes a normal batch.
        Oversize and disabled admission do not evict existing residents. The same
        prepared object is charged once, even through another wrapper. Negative
        or noninteger (including bool) nbytes, evolve lifetime and conflicting
        resident metadata raise ValueError before eviction. Raw caller references
        survive cache removal.
        """
        if payload.lifetime != "model-only":
            raise ValueError("persistent cache requires model-only payloads")
        size = _nonnegative_integer(payload.prepared.nbytes)
        identity = id(payload.prepared)
        existing = self._resident.get(identity)
        if existing is not None:
            if (
                existing.namespace != namespace
                or existing.payload.keys != payload.keys
                or existing.nbytes != size
            ):
                raise ValueError("conflicting allocation metadata")
            if not rejected:
                self._rejected.pop(identity, None)
            elif identity in self._rejected:
                self._rejected.move_to_end(identity)
            self._resident.move_to_end(identity)
            return
        if size > self.limit or self.limit == 0:
            return
        if rejected and self._bytes + size > self.limit:
            return
        # Separate priority index avoids scanning ordinary residents on eviction.
        while self._bytes + size > self.limit:
            if self._rejected:
                victim, _ = self._rejected.popitem(last=False)
                evicted = self._resident.pop(victim)
            else:
                _, evicted = self._resident.popitem(last=False)
            self._bytes -= evicted.nbytes
        self._resident[identity] = _Allocation(namespace, payload, size)
        if rejected:
            self._rejected[identity] = None
        self._bytes += size

    def replace(
        self,
        namespace: CacheNamespace,
        previous: PreparedPayload,
        payload: PreparedPayload,
    ) -> None:
        """Replace an immutable whole allocation with its owner's exact extension.

        The numerical owner guarantees unchanged old data and a superset domain.
        Both payloads must be model-only with identical whole-batch keys. An
        absent predecessor is a no-op; oversize replacements retain the old
        allocation. Rejected predecessors remain low priority and may grow only
        into idle capacity. Validation failures raise ValueError before removal.
        Existing strong references remain valid. Requires the Engine caller lock.
        """
        size = _nonnegative_integer(payload.prepared.nbytes)
        if (
            previous.lifetime != "model-only"
            or payload.lifetime != "model-only"
            or previous.keys != payload.keys
        ):
            raise ValueError("replacement requires identical model-only whole keys")
        identity = id(previous.prepared)
        existing = self._resident.get(identity)
        if existing is None:
            return
        if existing.namespace != namespace or existing.payload.keys != previous.keys:
            raise ValueError("conflicting replacement namespace or keys")
        if payload.prepared is previous.prepared:
            self.add(namespace, payload, rejected=self.is_rejected(previous))
            return
        if id(payload.prepared) in self._resident:
            raise ValueError("replacement allocation is already resident")
        rejected = self.is_rejected(previous)
        if size > self.limit or (
            rejected and self._bytes - existing.nbytes + size > self.limit
        ):
            return
        del self._resident[identity]
        self._rejected.pop(identity, None)
        self._bytes -= existing.nbytes
        self.add(namespace, payload, rejected=rejected)

    def clear(self) -> None:
        """Release all residents without invalidating caller-held references."""
        self._resident.clear()
        self._rejected.clear()
        self._bytes = 0


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cache limit and prepared nbytes must be nonnegative integers")
    return value
