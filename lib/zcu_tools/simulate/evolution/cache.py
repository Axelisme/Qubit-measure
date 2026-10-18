"""Whole opaque allocation inventory and the model-only preparation LRU.

Calls, including handle release, require the owning Engine call lock or equivalent
external serialization. Inventory and reservation never change recency. Explicit
find/add operations touch/admit in caller order; planners must use canonical order.
Only model-only prepare/guard products may persist. Input bindings, densities and
endpoint results belong to one evolve call, not this cache. A hit or stored proof
never replaces the current row's guard.
"""

from collections import OrderedDict
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from .kernel import Prepared

type CacheKey = bytes
"""An opaque kernel prepare key, with no cache interpretation of its content."""

type CacheNamespace = tuple[str, str, float]
"""Kernel reuse name, whole-source model identity and numeric accuracy budget."""


@dataclass(frozen=True)
class PreparedPayload:
    """Whole read-only batch, with metadata supplied by its numerical owner.

    prepared is the kernel-owned allocation; nbytes is its nonnegative integer
    resident array charge, not process RSS. keys lists every entry in kernel order,
    even when a consumer needs only a subset. revision identifies the payload
    algorithm revision, defaulting to legacy for unversioned kernels. proof holds
    opaque evidence identifiers, not a guarantee of acceptance on another row.
    lifetime is model-only for persistent preparation or evolve for input-bound
    wrappers. The allocation and all metadata must remain unchanged while retained.
    """

    prepared: Prepared
    keys: tuple[bytes, ...]
    revision: str = "legacy"
    proof: tuple[str, ...] = ()
    lifetime: Literal["model-only", "evolve"] = "model-only"


@dataclass(frozen=True)
class CacheEntry:
    """Immutable inventory metadata with no payload reference.

    allocation_id is a cache-local, monotonically assigned allocation identity.
    namespace partitions kernel/source/budget reuse. keys covers the whole opaque
    batch, nbytes charges that entire allocation, revision names its algorithm,
    and proof carries owner-supplied evidence identifiers. Only descriptors returned
    by this cache can be reserved; a copied or stale descriptor is not a hit.
    """

    allocation_id: int
    namespace: CacheNamespace
    keys: tuple[bytes, ...]
    nbytes: int
    revision: str
    proof: tuple[str, ...]


@dataclass
class _Allocation:
    entry: CacheEntry
    payload: PreparedPayload
    holders: int = 0


class ConsumerReference:
    """Explicit strong reference to one whole allocation, independent of residency.

    Obtain from PrepareCache.reference or CacheReservation.acquire, not directly.
    release is idempotent and drops ownership; payload access after release raises
    RuntimeError. A context manager releases on exit, including exceptions. Ordinary
    Python references copied out of payload are caller-owned and are not tracked.
    """

    def __init__(self, release: Callable[[_Allocation], None], allocation: _Allocation):
        self._release_allocation: Callable[[_Allocation], None] | None = release
        self._allocation: _Allocation | None = allocation
        allocation.holders += 1

    @property
    def payload(self) -> PreparedPayload:
        """Return the whole batch, or raise RuntimeError after release."""
        if self._allocation is None:
            raise RuntimeError("consumer reference has been released")
        return self._allocation.payload

    def release(self) -> None:
        """Drop this reference without removing any resident cache entry."""
        if self._allocation is not None and self._release_allocation is not None:
            allocation = self._allocation
            self._allocation = None
            self._release_allocation(allocation)
            self._release_allocation = None

    def __enter__(self) -> "ConsumerReference":
        """Enter an unreleased reference, otherwise raise RuntimeError."""
        if self._allocation is None:
            raise RuntimeError("consumer reference has been released")
        return self

    def __exit__(self, *_: object) -> None:
        """Release ownership on normal or exceptional exit."""
        self.release()

    def __del__(self) -> None:
        self.release()


class CacheReservation:
    """Pin selected hits until consumers acquire independent references.

    Created only by PrepareCache.reserve. entries is the immutable, deduplicated
    selection in allocation admission order. nbytes is its whole-allocation charge,
    including keys the plan does not use. Both remain readable after release.
    acquire does not consume the reservation, so multiple consumers can share it.
    release drops all pins but never invalidates existing ConsumerReferences.
    Use a context manager or explicit release, including on cancellation/failure.
    """

    def __init__(
        self, release: Callable[[_Allocation], None], allocations: list[_Allocation]
    ):
        self._release_allocation: Callable[[_Allocation], None] | None = release
        self._allocations = {
            allocation.entry.allocation_id: allocation for allocation in allocations
        }
        self._entries = tuple(allocation.entry for allocation in allocations)
        for allocation in allocations:
            allocation.holders += 1

    @property
    def entries(self) -> tuple[CacheEntry, ...]:
        """Return selected metadata without exposing any payload."""
        return self._entries

    @property
    def nbytes(self) -> int:
        """Return the selected whole-allocation byte charge, once per allocation."""
        return sum(entry.nbytes for entry in self._entries)

    def acquire(self, entry: CacheEntry) -> ConsumerReference:
        """Retain one selected allocation, even after eviction or clear.

        Raise RuntimeError after reservation release, or ValueError if entry is
        not the original descriptor of one of this reservation's selected hits.
        """
        if self._release_allocation is None:
            raise RuntimeError("cache reservation has been released")
        allocation = self._allocations.get(entry.allocation_id)
        if allocation is None or allocation.entry is not entry:
            raise ValueError("entry is not selected by this reservation")
        return ConsumerReference(self._release_allocation, allocation)

    def release(self) -> None:
        """Drop all reservation pins exactly once; consumer ownership survives."""
        if self._release_allocation is not None:
            for allocation in self._allocations.values():
                self._release_allocation(allocation)
            self._allocations.clear()
            self._release_allocation = None

    def __enter__(self) -> "CacheReservation":
        """Enter an unreleased reservation, otherwise raise RuntimeError."""
        if self._release_allocation is None:
            raise RuntimeError("cache reservation has been released")
        return self

    def __exit__(self, *_: object) -> None:
        """Release all pins on normal or exceptional exit."""
        self.release()

    def __del__(self) -> None:
        self.release()


class PrepareCache:
    """Model-only LRU with independent reservation and consumer lifetimes.

    limit is a nonnegative integer resident byte budget (not bool); zero
    disables admission. Invalid limits raise ValueError. bytes
    counts resident allocations once. live_bytes counts allocations held by the
    explicit handles once, including resident pins, plus registered evolve-local
    workspaces at their current size, not raw find results or RSS.
    Eviction and clear only drop resident ownership; handles remain usable.
    Reusing one prepared object requires identical namespace and whole-batch
    metadata. It never splits an opaque batch or charges one allocation twice.
    """

    def __init__(self, limit: int):
        self.limit = _nonnegative_integer(limit)
        self._bytes = 0
        self._resident: OrderedDict[int, _Allocation] = OrderedDict()
        self._allocations: dict[int, _Allocation] = {}
        self._next_id = 0
        self._workspaces: list[Prepared] = []

    @property
    def bytes(self) -> int:
        """Resident whole-allocation array bytes, bounded by limit."""
        return self._bytes

    @property
    def live_bytes(self) -> int:
        """Whole bytes pinned by reservations/consumers, deduplicated across both.

        Registered evolve-local workspaces add their current numeric-array bytes.
        This can exceed limit and is not total live work or process RSS. Ordinary
        find results and caller-copied Python references are not tracked handles.
        """
        return sum(
            allocation.entry.nbytes
            for allocation in self._allocations.values()
            if allocation.holders
        ) + sum(
            _nonnegative_integer(workspace.nbytes) for workspace in self._workspaces
        )

    @contextmanager
    def track_workspace(self, workspace: Prepared) -> Generator[None]:
        """Charge growing evolve-local prepared arrays until scope exit.

        workspace supplies current nonnegative integer nbytes, including only
        arrays it owns. The Engine registers it under its call lock before
        workers start; workers may grow it with synchronized nbytes reads.
        live_bytes includes its current charge, independent of resident bytes
        and the LRU limit. It never enters inventory, find, or persistent reuse.
        Scope exit releases ownership even on cancellation or numerical failure.
        Register a given workspace only once per scope. Invalid initial charges
        or duplicate registration raise ValueError.
        """
        _nonnegative_integer(workspace.nbytes)
        if any(item is workspace for item in self._workspaces):
            raise ValueError("workspace is already tracked")
        self._workspaces.append(workspace)
        try:
            yield
        finally:
            self._workspaces = [
                item for item in self._workspaces if item is not workspace
            ]

    def inventory(self) -> tuple[CacheEntry, ...]:
        """Snapshot resident whole-batch metadata in LRU order without touching.

        Descriptors own no arrays. Later mutations do not change the snapshot.
        An evicted descriptor cannot create a new reservation, even if an older
        reservation still retains its allocation.
        """
        return tuple(allocation.entry for allocation in self._resident.values())

    def reserve(self, entries: Sequence[CacheEntry]) -> CacheReservation:
        """Atomically pin resident hits without touching LRU order.

        Duplicate entries charge once. Selection is sorted by admission identity.
        Raise ValueError for any stale, foreign or copied descriptor, acquiring
        nothing on failure. Empty selections produce a releasable empty handle.
        """
        selected: dict[int, _Allocation] = {}
        for entry in entries:
            allocation = self._resident.get(entry.allocation_id)
            if allocation is None or allocation.entry is not entry:
                raise ValueError("reservation requires original resident entries")
            selected[entry.allocation_id] = allocation
        return CacheReservation(
            self._release, [selected[key] for key in sorted(selected)]
        )

    def reference(
        self, namespace: CacheNamespace, payload: PreparedPayload
    ) -> ConsumerReference:
        """Track consumer ownership without admission, touch or eviction.

        Oversized/uncached model-only allocations are allowed. Invalid charges,
        evolve-local payloads or conflicting metadata raise ValueError, as in add.
        """
        return ConsumerReference(self._release, self._allocation(namespace, payload))

    def find(
        self, namespace: CacheNamespace, keys: set[CacheKey]
    ) -> list[PreparedPayload]:
        """Return whole batches intersecting keys and mark them recently used.

        Matches require namespace equality, not row guard acceptance. Returned
        ordinary strong references survive eviction but are not live_bytes handles.
        Matching batches are touched in their existing LRU order, as on the legacy
        active path. Partial key coverage still owns and charges the whole batch.
        """
        found: list[PreparedPayload] = []
        for identity, allocation in list(self._resident.items()):
            entry = allocation.entry
            if entry.namespace == namespace and keys.intersection(entry.keys):
                found.append(allocation.payload)
                self._resident.move_to_end(identity)
        return found

    def add(self, namespace: CacheNamespace, payload: PreparedPayload) -> None:
        """Admit/touch a whole model-only batch, evicting oldest residents first.

        Oversize and disabled admission do not evict existing residents. The same
        prepared object is charged once, even through another wrapper. Negative or
        noninteger (including bool) nbytes, evolve lifetime and conflicting metadata raise
        ValueError before eviction. Raw caller references survive cache removal.
        """
        allocation = self._allocation(namespace, payload)
        entry = allocation.entry
        if entry.nbytes > self.limit or self.limit == 0:
            self._discard_unheld(allocation)
            return
        if entry.allocation_id in self._resident:
            self._resident.move_to_end(entry.allocation_id)
            return
        while self._bytes + entry.nbytes > self.limit:
            _, evicted = self._resident.popitem(last=False)
            self._bytes -= evicted.entry.nbytes
            self._discard_unheld(evicted)
        self._resident[entry.allocation_id] = allocation
        self._bytes += entry.nbytes

    def clear(self) -> None:
        """Drop all residents, without invalidating reservation/consumer handles."""
        residents = tuple(self._resident.values())
        self._resident.clear()
        self._bytes = 0
        for allocation in residents:
            self._discard_unheld(allocation)

    def _allocation(
        self, namespace: CacheNamespace, payload: PreparedPayload
    ) -> _Allocation:
        if payload.lifetime != "model-only":
            raise ValueError("persistent cache requires model-only payloads")
        size = _nonnegative_integer(payload.prepared.nbytes)
        existing = self._allocations.get(id(payload.prepared))
        if existing is not None:
            entry = existing.entry
            if (
                entry.namespace != namespace
                or entry.keys != payload.keys
                or entry.nbytes != size
                or entry.revision != payload.revision
                or entry.proof != payload.proof
            ):
                raise ValueError("conflicting allocation metadata")
            return existing
        entry = CacheEntry(
            self._next_id,
            namespace,
            payload.keys,
            size,
            payload.revision,
            payload.proof,
        )
        self._next_id += 1
        allocation = _Allocation(entry, payload)
        self._allocations[id(payload.prepared)] = allocation
        return allocation

    def _discard_unheld(self, allocation: _Allocation) -> None:
        if (
            not allocation.holders
            and allocation.entry.allocation_id not in self._resident
        ):
            del self._allocations[id(allocation.payload.prepared)]

    def _release(self, allocation: _Allocation) -> None:
        allocation.holders -= 1
        self._discard_unheld(allocation)


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cache limit and prepared nbytes must be nonnegative integers")
    return value
