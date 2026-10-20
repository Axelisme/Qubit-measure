"""Unified resident/pending ownership, with caller-only LRU mutation."""

from collections import deque
from collections.abc import Mapping
from threading import Lock

from .cache import CacheKey, CacheNamespace, PrepareCache, PreparedPayload


class ResidentAdmission:
    """One evolve's admission account, shared by prefixes and timeline workers.

    The caller supplies the complete candidate inventory before any workers.
    Whole resident hits stay in cache; unused residents transfer to a stale
    queue, keeping their bytes charged. Offers release stale ownership only as
    needed, under one lock. Workers never read or mutate the caller's LRU.
    Failed products yield before ordinary stale products and accepted offers.
    Active builds/apply scratch, including nonretained oversize batches, are
    separate from this bounded retention account. close runs after pool drain,
    committing offers and restoring surviving stale residents in caller order.
    """

    def __init__(
        self, cache: PrepareCache, inventory: Mapping[CacheNamespace, set[CacheKey]]
    ):
        self._cache = cache
        stale = cache.detach_unused(inventory)
        self._stale = deque(item for item in stale if not item.rejected)
        self._stale_rejected = deque(item for item in stale if item.rejected)
        self._stale_bytes = sum(item.payload.prepared.nbytes for item in stale)
        self._remaining = cache.limit - cache.bytes - self._stale_bytes
        self._protected = {id(payload.prepared) for payload in cache.payloads}
        self._offered: set[int] = set()
        self._pending: list[tuple[CacheNamespace, PreparedPayload]] = []
        self._rejected: list[tuple[CacheNamespace, PreparedPayload]] = []
        self._lock = Lock()

    @property
    def cache(self) -> PrepareCache:
        """Caller-only LRU for inventory lookups; never access it from workers."""
        return self._cache

    def _make_room(self, size: int) -> bool:
        """Release only reclaimable ownership; impossible/oversize offers do nothing."""
        if size <= self._remaining:
            return True
        rejected_bytes = sum(payload.prepared.nbytes for _, payload in self._rejected)
        if size > self._remaining + self._stale_bytes + rejected_bytes:
            return False
        while size > self._remaining:
            if self._rejected:
                _, payload = self._rejected.pop()
                self._remaining += payload.prepared.nbytes
                self._offered.remove(id(payload.prepared))
            else:
                queue = self._stale_rejected or self._stale
                item = queue.popleft()
                self._remaining += item.payload.prepared.nbytes
                self._stale_bytes -= item.payload.prepared.nbytes
        return True

    def retain_shared(
        self, namespace: CacheNamespace, payload: PreparedPayload
    ) -> None:
        """Reserve caller shared bindings, admitting only beside whole current hits.

        Call before worker offers. Unretainable shared/oversize batches stay
        owned by their active bindings and stream without evicting current hits.
        """
        identity = id(payload.prepared)
        if identity in self._protected or identity in self._offered:
            return
        size = payload.prepared.nbytes
        if self._make_room(size):
            self._remaining -= size
            self._protected.add(identity)
            self._cache.add(namespace, payload)
        else:
            # Shared bindings remain live throughout the schedule even without
            # retention; don't fill capacity with more pending local products.
            self._remaining -= size

    def replace(
        self,
        namespace: CacheNamespace,
        previous: PreparedPayload,
        payload: PreparedPayload,
    ) -> None:
        """Caller-only admission of an exact extension, charging only its growth.

        On insufficient capacity keep the old resident. The current guard may
        still borrow the extension as active nonretained work. Numerical domain
        and whole-key guarantees remain with the numerical owner/cache contract.
        """
        growth = payload.prepared.nbytes - previous.prepared.nbytes
        if payload.prepared.nbytes > self._cache.limit:
            return
        if self._cache.is_rejected(previous) and growth > self._remaining:
            # Exact-domain extension cannot promote a failed guard's priority.
            return
        if not self._make_room(max(0, growth)):
            return
        self._cache.replace(namespace, previous, payload)
        self._remaining -= growth
        self._protected.discard(id(previous.prepared))
        self._offered.discard(id(previous.prepared))
        self._protected.add(id(payload.prepared))

    def offer(self, namespace: CacheNamespace, payload: PreparedPayload) -> None:
        """Keep an accepted model-only worker product if unified capacity permits.

        Completion order affects retention, never numerical routing. A worker
        owns at most its current active payload(s) outside this account; input-
        bound wrappers must not be offered here. No caller LRU operation occurs.
        """
        identity = id(payload.prepared)
        with self._lock:
            if identity in self._protected:
                return
            if identity in self._offered:
                for index, (_, rejected) in enumerate(self._rejected):
                    if rejected.prepared is payload.prepared:
                        self._rejected.pop(index)
                        self._pending.append((namespace, payload))
                        break
                return
            size = payload.prepared.nbytes
            if not self._make_room(size) or self._remaining == 0:
                return
            self._remaining -= size
            self._offered.add(identity)
            self._pending.append((namespace, payload))

    def offer_rejected(
        self, namespace: CacheNamespace, payload: PreparedPayload
    ) -> None:
        """Hold a failed guard only in idle capacity, without evicting residents."""
        identity = id(payload.prepared)
        with self._lock:
            if identity in self._protected or identity in self._offered:
                return
            if payload.prepared.nbytes > self._remaining or self._remaining == 0:
                return
            self._remaining -= payload.prepared.nbytes
            self._offered.add(identity)
            self._rejected.append((namespace, payload))

    def commit(self, cache: PrepareCache) -> None:
        """Publish pending ownership in caller LRU after a prefix/timeline join.

        Capacity stays charged across schedules; prefixes and suffixes share
        this collector. Future consumers can find committed prefix products.
        """
        for namespace, payload in self._pending:
            cache.add(namespace, payload)
            self._protected.add(id(payload.prepared))
        self._pending.clear()
        for namespace, payload in self._rejected:
            cache.add(namespace, payload, rejected=True)
            self._protected.add(id(payload.prepared))
        self._rejected.clear()

    def close(self) -> None:
        """Caller-only final ownership transfer after pool drain, even on failure."""
        self.commit(self._cache)
        self._cache.restore([*self._stale_rejected, *self._stale])
        self._stale_rejected.clear()
        self._stale.clear()
