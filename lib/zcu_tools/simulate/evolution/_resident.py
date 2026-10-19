"""Incremental resident/pending ownership with caller-only lookup and LRU touches."""

from collections.abc import Mapping
from threading import Lock

from .cache import CacheKey, CacheNamespace, PrepareCache, PreparedPayload


class ResidentAdmission:
    """One evolve's retention account, shared by prefixes and timeline workers.

    Complete-inventory hits are protected before workers start. Other residents
    stay in place until offers need room. One lock serializes pending offers and
    stale reclamation; workers never perform caller lookups or LRU touches.
    Failed products yield before ordinary stale products and accepted offers.
    Active build/apply scratch is outside this bounded retention account.
    close publishes offers and releases protection only after the pool drains.
    """

    def __init__(
        self, cache: PrepareCache, inventory: Mapping[CacheNamespace, set[CacheKey]]
    ):
        self._cache = cache
        self._protected = cache.protect(inventory)
        self._remaining = cache.limit - cache.charged_bytes
        self._rejected_bytes = 0
        self._offered: set[int] = set()
        self._pending: list[tuple[CacheNamespace, PreparedPayload]] = []
        self._rejected: list[tuple[CacheNamespace, PreparedPayload]] = []
        self._lock = Lock()

    @property
    def cache(self) -> PrepareCache:
        """Caller-only cache for inventory lookups; never look up from workers."""
        return self._cache

    def _make_room(self, size: int) -> bool:
        """Release reclaimable ownership; impossible/oversize offers do nothing."""
        if size <= self._remaining:
            return True
        if (
            size
            > self._remaining + self._cache.reclaimable_bytes + self._rejected_bytes
        ):
            return False
        while self._rejected and size > self._remaining:
            _, payload = self._rejected.pop()
            self._remaining += payload.charged_bytes
            self._rejected_bytes -= payload.charged_bytes
            self._offered.remove(id(payload.prepared))
        self._remaining += self._cache.reclaim(max(0, size - self._remaining))
        return True

    def retain_shared(
        self, namespace: CacheNamespace, payload: PreparedPayload
    ) -> None:
        """Reserve caller shared bindings without evicting whole current hits.

        Call before worker offers. Unretainable bindings stay active and stream
        without retention, reserving their live charge against further offers.
        """
        identity = id(payload.prepared)
        if identity in self._protected or identity in self._offered:
            return
        size = payload.charged_bytes
        if self._make_room(size):
            self._remaining -= size
            self._protected.add(identity)
            self._cache.add(namespace, payload)
        else:
            self._remaining -= size

    def replace(
        self,
        namespace: CacheNamespace,
        previous: PreparedPayload,
        payload: PreparedPayload,
    ) -> None:
        """Caller-only exact extension, charging growth without promoting failures.

        Insufficient capacity keeps the old resident; the extension may still
        serve as active nonretained work. Numerical domain guarantees belong to
        the numerical owner and whole-key validation belongs to the cache.
        """
        growth = payload.charged_bytes - previous.charged_bytes
        if payload.charged_bytes > self._cache.limit:
            return
        if self._cache.is_rejected(previous) and growth > self._remaining:
            return
        if not self._make_room(max(0, growth)):
            return
        self._cache.replace(namespace, previous, payload)
        self._remaining -= growth
        self._protected.discard(id(previous.prepared))
        self._offered.discard(id(previous.prepared))
        self._protected.add(id(payload.prepared))

    def offer(self, namespace: CacheNamespace, payload: PreparedPayload) -> None:
        """Retain accepted model-only worker products within unified capacity.

        Completion order changes retention, never routing. Worker active scratch
        is separate; input-bound wrappers must not be offered. No lookup or LRU
        touch occurs here; only reclaimable stale ownership may be removed.
        """
        identity = id(payload.prepared)
        with self._lock:
            if identity in self._protected:
                return
            if identity in self._offered:
                for index, (_, rejected) in enumerate(self._rejected):
                    if rejected.prepared is payload.prepared:
                        self._rejected.pop(index)
                        self._rejected_bytes -= rejected.charged_bytes
                        self._pending.append((namespace, payload))
                        break
                return
            size = payload.charged_bytes
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
            size = payload.charged_bytes
            if size > self._remaining or self._remaining == 0:
                return
            self._remaining -= size
            self._rejected_bytes += size
            self._offered.add(identity)
            self._rejected.append((namespace, payload))

    def commit(self, cache: PrepareCache) -> None:
        """Publish pending products after a prefix/timeline join, keeping charges.

        Prefixes and suffixes share this collector and can borrow committed
        products in subsequent caller lookups. Requires a drained worker pool.
        """
        for namespace, payload in self._pending:
            cache.add(namespace, payload)
            self._protected.add(id(payload.prepared))
        self._pending.clear()
        for namespace, payload in self._rejected:
            cache.add(namespace, payload, rejected=True)
            self._protected.add(id(payload.prepared))
        self._rejected.clear()
        self._rejected_bytes = 0

    def close(self) -> None:
        """Publish pending products and release protection after pool drain."""
        self.commit(self._cache)
        self._cache.release_protection()
