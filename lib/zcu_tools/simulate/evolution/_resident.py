"""Bounded worker-to-caller admission of model-only local products."""

from threading import Lock

from .cache import CacheNamespace, PrepareCache, PreparedPayload


class ResidentAdmission:
    """Retain local products within the capacity left by current protected work.

    limit is the resident byte budget. The caller must protect all current hits
    and newly built shared allocations before workers offer local products.
    Offers are thread-safe and charge each whole allocation once. They never
    call PrepareCache. After workers drain, the caller commits offers to the
    existing LRU, allowing stale residents to yield to the new workload.
    """

    def __init__(self, limit: int):
        self._remaining = limit
        self._protected: set[int] = set()
        self._offered: set[int] = set()
        self._pending: list[tuple[CacheNamespace, PreparedPayload]] = []
        self._lock = Lock()

    def protect(self, payload: PreparedPayload) -> None:
        """Reserve a whole current hit/shared allocation, once, before offers."""
        identity = id(payload.prepared)
        if identity not in self._protected:
            self._protected.add(identity)
            self._remaining = max(0, self._remaining - payload.prepared.nbytes)

    def offer(self, namespace: CacheNamespace, payload: PreparedPayload) -> None:
        """Keep a model-only allocation if it fits; otherwise let it stream.

        Worker completion order may affect retention, never numerical routing.
        Input-bound wrappers must not be offered here.
        """
        identity = id(payload.prepared)
        size = payload.prepared.nbytes
        with self._lock:
            if identity in self._protected or identity in self._offered:
                return
            if size > self._remaining or self._remaining == 0:
                return
            self._remaining -= size
            self._offered.add(identity)
            self._pending.append((namespace, payload))

    def commit(self, cache: PrepareCache) -> None:
        """Transfer pending ownership through caller-only LRU adds after join."""
        for namespace, payload in self._pending:
            cache.add(namespace, payload)
        self._pending.clear()
