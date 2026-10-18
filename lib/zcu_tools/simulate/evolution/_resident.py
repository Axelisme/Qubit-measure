"""Bounded worker-to-caller admission of model-only local products."""

from threading import Lock

from .cache import CacheNamespace, PrepareCache, PreparedPayload


class ResidentAdmission:
    """Retain local products within the capacity left by current protected work.

    limit is the resident byte budget. The caller protects ordinary current hits
    and newly built shared allocations before workers offer local products.
    Failed guard products use only spare capacity and yield to accepted offers.
    Offers are thread-safe and charge each whole allocation once. They never
    call PrepareCache. After workers drain, the caller commits offers to the
    existing LRU, allowing stale residents to yield to the new workload.
    """

    def __init__(self, limit: int):
        self._remaining = limit
        self._protected: set[int] = set()
        self._offered: set[int] = set()
        self._pending: list[tuple[CacheNamespace, PreparedPayload]] = []
        self._rejected: list[tuple[CacheNamespace, PreparedPayload]] = []
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
            if identity in self._protected:
                return
            if identity in self._offered:
                for index, (_, rejected) in enumerate(self._rejected):
                    if rejected.prepared is payload.prepared:
                        self._rejected.pop(index)
                        self._pending.append((namespace, payload))
                        break
                return
            # Accepted local consumers can reclaim pending failed guards.
            while self._rejected and size > self._remaining:
                _, discarded = self._rejected.pop()
                self._remaining += discarded.prepared.nbytes
                self._offered.remove(id(discarded.prepared))
            if size > self._remaining or self._remaining == 0:
                return
            self._remaining -= size
            self._offered.add(identity)
            self._pending.append((namespace, payload))

    def offer_rejected(
        self, namespace: CacheNamespace, payload: PreparedPayload
    ) -> None:
        """Hold a failed guard only in spare pending capacity, below consumers."""
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
        """Transfer pending ownership through caller-only LRU adds after join."""
        for namespace, payload in self._pending:
            cache.add(namespace, payload)
        self._pending.clear()
        for namespace, payload in self._rejected:
            cache.add(namespace, payload, rejected=True)
        self._rejected.clear()
