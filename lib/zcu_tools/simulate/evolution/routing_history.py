"""Caller-owned routing history for evolution acquires."""

from collections.abc import Mapping
from threading import Lock


class RoutingHistory:
    """Retain usage of distinct acquire contents across sources and acquires.

    Evolution namespaces each usage key by physical model identity. Sharing this
    object shares amortization only for matching physics. Storage grows with
    distinct content/key pairs, independently of engine residency. Serialize all
    acquires sharing a history, including across sources and Engines. The caller
    controls its lifetime with clear(); dispatch owns usage-key construction.
    """

    def __init__(self) -> None:
        """Create an empty history without retaining models or numerical results."""
        self._gate_usage: dict[bytes, int] = {}
        self._gate_contents: dict[bytes, set[bytes]] = {}
        self._lock = Lock()
        self._generation = 0

    def clear(self) -> None:
        """Discard usage and invalidate pre-clear preparations until evolved again.

        Call only after all acquires sharing this history have drained. Repeated
        clears leave an empty history. No models, densities, sampling RNG or
        simulation clocks are owned or reset here. This operation cannot fail
        under the serialized-acquire contract.
        """
        with self._lock:
            self._gate_usage.clear()
            self._gate_contents.clear()
            self._generation += 1

    @property
    def generation(self) -> int:
        """Return the monotonic clear count for Engine routing-fact validation.

        Routing snapshots compare this value before permitting preparation reuse.
        Reading it does not mutate usage or fail under the serialization contract.
        """
        with self._lock:
            return self._generation

    def projected_gate_usage(
        self, inventory: Mapping[bytes, int], *, content: bytes
    ) -> dict[bytes, int]:
        """Predict usage totals for evolution routing snapshots without mutation.

        inventory maps namespaced physics-key bytes to nonnegative row counts;
        content is evolution's complete numerical-input digest. Each key adds
        its count only if absent from that content. Only supplied keys appear
        in the returned detached mapping. Valid inventories do not raise domain
        errors. Serialize projection and the following record/evolve operation.
        """
        with self._lock:
            seen = self._gate_contents.get(content, set())
            return {
                key: self._gate_usage.get(key, 0) + (0 if key in seen else count)
                for key, count in inventory.items()
            }

    def record_gate_usage(
        self, inventory: Mapping[bytes, int], *, content: bytes
    ) -> dict[bytes, int]:
        """Record complete row inventories for evolution dispatch and route reuse.

        inventory maps namespaced physics-key bytes to nonnegative row counts;
        content is evolution's complete numerical-input digest, independent of
        cache state. Each key contributes its row count only on its first
        appearance in that content. Return a detached mapping of supplied keys
        to cumulative totals. Repeated contents do not increment totals, whether
        preparations hit, evict or are disabled. clear() releases content
        membership and counters. Negative counts raise ValueError before mutation.
        Callers serialize acquires sharing a history, even across Engines.
        """
        if any(count < 0 for count in inventory.values()):
            raise ValueError("gate usage counts must be nonnegative")
        with self._lock:
            seen = self._gate_contents.setdefault(content, set())
            for key, count in inventory.items():
                if key not in seen:
                    self._gate_usage[key] = self._gate_usage.get(key, 0) + count
                    seen.add(key)
            return {key: self._gate_usage[key] for key in inventory}
